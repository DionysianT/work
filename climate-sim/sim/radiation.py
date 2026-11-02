"""外殼＋地面累積日射量 (kWh/m²)。

- 直達：太陽向量射線遮蔽。「逐時」= 每個日照小時一條射線；「分區」= 先把太陽位置
  累積到天空分區（Tregenza/Reinhart），速度快很多，誤差通常 < 數 %。
- 擴散：等向天空，天空分區射線遮蔽（Ladybug 用 Perez，斜面/立面可能差幾成）。
- 地面反射：albedo × GHI × (1 − cosβ)/2，不考慮遮蔽。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh

from .geometry import Scene
from .weather import Weather, hour_mask

TREGENZA = [(30, 6), (30, 18), (24, 30), (24, 42), (18, 54), (12, 66), (6, 78)]


def sky_patches(mf: int = 1):
    """Tregenza (mf=1, 145 區) / Reinhart (mf=2 → 577 區) 天空分區：回傳方向與立體角。"""
    dirs, om = [], []
    step = 12.0 / mf
    for bi, (n, _) in enumerate(TREGENZA):
        for sub in range(mf):
            a0 = np.radians(bi * 12 + sub * step)
            a1 = a0 + np.radians(step)
            ac = (a0 + a1) / 2
            nn = n * mf
            o = 2 * np.pi * (np.sin(a1) - np.sin(a0)) / nn
            for k in range(nn):
                A = 2 * np.pi * k / nn
                dirs.append([np.cos(ac) * np.sin(A), np.cos(ac) * np.cos(A), np.sin(ac)])
                om.append(o)
    a0 = np.radians(84)
    dirs.append([0, 0, 1.0])
    om.append(2 * np.pi * (1 - np.sin(a0)))
    return np.array(dirs), np.array(om)


@dataclass
class RadParams:
    months: tuple = tuple(range(1, 13))
    hours: tuple = (0, 24)
    albedo: float = 0.2
    bldg_grid: float = 1.5        # 外殼分析網格邊長 (m)
    ground_grid: float = 2.0      # 地面分析網格 (m)
    ground_margin: float = 30.0   # 地面分析範圍：模型外擴 (m)；0 = 不算地面
    direct_mode: str = "binned"   # 'hourly' 或 'binned'
    sky_mf: int = 1               # 天空分區細分：1=145、2=577


def _sun_vectors(w: Weather, msk: np.ndarray):
    import pvlib

    sp = pvlib.solarposition.get_solarposition(w.idx, w.lat, w.lon)
    alt = np.radians(sp["apparent_elevation"].values)
    az = np.radians(sp["azimuth"].values)
    up = msk & (alt > 0) & (w.dni > 0)
    sun = np.stack([np.cos(alt) * np.sin(az), np.cos(alt) * np.cos(az), np.sin(alt)], 1)[up]
    return sun, w.dni[up] / 1000.0


def _ray_engine(mesh):
    try:
        from trimesh.ray.ray_pyembree import RayMeshIntersector
        return RayMeshIntersector(mesh)
    except Exception:  # 沒有 embree 時退回純 Python（慢）
        return trimesh.ray.ray_triangle.RayMeshIntersector(mesh)


def analysis_mesh(main: trimesh.Trimesh, grid: float) -> trimesh.Trimesh:
    v, f = trimesh.remesh.subdivide_to_size(main.vertices, main.faces, max_edge=grid)
    bm = trimesh.Trimesh(v, f, process=False)
    bm.update_faces(bm.face_normals[:, 2] > -0.9)   # 去掉底面
    bm.remove_unreferenced_vertices()
    return bm


def compute(scene: Scene, w: Weather, p: RadParams, progress=None) -> dict:
    progress = progress or (lambda f, s: None)
    msk = hour_mask(w, p.months, p.hours)
    sun, sun_e = _sun_vectors(w, msk)
    ghi_sum = w.ghi[msk].sum() / 1000.0
    dhi_sum = w.dhi[msk].sum() / 1000.0
    sky, omega = sky_patches(p.sky_mf)
    L = dhi_sum / np.pi
    if p.direct_mode == "binned" and len(sun):
        k = np.argmax(sun @ sky.T, axis=1)
        e = np.bincount(k, sun_e, minlength=len(sky))
        keep = e > 0
        sun, sun_e = sky[keep], e[keep]

    ray = _ray_engine(scene.combined)

    bm = analysis_mesh(scene.main, p.bldg_grid)
    bpts = bm.triangles_center + bm.face_normals * 0.05
    bnrm = bm.face_normals

    gp = np.zeros((0, 3))
    GX = GY = None
    inside = np.zeros(0, bool)
    if p.ground_margin > 0:
        lo, hi = scene.main.bounds
        g = p.ground_grid
        xs = np.arange(lo[0] - p.ground_margin, hi[0] + p.ground_margin, g) + g / 2
        ys = np.arange(lo[1] - p.ground_margin, hi[1] + p.ground_margin, g) + g / 2
        GX, GY = np.meshgrid(xs, ys)
        gp = np.stack([GX.ravel(), GY.ravel(), np.full(GX.size, 0.05)], 1)
        inside = scene.contains(gp + [0, 0, 0.5])

    P = np.vstack([bpts, gp[~inside]])
    N = np.vstack([bnrm, np.tile([0, 0, 1.0], ((~inside).sum(), 1))])
    out = np.zeros(len(P))
    direct_out = np.zeros(len(P))
    # 每批射線數控制在 ~2e6 以內
    per_pt = max(len(sun), len(sky)) // 2 + 1
    chunk = int(np.clip(2_000_000 // per_pt, 200, 20000))
    for s in range(0, len(P), chunk):
        pp, nr = P[s:s + chunk], N[s:s + chunk]
        k = len(pp)
        direct = np.zeros(k)
        if len(sun):
            cosi = nr @ sun.T
            cand = np.argwhere(cosi > 0)
            if len(cand):
                hit = ray.intersects_any(pp[cand[:, 0]], sun[cand[:, 1]])
                val = np.where(hit, 0, cosi[cand[:, 0], cand[:, 1]] * sun_e[cand[:, 1]])
                direct = np.bincount(cand[:, 0], val, minlength=k)
        cosk = nr @ sky.T
        cand = np.argwhere(cosk > 0)
        hit = ray.intersects_any(pp[cand[:, 0]], sky[cand[:, 1]])
        val = np.where(hit, 0, L * omega[cand[:, 1]] * cosk[cand[:, 0], cand[:, 1]])
        diffuse = np.bincount(cand[:, 0], val, minlength=k)
        refl = ghi_sum * p.albedo * (1 - nr[:, 2]) / 2
        out[s:s + k] = direct + diffuse + refl
        direct_out[s:s + k] = direct
        progress(min(1.0, (s + k) / len(P)), f"日射計算 {s + k:,}/{len(P):,} 點")

    nb = len(bpts)
    gval = None
    if GX is not None:
        gval = np.full(len(gp), np.nan)
        gval[~inside] = out[nb:]
        gval = gval.reshape(GX.shape)
    return {"bv": bm.vertices, "bf": bm.faces, "bnrm": bnrm, "bval": out[:nb],
            "bdirect": direct_out[:nb], "GX": GX, "GY": GY, "gval": gval,
            "ghi": ghi_sum, "dhi": dhi_sum, "dni": w.dni[msk].sum() / 1000.0,
            "sun_hours": int(((w.dni > 0) & msk).sum()),
            "area": bm.area_faces, "source": w.source, "city": w.city}


ORIENT = ["北", "東北", "東", "東南", "南", "西南", "西", "西北"]


def summary(res: dict):
    """依朝向分類的面積加權平均日射量。"""
    import pandas as pd

    n, v, a = res["bnrm"], res["bval"], res["area"]
    cls = np.full(len(v), "其他斜面", dtype=object)
    cls[n[:, 2] > 0.9] = "屋頂/水平面"
    wall = np.abs(n[:, 2]) < 0.3
    az = np.degrees(np.arctan2(n[:, 0], n[:, 1])) % 360
    sec = (np.floor(az / 45 + 0.5) % 8).astype(int)
    cls[wall] = [f"{ORIENT[i]}向立面" for i in sec[wall]]
    df = pd.DataFrame({"類別": cls, "v": v, "a": a})
    g = df.groupby("類別")
    out = pd.DataFrame({"面積 m²": g["a"].sum(),
                        "平均 kWh/m²": g.apply(lambda x: np.average(x["v"], weights=x["a"]),
                                               include_groups=False),
                        "最小": g["v"].min(), "最大": g["v"].max()})
    order = ["屋頂/水平面"] + [f"{o}向立面" for o in ORIENT] + ["其他斜面"]
    out = out.reindex([o for o in order if o in out.index]).round(0)
    if res["gval"] is not None:
        gv = res["gval"][np.isfinite(res["gval"])]
        out.loc["地面"] = [float("nan"), round(gv.mean()), round(gv.min()), round(gv.max())]
    return out
