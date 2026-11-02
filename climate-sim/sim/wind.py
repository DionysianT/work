"""簡化 3D 風場：D3Q19 Lattice Boltzmann + Smagorinsky LES（純 numpy）。

只適合比較方案，不能當正式風環境評估：
- 網格粗（預設 3 m），雷諾數不是真實值，建物用 bounce-back；
- 入流為指數律剖面 U(z) = U10·(z/10)^α，頂面給定自由流，出流零梯度；
- 地面可選 free-slip（無摩擦，近地風速不會被過度削弱，但縫隙加速可能高估）
  或 no-slip（bounce-back，3 m 網格下近地風速會明顯偏低）。
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .geometry import Scene

C = np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1],
              [1, 1, 0], [-1, -1, 0], [1, -1, 0], [-1, 1, 0], [1, 0, 1], [-1, 0, -1], [1, 0, -1],
              [-1, 0, 1], [0, 1, 1], [0, -1, -1], [0, 1, -1], [0, -1, 1]])
W = np.array([1 / 3] + [1 / 18] * 6 + [1 / 36] * 12, np.float32)
OPP = np.array([0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17])
MIR = np.array([[j for j in range(19) if (C[j] == C[i] * [1, 1, -1]).all()][0] for i in range(19)])
CF = C.astype(np.float32)
PAIRS = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]


@dataclass
class WindParams:
    wd: float = 90.0            # 風的來向（度，北=0 順時針；90 = 東風）
    u10: float = 3.0            # 10 m 高參考風速 (m/s)
    alpha: float = 0.25         # 指數律指數（開闊 0.14、郊區 0.25、市區 0.33）
    dx: float = 3.0             # 網格 (m)
    upstream: float = 100.0     # 量體上游距離 (m)
    downstream: float = 200.0   # 下游距離 (m)
    lateral: float = 80.0       # 兩側距離 (m)
    height: float = 80.0        # 計算域高度 (m)
    steps: int = 0              # 0 = 自動（約 1.5 次流通時間）
    avg_frac: float = 0.33      # 最後多少比例的步數取時間平均
    ground: str = "free-slip"   # 'free-slip' 或 'no-slip'
    cs: float = 0.17            # Smagorinsky 常數
    ulat10: float = 0.055       # lattice 速度 ↔ U10（越小越穩定、越慢）
    tau0: float = 0.505


def domain(scene: Scene, p: WindParams):
    th = np.radians(p.wd + 180)                       # 氣流前進方向
    ex = np.array([np.sin(th), np.cos(th)])
    ey = np.array([-ex[1], ex[0]])
    v = np.vstack([m.vertices[:, :2] for m in scene.parts])
    s, t = v @ ex, v @ ey
    x0 = s.min() - p.upstream
    y0 = t.min() - p.lateral
    nx = int(np.ceil((s.max() - s.min() + p.upstream + p.downstream) / p.dx))
    ny = int(np.ceil((t.max() - t.min() + 2 * p.lateral) / p.dx))
    nz = int(np.ceil(p.height / p.dx)) + 1
    xs = x0 + (np.arange(nx) + 0.5) * p.dx
    ys = y0 + (np.arange(ny) + 0.5) * p.dx
    zs = (np.arange(nz) - 0.5) * p.dx                 # k=0 為地面實體層，k=1 中心 = dx/2
    return ex, ey, xs, ys, zs


def cell_count(scene: Scene, p: WindParams) -> int:
    _, _, xs, ys, zs = domain(scene, p)
    return len(xs) * len(ys) * len(zs)


def auto_steps(nx: int, p: WindParams) -> int:
    return int(max(1200, 1.5 * nx / p.ulat10))


def _feq(rho, u):
    cu = np.einsum("qd,d...->q...", CF, u)
    usq = (u ** 2).sum(0)
    return (W.reshape((19,) + (1,) * (u.ndim - 1)) * rho *
            (1 + 3 * cu + 4.5 * cu ** 2 - 1.5 * usq)).astype(np.float32)


def run(scene: Scene, p: WindParams, progress=None, should_stop=None) -> dict:
    progress = progress or (lambda f, s: None)
    ex, ey, xs, ys, zs = domain(scene, p)
    NX, NY, NZ = len(xs), len(ys), len(zs)
    I, J, K = np.meshgrid(xs, ys, zs, indexing="ij")
    Wp = np.stack([I * ex[0] + J * ey[0], I * ex[1] + J * ey[1], K], -1).reshape(-1, 3)
    solid = np.zeros(len(Wp), bool)
    q = Wp[:, 2] > 0
    progress(0.0, f"建立網格 {NX}×{NY}×{NZ} = {NX * NY * NZ:,} 格")
    solid[q] = scene.contains(Wp[q])
    solid = solid.reshape(NX, NY, NZ)
    solid[:, :, 0] = True
    del I, J, K, Wp

    zphys = np.maximum(zs, 0.5)
    prof = (p.ulat10 * (zphys / 10.0) ** p.alpha).astype(np.float32)
    prof[0] = 0
    uin = np.zeros((3, 1, NY, NZ), np.float32)
    uin[0] = prof[None, None, :]
    u0 = np.zeros((3, NX, NY, NZ), np.float32)
    u0[0] = prof[None, None, :]
    u0[:, solid] = 0
    f = _feq(np.ones((NX, NY, NZ), np.float32), u0)
    del u0
    fin_in = _feq(np.ones((1, NY, NZ), np.float32), uin)
    ftop = np.broadcast_to(fin_in[:, :, :, -1], (19, NX, NY)).copy()

    steps = p.steps or auto_steps(NX, p)
    n_avg_target = max(1, int(steps * p.avg_frac))
    usum = np.zeros((3, NX, NY, NZ), np.float64)
    n_avg = 0
    t0 = time.time()
    for it in range(steps):
        rho = f.sum(0)
        u = np.einsum("qd,q...->d...", CF, f) / rho
        u[:, solid] = 0
        fneq = f - _feq(rho, u)
        Q = np.zeros_like(rho)
        for a, b in PAIRS:
            pab = np.einsum("q,q...->...", CF[:, a] * CF[:, b], fneq)
            Q += pab * pab * (1 if a == b else 2)
        tau = 0.5 * (p.tau0 + np.sqrt(p.tau0 ** 2 + 18 * np.sqrt(2) * p.cs ** 2 * np.sqrt(Q) / rho))
        fpost = f - fneq / tau
        fpost[:, solid] = f[OPP][:, solid]                 # bounce-back
        if p.ground == "free-slip":
            fpost[:, :, :, 0] = f[MIR][:, :, :, 0]         # 地面鏡面反射
        for i in range(19):
            f[i] = np.roll(fpost[i], shift=tuple(C[i]), axis=(0, 1, 2))
        f[:, 0:1] = fin_in
        f[:, -1] = f[:, -2]
        f[:, :, :, -1] = ftop
        if it >= steps - n_avg_target:
            usum += u
            n_avg += 1
        if it % 50 == 0 or it == steps - 1:
            umax = float(np.abs(u[0]).max())
            if not np.isfinite(umax) or umax > 0.4:
                raise RuntimeError(f"第 {it} 步數值發散（umax={umax:.3g}）。"
                                   "請加大網格、降低 lattice 速度或增加 Cs。")
            el = time.time() - t0
            eta = el / (it + 1) * (steps - it - 1)
            progress((it + 1) / steps, f"LBM 第 {it + 1}/{steps} 步，已 {el:.0f}s，約剩 {eta:.0f}s")
            if should_stop and should_stop():
                raise InterruptedError("使用者中止")
    umean = (usum / n_avg) * (p.u10 / p.ulat10)
    return {"u": umean.astype(np.float32), "solid": solid, "xs": xs, "ys": ys, "zs": zs,
            "ex": ex, "ey": ey, "wd": p.wd, "u10": p.u10, "alpha": p.alpha, "dx": p.dx,
            "steps": steps, "seconds": time.time() - t0, "ground": p.ground}


def ref_speed(res: dict, z: float) -> float:
    """入流剖面在高度 z 的風速（自由流參考值）。"""
    return res["u10"] * (max(z, 0.5) / 10.0) ** res["alpha"]


def level_index(res: dict, z: float) -> int:
    return int(np.clip(np.argmin(np.abs(res["zs"] - z)), 1, len(res["zs"]) - 1))


def plan_slice(res: dict, z: float, extent=None, step: float | None = None):
    """在高度 z 取水平切面，重取樣到世界座標規則網格（X=東、Y=北）。

    回傳 X, Y (1D)、speed、ux, uy（世界座標分量）、solid mask。
    """
    from scipy.interpolate import RegularGridInterpolator

    k = level_index(res, z)
    u = res["u"][:, :, :, k]
    ex, ey = res["ex"], res["ey"]
    xs, ys = res["xs"], res["ys"]
    step = step or res["dx"] / 2
    if extent is None:
        cx = np.array([xs[[0, -1, 0, -1]], ys[[0, 0, -1, -1]]])
        wx = cx[0] * ex[0] + cx[1] * ey[0]
        wy = cx[0] * ex[1] + cx[1] * ey[1]
        extent = (wx.min(), wx.max(), wy.min(), wy.max())
    X = np.arange(extent[0], extent[1], step)
    Y = np.arange(extent[2], extent[3], step)
    GX, GY = np.meshgrid(X, Y)
    s = GX * ex[0] + GY * ex[1]
    t = GX * ey[0] + GY * ey[1]
    pts = np.stack([s.ravel(), t.ravel()], 1)
    kw = dict(bounds_error=False, fill_value=np.nan)
    us = RegularGridInterpolator((xs, ys), u[0], **kw)(pts).reshape(GX.shape)
    ut = RegularGridInterpolator((xs, ys), u[1], **kw)(pts).reshape(GX.shape)
    sol = RegularGridInterpolator((xs, ys), res["solid"][:, :, k].astype(np.float32),
                                  method="nearest", **kw)(pts).reshape(GX.shape) > 0.5
    ux = us * ex[0] + ut * ey[0]
    uy = us * ex[1] + ut * ey[1]
    spd = np.hypot(ux, uy)
    spd[sol] = np.nan
    return X, Y, spd, ux, uy, sol, float(res["zs"][k])
