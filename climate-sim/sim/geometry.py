"""量體模型載入、定位與周邊量體。

座標慣例（世界座標）：X = 東、Y = 北、Z = 上，單位 m。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import trimesh

UNIT_SCALE = {"m": 1.0, "cm": 0.01, "mm": 0.001, "ft": 0.3048}


def load_mesh(path: str | Path) -> trimesh.Trimesh:
    """讀取 IFC / OBJ / STL / PLY / GLB / OFF 成單一三角網格。"""
    path = Path(path)
    if path.suffix.lower() == ".ifc":
        import ifcopenshell
        import ifcopenshell.geom

        f = ifcopenshell.open(str(path))
        s = ifcopenshell.geom.settings()
        s.set(s.USE_WORLD_COORDS, True)
        it = ifcopenshell.geom.iterator(s, f)
        meshes = []
        if it.initialize():
            while True:
                g = it.get().geometry
                v = np.array(g.verts, float).reshape(-1, 3)
                fa = np.array(g.faces, int).reshape(-1, 3)
                if len(fa):
                    meshes.append(trimesh.Trimesh(v, fa))
                if not it.next():
                    break
        if not meshes:
            raise ValueError("IFC 內沒有可轉換的幾何")
        return trimesh.util.concatenate(meshes)
    m = trimesh.load(str(path), force="mesh")
    if not isinstance(m, trimesh.Trimesh) or len(m.faces) == 0:
        raise ValueError("模型檔沒有三角面")
    return m


def place(mesh: trimesh.Trimesh, up_azimuth: float, unit: str = "m",
          center: bool = True, drop_to_ground: bool = False,
          y_up: bool = False) -> trimesh.Trimesh:
    """把模型放到世界座標。

    up_azimuth：模型的 +Y 軸（平面圖的「朝上」）指向的方位角，度，北=0 順時針。
    y_up：模型是 Y 軸朝上（OBJ/GLB 常見）時，先轉成 Z 朝上。
    """
    m = mesh.copy()
    m.apply_scale(UNIT_SCALE[unit])
    if y_up:
        m.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
    if center:
        c = m.bounds.mean(0)
        c[2] = 0
        m.apply_translation(-c)
    if drop_to_ground:
        m.apply_translation([0, 0, -m.bounds[0, 2]])
    # 模型 +Y → 方位角 A：繞 Z 軸旋轉 -A（逆時針為正）
    m.apply_transform(trimesh.transformations.rotation_matrix(np.radians(-up_azimuth), [0, 0, 1]))
    return m


def box(cx: float, cy: float, w: float, d: float, h: float, rot: float = 0.0,
        z0: float = 0.0) -> trimesh.Trimesh:
    """世界座標的方塊量體：中心 (cx, cy)，寬 w（X）、深 d（Y）、高 h，rot 為平面逆時針旋轉角（度）。"""
    b = trimesh.creation.box(extents=[w, d, h])
    b.apply_translation([0, 0, h / 2 + z0])
    b.apply_transform(trimesh.transformations.rotation_matrix(np.radians(rot), [0, 0, 1]))
    b.apply_translation([cx, cy, 0])
    return b


def demo_model() -> trimesh.Trimesh:
    """示範量體：L 形西棟（中庭朝東北開口）＋東棟，兩棟間留一條 8 m 縫（模型座標，+Y 為圖面朝上）。"""
    from shapely.geometry import Polygon

    west = Polygon([(-34, -30), (-10, -30), (-10, 14), (6, 14), (6, 30), (-34, 30)])
    east = Polygon([(14, -30), (34, -30), (34, 18), (14, 18)])
    annex = Polygon([(14, 22), (34, 22), (34, 30), (14, 30)])
    parts = [trimesh.creation.extrude_polygon(west, 24),
             trimesh.creation.extrude_polygon(east, 30),
             trimesh.creation.extrude_polygon(annex, 12)]
    return trimesh.util.concatenate(parts)


@dataclass
class Scene:
    main: trimesh.Trimesh
    context: list[trimesh.Trimesh] = field(default_factory=list)

    @property
    def parts(self) -> list[trimesh.Trimesh]:
        return [self.main] + list(self.context)

    @property
    def combined(self) -> trimesh.Trimesh:
        return trimesh.util.concatenate(self.parts) if self.context else self.main

    def contains(self, pts: np.ndarray) -> np.ndarray:
        """點是否在任一量體內（逐個量體判斷，避免合併網格重疊時奇偶判斷出錯）。"""
        out = np.zeros(len(pts), bool)
        for p in self.parts:
            lo, hi = p.bounds
            q = ~out & np.all((pts >= lo) & (pts <= hi), axis=1)
            if q.any():
                out[q] = p.contains(pts[q])
        return out

    @property
    def height(self) -> float:
        return float(max(p.bounds[1, 2] for p in self.parts))


def mesh_info(m: trimesh.Trimesh) -> dict:
    lo, hi = m.bounds
    return {"faces": int(len(m.faces)), "watertight": bool(m.is_watertight),
            "size_x": float(hi[0] - lo[0]), "size_y": float(hi[1] - lo[1]),
            "z_min": float(lo[2]), "z_max": float(hi[2])}
