"""互動圖（plotly）與報告用 PNG（matplotlib）。"""
from __future__ import annotations

import io

import numpy as np
import plotly.graph_objects as go

# 和參考圖一致的日射色階（藍 → 黃 → 紅）
RAD_COLORS = ["#4b6fb5", "#6fa8dc", "#a9c9e8", "#f6f2c4", "#f9d35a", "#f39c34", "#e2481e", "#c4161c"]
RAD_SCALE = [[i / (len(RAD_COLORS) - 1), c] for i, c in enumerate(RAD_COLORS)]
WIND_SCALE = "Turbo"

CAMERAS = {
    "西南鳥瞰": dict(eye=dict(x=-1.3, y=-1.3, z=1.1)),
    "東北鳥瞰": dict(eye=dict(x=1.3, y=1.3, z=1.1)),
    "東南鳥瞰": dict(eye=dict(x=1.3, y=-1.3, z=1.1)),
    "西北鳥瞰": dict(eye=dict(x=-1.3, y=1.3, z=1.1)),
    "正上方": dict(eye=dict(x=0, y=0, z=2.2), up=dict(x=0, y=1, z=0)),
}


def _mesh3d(m_v, m_f, **kw):
    return go.Mesh3d(x=m_v[:, 0], y=m_v[:, 1], z=m_v[:, 2],
                     i=m_f[:, 0], j=m_f[:, 1], k=m_f[:, 2], flatshading=True, **kw)


def _layout(fig, camera, title=""):
    fig.update_layout(scene=dict(aspectmode="data", camera=camera,
                                 xaxis_title="東 X (m)", yaxis_title="北 Y (m)", zaxis_title="Z (m)"),
                      margin=dict(l=0, r=0, t=30, b=0), height=650, title=title)
    return fig


def context_traces(scene, opacity=1.0):
    out = []
    for m in scene.context:
        out.append(_mesh3d(m.vertices, m.faces, color="#cfcfcf", opacity=opacity,
                           name="周邊量體", hoverinfo="skip", showscale=False,
                           lighting=dict(ambient=0.7, diffuse=0.5)))
    return out


def radiation_3d(res, scene, vmax, camera="西南鳥瞰"):
    hover = "%{intensity:.0f} kWh/m²<extra></extra>"
    tr = [_mesh3d(res["bv"], res["bf"], intensity=res["bval"], intensitymode="cell",
                  colorscale=RAD_SCALE, cmin=0, cmax=vmax, hovertemplate=hover,
                  colorbar=dict(title="kWh/m²"), lighting=dict(ambient=1, diffuse=0, specular=0))]
    if res["gval"] is not None:
        gx, gy, gv = res["GX"][0], res["GY"][:, 0], res["gval"]
        tr.append(go.Surface(x=gx, y=gy, z=np.where(np.isfinite(gv), 0.0, np.nan), surfacecolor=gv,
                             colorscale=RAD_SCALE, cmin=0, cmax=vmax, showscale=False,
                             hovertemplate="地面 %{surfacecolor:.0f} kWh/m²<extra></extra>",
                             lighting=dict(ambient=1, diffuse=0, specular=0)))
    tr += context_traces(scene, 0.5)
    return _layout(go.Figure(tr), CAMERAS[camera])


def wind_rose(df, title=""):
    cols = [c for c in df.columns if "–" in c or c.startswith(">")]
    colors = ["#c6dbef", "#9ecae1", "#6baed6", "#3182bd", "#08519c", "#08306b"]
    fig = go.Figure()
    for c, col in zip(cols, colors):
        fig.add_barpolar(r=df[c], theta=df["方位"], name=f"{c} m/s", marker_color=col,
                         hovertemplate="%{theta} %{r:.1f}%<extra>" + c + " m/s</extra>")
    fig.update_layout(polar=dict(angularaxis=dict(direction="clockwise", rotation=90),
                                 radialaxis=dict(ticksuffix="%")),
                      height=480, margin=dict(l=30, r=30, t=40, b=30), title=title,
                      legend=dict(title="風速"))
    return fig


def footprint(scene, title="平面（+Y = 北）"):
    fig = go.Figure()
    for m, col, name in [(scene.main, "#4b6fb5", "主量體")] + [(c, "#bbbbbb", "周邊") for c in scene.context]:
        top = m.face_normals[:, 2] > 0.9
        tri = m.triangles[top]
        if not len(tri):
            continue
        h = tri[:, :, 2].max(1)
        for t, z in zip(tri, h):
            fig.add_scatter(x=list(t[:, 0]) + [t[0, 0]], y=list(t[:, 1]) + [t[0, 1]], fill="toself",
                            mode="lines", line=dict(width=0.5, color="#333"), fillcolor=col,
                            opacity=0.35 + 0.6 * min(z / max(scene.height, 1), 1), name=name,
                            hovertemplate=f"{name} 高 {z:.1f} m<extra></extra>", showlegend=False)
    fig.update_layout(yaxis=dict(scaleanchor="x"), height=480, title=title,
                      margin=dict(l=40, r=10, t=40, b=40), xaxis_title="東 X (m)", yaxis_title="北 Y (m)")
    return fig


def wind_plan(X, Y, spd, ux, uy, sol, vmax, title="", arrow_every=6, ratio_ref=None):
    z = spd / ratio_ref if ratio_ref else spd
    unit = "風速比" if ratio_ref else "m/s"
    fig = go.Figure(go.Heatmap(x=X, y=Y, z=z, colorscale=WIND_SCALE, zmin=0, zmax=vmax,
                               colorbar=dict(title=unit),
                               hovertemplate="x %{x:.0f}  y %{y:.0f}<br>%{z:.2f} " + unit + "<extra></extra>"))
    fig.add_contour(x=X, y=Y, z=sol.astype(float), contours=dict(start=0.5, end=0.5, coloring="none"),
                    line=dict(color="black", width=2), showscale=False, showlegend=False, hoverinfo="skip")
    s = slice(None, None, arrow_every)
    gx, gy = np.meshgrid(X[s], Y[s])
    u, v = ux[s, s], uy[s, s]
    ok = np.isfinite(u) & ~sol[s, s]
    mag = np.hypot(u, v)[ok]
    L = (X[1] - X[0]) * arrow_every * 0.8 / max(np.nanmax(mag), 1e-6) if mag.size else 0
    xs, ys = [], []
    for x0, y0, du, dv in zip(gx[ok], gy[ok], u[ok] * L, v[ok] * L):
        x1, y1 = x0 + du, y0 + dv
        a = np.arctan2(dv, du)
        h = 0.3 * np.hypot(du, dv)
        xs += [x0, x1, x1 - h * np.cos(a - 0.4), x1, x1 - h * np.cos(a + 0.4), None]
        ys += [y0, y1, y1 - h * np.sin(a - 0.4), y1, y1 - h * np.sin(a + 0.4), None]
    fig.add_scatter(x=xs, y=ys, mode="lines", line=dict(color="rgba(0,0,0,0.55)", width=1),
                    hoverinfo="skip", showlegend=False)
    fig.update_layout(yaxis=dict(scaleanchor="x"), height=680, title=title,
                      margin=dict(l=40, r=10, t=40, b=40), xaxis_title="東 X (m)", yaxis_title="北 Y (m)")
    return fig


def wind_3d(scene, X, Y, spd, z0, vmax, camera="西南鳥瞰", ratio_ref=None):
    zz = spd / ratio_ref if ratio_ref else spd
    tr = [go.Surface(x=X, y=Y, z=np.where(np.isfinite(zz), z0, np.nan), surfacecolor=zz,
                     colorscale=WIND_SCALE, cmin=0, cmax=vmax,
                     colorbar=dict(title="風速比" if ratio_ref else "m/s"),
                     hovertemplate="%{surfacecolor:.2f}<extra></extra>",
                     lighting=dict(ambient=1, diffuse=0, specular=0))]
    tr.append(_mesh3d(scene.main.vertices, scene.main.faces, color="#eeeeee", hoverinfo="skip",
                      lighting=dict(ambient=0.6, diffuse=0.6)))
    tr += context_traces(scene, 0.9)
    return _layout(go.Figure(tr), CAMERAS[camera])


# ---------------- PNG（報告用） ----------------

def _mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    want = ["Noto Sans CJK TC", "Noto Sans TC", "Microsoft JhengHei", "PingFang TC",
            "Heiti TC", "Arial Unicode MS", "DejaVu Sans"]
    have = {f.name for f in font_manager.fontManager.ttflist}
    plt.rcParams["font.family"] = [f for f in want if f in have] or ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def _png(fig, plt):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def radiation_png(res, scene, vmax, title):
    """俯視平面 PNG：地面＋屋頂（由上往下看到的最高面）。"""
    plt = _mpl()
    from matplotlib.collections import PolyCollection
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("rad", RAD_COLORS)
    fig, ax = plt.subplots(figsize=(10, 9))
    if res["gval"] is not None:
        g = res["GX"][0, 1] - res["GX"][0, 0]
        ax.imshow(res["gval"], origin="lower", cmap=cmap, vmin=0, vmax=vmax,
                  extent=(res["GX"].min() - g / 2, res["GX"].max() + g / 2,
                          res["GY"].min() - g / 2, res["GY"].max() + g / 2))
    up = res["bnrm"][:, 2] > 0.05
    tri = res["bv"][res["bf"][up]]
    order = np.argsort(tri[:, :, 2].mean(1))
    pc = PolyCollection(tri[order][:, :, :2], array=res["bval"][up][order], cmap=cmap,
                        edgecolor="face", linewidth=0.2)
    pc.set_clim(0, vmax)
    ax.add_collection(pc)
    for m in scene.context:
        top = m.face_normals[:, 2] > 0.9
        ax.add_collection(PolyCollection(m.triangles[top][:, :, :2], facecolor="#cccccc",
                                         edgecolor="#888", lw=0.3))
    ax.set_aspect("equal")
    ax.autoscale_view()
    ax.set_xlabel("東 X (m)")
    ax.set_ylabel("北 Y (m)")
    ax.set_title(title)
    fig.colorbar(pc, ax=ax, shrink=0.7, label="kWh/m²")
    return _png(fig, plt)


def wind_png(X, Y, spd, ux, uy, sol, vmax, title, unit="m/s"):
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.pcolormesh(X, Y, spd, cmap="turbo", vmin=0, vmax=vmax, shading="auto")
    ax.contour(X, Y, sol.astype(float), levels=[0.5], colors="k", linewidths=1.5)
    u = np.where(np.isfinite(ux) & ~sol, ux, 0)
    v = np.where(np.isfinite(uy) & ~sol, uy, 0)
    try:
        ax.streamplot(X, Y, u, v, color=(0, 0, 0, 0.45), density=1.6, linewidth=0.6, arrowsize=0.8)
    except Exception:
        pass
    ax.set_aspect("equal")
    ax.set_xlabel("東 X (m)")
    ax.set_ylabel("北 Y (m)")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.7, label=unit)
    return _png(fig, plt)
