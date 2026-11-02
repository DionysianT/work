"""建築外殼日射 ＋ 都市風場 互動模擬 App

啟動：streamlit run app.py
"""
from __future__ import annotations

import hashlib
import io
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from sim import geometry, plots, radiation, weather, wind

st.set_page_config(page_title="外殼日射與風場模擬", layout="wide")
ss = st.session_state
MONTHS = list(range(1, 13))


# ---------------- 快取載入 ----------------
def _save(upload) -> str:
    data = upload.getvalue()
    d = Path(tempfile.gettempdir()) / "climate_sim_uploads"
    d.mkdir(exist_ok=True)
    p = d / (hashlib.md5(data).hexdigest()[:12] + Path(upload.name).suffix.lower())
    if not p.exists():
        p.write_bytes(data)
    return str(p)


@st.cache_resource(show_spinner="讀取模型中…")
def _load_mesh(path: str):
    return geometry.load_mesh(path)


@st.cache_resource(show_spinner="讀取 EPW…")
def _load_epw(path: str):
    return weather.load_epw(path)


def _progress_cb(bar):
    return lambda f, s: bar.progress(float(min(max(f, 0), 1)), text=s)


# ---------------- 側欄：輸入 ----------------
with st.sidebar:
    st.header("1. 氣象檔")
    epw_up = st.file_uploader("EPW 氣象檔", type=["epw"])
    if epw_up:
        W = _load_epw(_save(epw_up))
    else:
        st.caption("沒有 EPW 時，日射改用晴空模型（偏高，只能看相對分布）；風場須手動輸入風速。")
        c1, c2, c3 = st.columns(3)
        lat = c1.number_input("緯度", value=25.035, format="%.3f")
        lon = c2.number_input("經度", value=121.507, format="%.3f")
        tzh = c3.number_input("時區", value=8.0, step=0.5)
        W = weather.clearsky(lat, lon, tzh, "晴空模型")

    st.header("2. 量體模型")
    mdl_up = st.file_uploader("IFC / OBJ / STL / PLY / GLB", type=["ifc", "obj", "stl", "ply", "glb", "gltf", "off"])
    if mdl_up:
        raw = _load_mesh(_save(mdl_up))
        model_name = mdl_up.name
    else:
        st.caption("未上傳時使用示範量體。")
        raw = geometry.demo_model()
        model_name = "示範量體"
    up_az = st.number_input("模型「朝上」(+Y) 指向方位角（北=0，東=90）", 0.0, 359.9, 45.0, 5.0,
                            help="例：平面圖朝上是東北方 → 45")
    c1, c2 = st.columns(2)
    unit = c1.selectbox("模型單位", list(geometry.UNIT_SCALE), 0)
    y_up = c2.checkbox("Y 軸朝上的檔案", False, help="OBJ / GLB 從某些軟體匯出時是 Y 朝上")
    drop = st.checkbox("模型底部貼齊地面 z=0", False)

    with st.expander("3. 周邊量體（簡單方塊）"):
        st.caption("世界座標：X=東、Y=北（m），模型中心在 (0,0)。rot 為平面逆時針旋轉角。")
        default_ctx = ss.get("ctx_df", pd.DataFrame(columns=["cx", "cy", "w", "d", "h", "rot"],
                                                       dtype=float))
        ctx_df = st.data_editor(default_ctx, num_rows="dynamic", key="ctx_editor",
                                column_config={c: st.column_config.NumberColumn(c, format="%.1f")
                                               for c in ["cx", "cy", "w", "d", "h", "rot"]})

main = geometry.place(raw, up_az, unit, center=True, drop_to_ground=drop, y_up=y_up)
ctx = []
for _, r in ctx_df.dropna(subset=["cx", "cy", "w", "d", "h"]).iterrows():
    if r["w"] > 0 and r["d"] > 0 and r["h"] > 0:
        ctx.append(geometry.box(r["cx"], r["cy"], r["w"], r["d"], r["h"],
                                0.0 if pd.isna(r.get("rot")) else r["rot"]))
scene = geometry.Scene(main, ctx)
scene_key = hashlib.md5(main.vertices.tobytes() + str(ctx_df.values.tolist()).encode()).hexdigest()

st.title("建築外殼日射 ＋ 都市風場模擬")
info = geometry.mesh_info(main)
st.caption(f"氣象：{W.city}（{'EPW 實測' if W.is_epw else '晴空模型'}，緯度 {W.lat:.3f}、經度 {W.lon:.3f}）｜"
           f"模型：{model_name}，{info['faces']:,} 面，{info['size_x']:.0f}×{info['size_y']:.0f} m，"
           f"高 {info['z_max']:.1f} m，周邊量體 {len(ctx)} 個")
if not info["watertight"]:
    st.warning("模型不是封閉網格（not watertight），地面/風場的『是否在建物內』判斷可能出錯。"
               "建議在原軟體把量體合併成封閉實體再匯出。")

tab0, tab1, tab2, tab3 = st.tabs(["模型與氣象", "外殼日射", "都市風場", "方法與限制"])

# ---------------- 模型與氣象 ----------------
with tab0:
    c1, c2 = st.columns([1, 1])
    c1.plotly_chart(plots.footprint(scene), width="stretch")
    if W.is_epw:
        season = c2.selectbox("風花期間", list(weather.SEASONS), 0)
        rose = weather.wind_rose(W, weather.SEASONS[season])
        c2.plotly_chart(plots.wind_rose(rose, f"{season} 風花（靜風 {rose.attrs['calm%']:.1f}%，"
                                               f"平均 {rose.attrs['mean']:.1f} m/s）"),
                        width="stretch")
        st.subheader("EPW 建議的風向情境")
        st.dataframe(pd.DataFrame(weather.scenarios(W)), hide_index=True, width="stretch")
        st.subheader("逐月統計")
        mo = weather.monthly(W)
        st.dataframe(mo.T, width="stretch")
        st.caption(f"全年 GHI {W.ghi.sum() / 1000:,.0f}、DNI {W.dni.sum() / 1000:,.0f}、"
                   f"DHI {W.dhi.sum() / 1000:,.0f} kWh/m²")
    else:
        c2.info("上傳 EPW 後可看風花、盛行風向與逐月統計。")

# ---------------- 日射 ----------------
with tab1:
    with st.form("rad_form"):
        c1, c2, c3 = st.columns(3)
        period = c1.selectbox("期間", ["全年", "自訂月份"] + [k for k in weather.SEASONS if k != "全年"])
        months = c1.multiselect("自訂月份", MONTHS, MONTHS) if period == "自訂月份" else \
            weather.SEASONS.get(period, MONTHS)
        hrs = c1.slider("每日時段（當地時間）", 0, 24, (0, 24))
        albedo = c2.slider("地面反射率 albedo", 0.0, 0.6, 0.2, 0.05)
        bgrid = c2.slider("外殼分析網格 (m)", 0.5, 5.0, 1.5, 0.25, help="越小越細、越慢")
        ggrid = c2.slider("地面分析網格 (m)", 0.5, 5.0, 2.0, 0.25)
        gmargin = c3.slider("地面範圍：模型外擴 (m)，0=不算地面", 0, 150, 30, 5)
        mode = c3.radio("直達日射計算", ["分區累積（快）", "逐時（精確、慢）"], horizontal=True)
        mf = c3.radio("天空分區", [1, 2], format_func=lambda x: {1: "Tregenza 145", 2: "Reinhart 577"}[x],
                      horizontal=True)
        go_rad = st.form_submit_button("▶ 執行日射模擬", type="primary")
    if go_rad:
        p = radiation.RadParams(tuple(months), tuple(hrs), albedo, bgrid, ggrid, gmargin,
                                "binned" if mode.startswith("分區") else "hourly", mf)
        bar = st.progress(0.0, "準備中…")
        try:
            ss["rad"] = (scene_key, p, radiation.compute(scene, W, p, _progress_cb(bar)))
            bar.empty()
        except Exception as e:
            bar.empty()
            st.exception(e)

    if "rad" in ss:
        key, p, res = ss["rad"]
        if key != scene_key:
            st.warning("模型或周邊量體已變更，以下是舊結果，請重新執行。")
        roof = res["bval"][res["bnrm"][:, 2] > 0.9]
        auto_vmax = 1500 if (W.is_epw and len(p.months) == 12 and p.hours == (0, 24)) else \
            float(np.ceil((np.percentile(roof, 98) if len(roof) else res["bval"].max()) / 50) * 50)
        c1, c2, c3 = st.columns([1, 1, 2])
        vmax = c1.number_input("色階上限 kWh/m²", 1.0, 5000.0, float(auto_vmax), 50.0)
        cam = c2.selectbox("視角", list(plots.CAMERAS), 0, key="rad_cam")
        m1, m2, m3 = c3.columns(3)
        m1.metric("水平面 GHI", f"{res['ghi']:,.0f}")
        m2.metric("屋頂平均", f"{np.average(roof, weights=res['area'][res['bnrm'][:, 2] > 0.9]):,.0f}"
                  if len(roof) else "—")
        m3.metric("日照時數", f"{res['sun_hours']:,}")
        st.plotly_chart(plots.radiation_3d(res, scene, vmax, cam), width="stretch")
        st.caption("屋頂值應接近 GHI；若明顯偏離，檢查模型方位與單位。單位 kWh/m²（期間累積）。")
        summ = radiation.summary(res)
        st.dataframe(summ, width="stretch")
        title = f"{res['city']} 累積日射量 月{min(p.months)}–{max(p.months)}，{p.hours[0]}–{p.hours[1]} 時"
        d1, d2, d3 = st.columns(3)
        d1.download_button("下載平面 PNG", plots.radiation_png(res, scene, vmax, title),
                           "radiation_plan.png", "image/png")
        d2.download_button("下載朝向統計 CSV", summ.to_csv().encode("utf-8-sig"),
                           "radiation_summary.csv", "text/csv")
        buf = io.BytesIO()
        np.savez_compressed(buf, **{k: v for k, v in res.items() if isinstance(v, np.ndarray)})
        d3.download_button("下載完整結果 NPZ", buf.getvalue(), "rad_result.npz")

# ---------------- 風場 ----------------
with tab2:
    H = scene.height
    scen = weather.scenarios(W) if W.is_epw else []
    preset_names = ["手動"] + [f"{s['情境']}：{s['中文']}風 {s['頻率%']}%，{s['平均風速']} m/s" for s in scen]
    preset = st.selectbox("從 EPW 套用情境", preset_names, 1 if scen else 0)
    ps = scen[preset_names.index(preset) - 1] if preset != "手動" else None
    with st.container(border=True):
        c1, c2, c3, c4 = st.columns(4)
        wd = c1.number_input("風的來向（度，北=0，東=90）", 0.0, 359.9,
                             float(ps["角度"]) if ps else 90.0, 22.5)
        u10 = c1.number_input("參考風速 U10 (m/s，10 m 高)", 0.3, 30.0,
                              float(ps["平均風速"]) if ps and ps["平均風速"] > 0 else 3.0, 0.1)
        terr = c1.selectbox("地況（指數律 α）", ["開闊 0.14", "郊區 0.25", "市區 0.33"], 1)
        alpha = float(terr.split()[-1])
        dx = c2.select_slider("網格 dx (m)", [2.0, 2.5, 3.0, 4.0, 5.0, 6.0], 3.0)
        ground = c2.radio("地面邊界", ["free-slip", "no-slip"], horizontal=True,
                          help="free-slip：無摩擦，近地風速不被削弱（縫隙加速可能高估）；"
                               "no-slip：粗網格下近地風速會偏低約一半")
        cs = c2.slider("Smagorinsky Cs", 0.10, 0.30, 0.17, 0.01)
        ups = c3.number_input("上游距離 (m)", 20.0, 1000.0, float(max(60, round(4 * H / 10) * 10)), 10.0)
        dns = c3.number_input("下游距離 (m)", 20.0, 2000.0, float(max(100, round(8 * H / 10) * 10)), 10.0)
        lat_ = c3.number_input("兩側距離 (m)", 20.0, 1000.0, float(max(60, round(3 * H / 10) * 10)), 10.0)
        hgt = c3.number_input("計算域高度 (m)", 10.0, 600.0, float(max(3 * H, H + 30)), 5.0)
        steps = c4.number_input("時間步數（0=自動）", 0, 50000, 0, 200)
        avgf = c4.slider("取平均的最後比例", 0.1, 0.8, 0.33, 0.01)
        ulat = c4.slider("lattice 速度", 0.03, 0.10, 0.055, 0.005, help="越大越快收斂，但越容易發散")
        p = wind.WindParams(wd, u10, alpha, dx, ups, dns, lat_, hgt, int(steps), avgf, ground, cs, ulat)
        ex, ey, xs_, ys_, zs_ = wind.domain(scene, p)
        n = len(xs_) * len(ys_) * len(zs_)
        nst = p.steps or wind.auto_steps(len(xs_), p)
        est = n * nst * 3e-7     # 粗估秒數（numpy 單執行緒約 0.3 µs/格/步）
        c4.caption(f"網格 {len(xs_)}×{len(ys_)}×{len(zs_)} = {n:,} 格，{nst:,} 步，"
                   f"記憶體約 {n * 19 * 4 * 5 / 1e9:.1f} GB，估計 {est / 60:.0f} 分鐘")
        go_wind = st.button("▶ 執行風場模擬", type="primary")
    if go_wind:
        bar = st.progress(0.0, "準備中…")
        try:
            res = wind.run(scene, p, _progress_cb(bar))
            ss.setdefault("wind_runs", {})[f"{p.wd:g}° {p.u10:g} m/s {p.ground} dx{p.dx:g}"] = (scene_key, p, res)
            ss["wind_sel"] = f"{p.wd:g}° {p.u10:g} m/s {p.ground} dx{p.dx:g}"
            bar.empty()
        except Exception as e:
            bar.empty()
            st.exception(e)

    runs = ss.get("wind_runs", {})
    if runs:
        names = list(runs)
        sel = st.selectbox("檢視結果", names, names.index(ss.get("wind_sel", names[-1])))
        key, p, res = runs[sel]
        if key != scene_key:
            st.warning("模型或周邊量體已變更，以下是舊結果，請重新執行。")
        c1, c2, c3, c4 = st.columns(4)
        zmax = float(res["zs"][-2])
        zc = c1.slider("切面高度 (m)", float(res["zs"][1]), zmax, float(res["zs"][1]), float(res["dx"]))
        show = c2.radio("顯示", ["風速 m/s", "風速比（÷ 同高度自由流）"], horizontal=False)
        cam = c3.selectbox("鳥瞰視角", list(plots.CAMERAS), 0, key="wind_cam")
        X, Y, spd, ux, uy, sol, zk = wind.plan_slice(res, zc)
        uref = wind.ref_speed(res, zk)
        ratio = uref if show.startswith("風速比") else None
        vdef = 1.6 if ratio else float(np.ceil(np.nanpercentile(spd, 99.5) * 2) / 2)
        vmax = c4.number_input("色階上限", 0.1, 50.0, vdef, 0.1)
        fin = spd[np.isfinite(spd)]
        m = st.columns(5)
        m[0].metric("切面實際高度", f"{zk:.1f} m")
        m[1].metric("同高度自由流", f"{uref:.2f} m/s")
        m[2].metric("最大風速", f"{fin.max():.2f} m/s")
        m[3].metric("> 1.3× 自由流面積比", f"{100 * (fin > 1.3 * uref).mean():.1f}%")
        m[4].metric("< 0.3 m/s 面積比", f"{100 * (fin < 0.3).mean():.1f}%")
        title = f"{p.wd:g}° 來風，U10 {p.u10:g} m/s，z = {zk:.1f} m（{p.ground}，dx {p.dx:g} m）"
        st.plotly_chart(plots.wind_plan(X, Y, spd, ux, uy, sol, vmax, title, ratio_ref=ratio),
                        width="stretch")
        st.plotly_chart(plots.wind_3d(scene, X, Y, spd, zk, vmax, cam, ratio_ref=ratio),
                        width="stretch")
        st.caption(f"計算 {res['steps']:,} 步，耗時 {res['seconds'] / 60:.1f} 分鐘。"
                   "建物輪廓為該切面高度的實體格子。")
        d1, d2 = st.columns(2)
        z_show = spd / ratio if ratio else spd
        d1.download_button("下載平面 PNG", plots.wind_png(X, Y, z_show, ux, uy, sol, vmax, title,
                                                      "風速比" if ratio else "m/s"),
                           f"wind_{p.wd:g}.png", "image/png")
        buf = io.BytesIO()
        np.savez_compressed(buf, **{k: np.asarray(v) for k, v in res.items()})
        d2.download_button("下載完整流場 NPZ", buf.getvalue(), f"wind_{p.wd:g}.npz")

# ---------------- 說明 ----------------
with tab3:
    st.markdown((Path(__file__).parent / "METHODS.md").read_text(encoding="utf-8"))
