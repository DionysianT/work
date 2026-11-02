"""命令列批次版（不開介面）。

python cli.py radiation --epw taipei.epw --model model.ifc --up 45
python cli.py wind --epw taipei.epw --model model.ifc --up 45 --season 全年 --rank 1
python cli.py wind --model model.ifc --up 45 --wd 270 --u10 2.3 --dx 3
python cli.py rose --epw taipei.epw
"""
import argparse
from pathlib import Path

import numpy as np

from sim import geometry, plots, radiation, weather, wind


def scene_from(a):
    raw = geometry.load_mesh(a.model) if a.model else geometry.demo_model()
    m = geometry.place(raw, a.up, a.unit, drop_to_ground=a.drop, y_up=a.y_up)
    ctx = [geometry.box(*map(float, b.split(","))) for b in a.box]
    return geometry.Scene(m, ctx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["radiation", "wind", "rose"])
    ap.add_argument("--epw")
    ap.add_argument("--model", help="IFC/OBJ/STL/PLY/GLB；省略則用示範量體")
    ap.add_argument("--up", type=float, default=0, help="模型 +Y 指向方位角（東北=45）")
    ap.add_argument("--unit", default="m", choices=list(geometry.UNIT_SCALE))
    ap.add_argument("--drop", action="store_true")
    ap.add_argument("--y-up", action="store_true")
    ap.add_argument("--box", action="append", default=[], help="周邊量體 cx,cy,w,d,h[,rot]，可重複")
    ap.add_argument("--out", default="out")
    # 日射
    ap.add_argument("--months", default="1-12")
    ap.add_argument("--albedo", type=float, default=0.2)
    ap.add_argument("--grid", type=float, default=1.5)
    ap.add_argument("--hourly", action="store_true")
    ap.add_argument("--vmax", type=float, default=0)
    # 風
    ap.add_argument("--season", choices=["全年", "夏季", "冬季", "春季", "秋季"],
                    help="從 EPW 取該季的風向與平均風速（覆蓋 --wd/--u10）")
    ap.add_argument("--rank", type=int, default=1, choices=[1, 2], help="1=盛行風、2=次要風")
    ap.add_argument("--wd", type=float, default=90)
    ap.add_argument("--u10", type=float, default=3.0)
    ap.add_argument("--alpha", type=float, default=0.25)
    ap.add_argument("--dx", type=float, default=3.0)
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--ground", default="free-slip", choices=["free-slip", "no-slip"])
    ap.add_argument("--z", type=float, default=1.5)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(exist_ok=True)
    W = weather.load_epw(a.epw) if a.epw else None
    log = lambda f, s: print(f"\r{s}", end="", flush=True)

    if a.cmd == "rose":
        for s in weather.scenarios(W):
            print(s)
        print(weather.monthly(W))
        return

    sc = scene_from(a)
    if a.cmd == "radiation":
        W = W or weather.clearsky(25.035, 121.507, 8)
        m0, m1 = map(int, a.months.split("-"))
        p = radiation.RadParams(months=tuple(range(m0, m1 + 1)), albedo=a.albedo, bldg_grid=a.grid,
                                direct_mode="hourly" if a.hourly else "binned")
        r = radiation.compute(sc, W, p, log)
        print()
        print(radiation.summary(r))
        vmax = a.vmax or (1500 if W.is_epw else float(np.nanmax(r["bval"])))
        (out / "radiation_plan.png").write_bytes(plots.radiation_png(r, sc, vmax, f"{W.city} 日射量"))
        for cam in ["西南鳥瞰", "東北鳥瞰"]:
            plots.radiation_3d(r, sc, vmax, cam).write_html(out / f"radiation_{cam}.html")
        radiation.summary(r).to_csv(out / "radiation_summary.csv", encoding="utf-8-sig")
    else:
        wd, u10 = a.wd, a.u10
        if a.season and W:
            cand = [s for s in weather.scenarios(W) if s["季節"].startswith(a.season)]
            s = cand[a.rank - 1]
            wd, u10 = s["角度"], s["平均風速"]
            print("情境", s)
        H = sc.height
        p = wind.WindParams(wd, u10, a.alpha, a.dx, max(60, 4 * H), max(100, 8 * H), max(60, 3 * H),
                            max(3 * H, H + 30), a.steps, ground=a.ground)
        r = wind.run(sc, p, log)
        print()
        X, Y, spd, ux, uy, sol, zk = wind.plan_slice(r, a.z)
        vmax = float(np.ceil(np.nanpercentile(spd, 99.5) * 2) / 2)
        t = f"{wd:g}° 來風 U10 {u10:g} m/s z={zk:.1f} m"
        (out / f"wind_{wd:g}.png").write_bytes(plots.wind_png(X, Y, spd, ux, uy, sol, vmax, t))
        plots.wind_3d(sc, X, Y, spd, zk, vmax).write_html(out / f"wind_{wd:g}_3d.html")
        np.savez_compressed(out / f"wind_{wd:g}.npz", **{k: np.asarray(v) for k, v in r.items()})
    print("輸出到", out.resolve())


if __name__ == "__main__":
    main()
