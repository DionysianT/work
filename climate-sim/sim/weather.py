"""EPW 氣象資料：讀取、風花統計、建議風向情境。"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta, timezone

import numpy as np
import pandas as pd

SECTORS16 = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
             "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
SECTORS16_ZH = ["北", "北北東", "東北", "東北東", "東", "東南東", "東南", "南南東",
                "南", "南南西", "西南", "西南西", "西", "西北西", "西北", "北北西"]
SEASONS = {"全年": list(range(1, 13)), "夏季 (6–8月)": [6, 7, 8],
           "冬季 (12–2月)": [12, 1, 2], "春季 (3–5月)": [3, 4, 5], "秋季 (9–11月)": [9, 10, 11]}


@dataclass
class Weather:
    idx: pd.DatetimeIndex          # 每小時中點，含時區
    dni: np.ndarray                # W/m²
    dhi: np.ndarray
    ghi: np.ndarray
    ws: np.ndarray                 # m/s（氣象站 10 m）
    wd: np.ndarray                 # 度，風的來向
    dbt: np.ndarray                # 乾球溫度 °C
    lat: float
    lon: float
    tz_hours: float
    city: str
    source: str                    # 'EPW' 或 'clearsky'

    @property
    def is_epw(self) -> bool:
        return self.source == "EPW"


def _index(tz_hours: float) -> pd.DatetimeIndex:
    tz = timezone(timedelta(hours=tz_hours))
    return pd.date_range("2023-01-01 00:30", periods=8760, freq="h", tz=tz)


def load_epw(path: str) -> Weather:
    from ladybug.epw import EPW

    e = EPW(path)
    loc = e.location
    arr = lambda c: np.asarray(c.values, float)[:8760]
    return Weather(_index(loc.time_zone), arr(e.direct_normal_radiation),
                   arr(e.diffuse_horizontal_radiation), arr(e.global_horizontal_radiation),
                   arr(e.wind_speed), arr(e.wind_direction), arr(e.dry_bulb_temperature),
                   loc.latitude, loc.longitude, loc.time_zone,
                   f"{loc.city} {loc.country}".strip(), "EPW")


def clearsky(lat: float, lon: float, tz_hours: float, city: str = "") -> Weather:
    """沒有 EPW 時的晴空模型（pvlib Ineichen，偏高，只能看相對分布）。"""
    import pvlib

    idx = _index(tz_hours)
    cs = pvlib.location.Location(lat, lon).get_clearsky(idx, model="ineichen")  # idx 已含時區
    z = np.zeros(8760)
    return Weather(idx, cs["dni"].values, cs["dhi"].values, cs["ghi"].values,
                   z, z, z, lat, lon, tz_hours, city or f"{lat:.2f},{lon:.2f}", "clearsky")


def hour_mask(w: Weather, months=range(1, 13), hours=(0, 24)) -> np.ndarray:
    m = np.isin(w.idx.month, list(months))
    h0, h1 = hours
    hr = w.idx.hour
    m &= (hr >= h0) & (hr < h1) if h0 <= h1 else ((hr >= h0) | (hr < h1))
    return m


def sector_of(wd: np.ndarray, n: int = 16) -> np.ndarray:
    return (np.floor((np.asarray(wd) % 360) / (360 / n) + 0.5) % n).astype(int)


def wind_rose(w: Weather, months=range(1, 13), hours=(0, 24), calm: float = 0.5,
              bins=(0.5, 2, 4, 6, 8, 100)) -> pd.DataFrame:
    """16 方位風花表：各方位頻率（%）、平均風速、各風速級距頻率。靜風（< calm）另計。"""
    msk = hour_mask(w, months, hours)
    ws, wd = w.ws[msk], w.wd[msk]
    n = len(ws)
    windy = ws >= calm
    sec = sector_of(wd[windy])
    rows = []
    for i, name in enumerate(SECTORS16):
        s = ws[windy][sec == i]
        r = {"方位": name, "中文": SECTORS16_ZH[i], "角度": i * 22.5,
             "頻率%": 100 * len(s) / max(n, 1), "平均風速": float(s.mean()) if len(s) else 0.0}
        for lo, hi in zip(bins[:-1], bins[1:]):
            lab = f"{lo:g}–{hi:g}" if hi < 100 else f">{lo:g}"
            r[lab] = 100 * ((s >= lo) & (s < hi)).sum() / max(n, 1)
        rows.append(r)
    df = pd.DataFrame(rows)
    df.attrs["calm%"] = 100 * (~windy).sum() / max(n, 1)
    df.attrs["mean"] = float(ws.mean()) if n else 0.0
    return df


def scenarios(w: Weather) -> list[dict]:
    """依 EPW 建議的模擬情境：各季節的最盛行與次盛行風向，附該方位平均風速。"""
    out = []
    for season, months in SEASONS.items():
        df = wind_rose(w, months).sort_values("頻率%", ascending=False)
        for rank, (_, r) in enumerate(df.head(2).iterrows()):
            out.append({"情境": f"{season} {'盛行' if rank == 0 else '次要'}", "季節": season,
                        "風向": r["方位"], "中文": r["中文"], "角度": r["角度"],
                        "頻率%": round(r["頻率%"], 1), "平均風速": round(r["平均風速"], 2)})
    return out


def monthly(w: Weather) -> pd.DataFrame:
    df = pd.DataFrame({"月": w.idx.month, "GHI": w.ghi / 1000, "DNI": w.dni / 1000,
                       "DHI": w.dhi / 1000, "風速": w.ws, "氣溫": w.dbt})
    g = df.groupby("月")
    return pd.DataFrame({"GHI kWh/m²": g["GHI"].sum(), "DNI kWh/m²": g["DNI"].sum(),
                         "DHI kWh/m²": g["DHI"].sum(), "平均風速 m/s": g["風速"].mean(),
                         "平均氣溫 °C": g["氣溫"].mean()}).round(1)
