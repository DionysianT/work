# 瀏覽器版（免安裝）

所有計算都在瀏覽器內執行（JavaScript + Web Worker），上傳的檔案不會離開你的電腦。

- 線上版：已發佈為 claude.ai Artifact，打開網址即可使用。
- 本機版：因為 IFC 解析器需要讀取 `web-ifc.wasm`，請用本機伺服器開啟，不要直接雙擊 html：

```bash
cd climate-sim/web
python3 -m http.server 8000
```

然後在瀏覽器開 `http://localhost:8000`。只用 OBJ / STL / GLB / PLY 的話，直接雙擊 `index.html` 也可以。

方法與 Python 版（`../app.py`）相同；風場在瀏覽器裡大約每秒 300–500 萬「格·步」，參數區會即時估算時間。
