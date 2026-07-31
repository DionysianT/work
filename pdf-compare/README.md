# PDF 比較工具

在本機比較兩份 PDF 檔案，列出文字差異與圖片（含圖示）差異，並產生一份可在瀏覽器開啟的 HTML 報告。

## 功能

- **文字差異**：逐頁比對文字內容，以新增（綠色）／刪除（紅色）標示不同之處。
- **圖片／圖示差異**：比對每頁內嵌的點陣圖片（Logo、圖示等），列出新增與刪除的圖片並附上縮圖。
- **版面視覺差異**：將每頁轉成圖片後比對像素，用紅框標示出兩份 PDF 中外觀不同的區域（可抓到向量圖形、線條、位置偏移等文字比對抓不到的差異）。
- 若兩份 PDF 頁數不同，多出來的頁面會標示為「僅存在於舊檔／新檔」。

## 安裝

需要 Python 3.9 以上版本。

```bash
cd pdf-compare
pip install -r requirements.txt
```

## 使用方式

```bash
python3 pdf_compare.py 舊版.pdf 新版.pdf -o report_dir
```

執行後：

- 終端機會印出每頁的差異摘要。
- 於 `report_dir/report.html` 產生完整的圖文報告，用瀏覽器打開即可查看逐頁比對結果（含頁面截圖、紅框標示差異區域、文字差異、圖片差異縮圖）。

### 參數說明

| 參數 | 說明 |
|------|------|
| `old_pdf` | 舊版 PDF 路徑（必填，第一個參數） |
| `new_pdf` | 新版 PDF 路徑（必填，第二個參數） |
| `-o, --output` | 輸出報告的資料夾，預設為 `pdf_compare_report` |
| `--zoom` | 頁面轉圖片的縮放倍率，數字越大畫質越清晰但速度越慢，預設 `2.0` |

### 範例

```bash
python3 pdf_compare.py contract_v1.pdf contract_v2.pdf -o diff_report
open diff_report/report.html   # macOS，Windows 可用 start，Linux 可用 xdg-open
```

## 判定原理

1. **文字**：用 [PyMuPDF](https://pymupdf.readthedocs.io/) 逐頁擷取純文字後，以 Python `difflib` 做逐行比對。
2. **圖片**：擷取每頁內嵌的點陣圖片，計算內容雜湊值（SHA-1），比對兩份 PDF 中該頁圖片雜湊值的差集，找出新增／刪除的圖片。
3. **版面**：把每頁以固定倍率轉成點陣圖，切成小區塊比對灰階像素差異，再合併相鄰有差異的區塊成矩形框，標示在新檔頁面截圖上。

可依需求調整 `pdf_compare.py` 開頭的 `BLOCK_SIZE`、`DIFF_THRESHOLD`、`MIN_DIFF_RATIO` 來調整視覺差異偵測的靈敏度。
