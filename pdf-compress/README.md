# PDF 壓縮工具

在本機縮小 PDF 檔案大小，不需上傳到任何網站。

## 安裝

需要 Python 3.9 以上版本。

```bash
cd pdf-compress
pip install -r requirements.txt
```

## 使用方式

```bash
python3 pdf_compress.py 檔案.pdf
```

會在原檔旁產生 `檔案_compressed.pdf`，並印出壓縮前後大小，例如：

```
檔案.pdf: 19.32 MB → 250.27 KB  減少 98.7%  → 檔案_compressed.pdf
```

### 壓縮等級

| 等級 | 圖片解析度 | JPEG 品質 | 適用情境 |
|------|-----------|-----------|----------|
| `lossless` | 不變 | 不變 | 畫質完全不變，只清理冗餘資料（縮減幅度較小） |
| `low` | 150 DPI | 85 | 需要列印 |
| `medium`（預設） | 120 DPI | 75 | 一般閱讀、存檔 |
| `high` | 72 DPI | 60 | Email 附件、上傳有大小限制的網站 |

### 參數說明

| 參數 | 說明 |
|------|------|
| `inputs` | 要壓縮的 PDF（可一次多個） |
| `-o, --output` | 單一檔案時為輸出檔名；多個檔案時為輸出資料夾 |
| `-l, --level` | 壓縮等級：`lossless` / `low` / `medium` / `high` |
| `--dpi` | 自訂圖片目標解析度（覆蓋等級設定） |
| `--quality` | 自訂 JPEG 品質 1–100（覆蓋等級設定） |
| `--gray` | 轉成灰階，可再縮小檔案 |

### 範例

```bash
python3 pdf_compress.py 報告.pdf -l high -o 報告_email.pdf   # 強力壓縮
python3 pdf_compress.py *.pdf -o compressed/                  # 批次壓縮整個資料夾
python3 pdf_compress.py 掃描檔.pdf --dpi 100 --quality 50 --gray
```

## 原理

1. **圖片重新壓縮**：PDF 檔案大多數的體積來自內嵌圖片（掃描頁、照片）。將高於門檻解析度的圖片降到目標 DPI，並以 JPEG 重新壓縮。
2. **字型子集化**：內嵌字型只保留實際用到的字元（對中文字型效果特別明顯）。
3. **結構清理**：移除未使用／重複的物件，並以 Deflate 壓縮所有資料流。

> 注意：純文字、向量圖為主的 PDF 本身已經很小，壓縮幅度有限；有密碼保護的 PDF 需先解除密碼。
