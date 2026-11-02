#!/usr/bin/env python3
"""本地端 PDF 壓縮工具。

透過「重新壓縮內嵌圖片（降低解析度 / JPEG 品質）」與「清理、壓縮 PDF 結構」
兩種方式縮小 PDF 檔案大小。

執行方式：
    python3 pdf_compress.py 輸入.pdf                    # 預設 medium 等級
    python3 pdf_compress.py 輸入.pdf -l high -o 小.pdf  # 指定壓縮等級與輸出檔
    python3 pdf_compress.py *.pdf                       # 批次壓縮多個檔案
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pymupdf

# 壓縮等級：(圖片超過此 DPI 才處理, 降到的目標 DPI, JPEG 品質)
# None 表示不處理圖片，只做無損的結構壓縮
PRESETS: dict[str, tuple[int, int, int] | None] = {
    "lossless": None,          # 畫質不變，只清理冗餘物件與重新壓縮資料流
    "low": (200, 150, 85),     # 輕度壓縮，適合列印
    "medium": (150, 120, 75),  # 一般用途（預設），畫面清晰、檔案明顯變小
    "high": (100, 72, 60),     # 強力壓縮，適合 Email / 上傳有大小限制的網站
}


def human_size(n: int) -> str:
    size = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.2f} {unit}"
        size /= 1024
    return f"{n} B"


def compress(src: Path, dst: Path, level: str, dpi: int | None,
             quality: int | None, gray: bool) -> tuple[int, int]:
    preset = PRESETS[level]
    doc = pymupdf.open(src)
    if doc.needs_pass:
        raise RuntimeError("此 PDF 有密碼保護，無法壓縮")

    if preset is not None or dpi or quality or gray:
        threshold, target, q = preset or (0, 0, 0)
        if dpi:
            target, threshold = dpi, dpi + dpi // 3
        if quality:
            q = quality
        doc.rewrite_images(
            dpi_threshold=threshold or None,
            dpi_target=target,
            quality=q or 75,
            set_to_gray=gray,
        )

    doc.subset_fonts()  # 字型只保留實際用到的字元
    doc.save(
        dst,
        garbage=4,           # 移除未使用及重複的物件
        deflate=True,        # 壓縮所有資料流
        deflate_images=True,
        deflate_fonts=True,
        clean=True,
        use_objstms=True,    # 使用物件串流，進一步縮小結構
    )
    doc.close()
    return src.stat().st_size, dst.stat().st_size


def main() -> int:
    parser = argparse.ArgumentParser(description="壓縮 PDF 檔案大小")
    parser.add_argument("inputs", nargs="+", type=Path, help="要壓縮的 PDF 檔（可多個）")
    parser.add_argument("-o", "--output", type=Path,
                        help="輸出檔名（單一檔案時）或輸出資料夾（多個檔案時）；"
                             "預設在原檔旁產生 原檔名_compressed.pdf")
    parser.add_argument("-l", "--level", choices=PRESETS, default="medium",
                        help="壓縮等級，預設 medium")
    parser.add_argument("--dpi", type=int, help="自訂圖片目標解析度（覆蓋等級設定）")
    parser.add_argument("--quality", type=int, help="自訂 JPEG 品質 1-100（覆蓋等級設定）")
    parser.add_argument("--gray", action="store_true", help="轉為灰階（可再縮小檔案）")
    args = parser.parse_args()

    multi = len(args.inputs) > 1
    if multi and args.output:
        args.output.mkdir(parents=True, exist_ok=True)

    failed = 0
    for src in args.inputs:
        if multi and args.output:
            dst = args.output / src.name
        elif args.output:
            dst = args.output
        else:
            dst = src.with_name(f"{src.stem}_compressed.pdf")
        if dst.resolve() == src.resolve():
            print(f"[略過] {src}：輸出檔不可與輸入檔相同", file=sys.stderr)
            failed += 1
            continue
        try:
            before, after = compress(src, dst, args.level, args.dpi, args.quality, args.gray)
        except Exception as e:  # noqa: BLE001
            print(f"[失敗] {src}：{e}", file=sys.stderr)
            failed += 1
            continue
        ratio = max(0.0, (1 - after / before) * 100) if before else 0.0
        note = "（已是最佳化狀態，可試試更高等級）" if after >= before else ""
        print(f"{src.name}: {human_size(before)} → {human_size(after)}"
              f"  減少 {ratio:.1f}%  → {dst}{note}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
