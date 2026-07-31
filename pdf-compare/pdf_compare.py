#!/usr/bin/env python3
"""本地端 PDF 比較工具。

比較兩份 PDF 檔案，逐頁列出：
  1. 文字差異（新增 / 刪除 / 修改的內容，逐行標示）
  2. 內嵌點陣圖片（圖示、Logo 等）的新增 / 刪除
  3. 版面視覺差異（將每頁轉成圖片後比對像素，框出不同的區域，
     可抓到文字比對抓不到的向量圖形、圖示位移等差異）

執行方式：
    python3 pdf_compare.py 舊檔.pdf 新檔.pdf -o report_dir

執行完成後，打開 report_dir/report.html 即可看到完整比對報告。
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import sys
from dataclasses import dataclass, field
from html import escape
from pathlib import Path

import fitz  # PyMuPDF
import numpy as np
from PIL import Image, ImageChops, ImageDraw

RENDER_ZOOM = 2.0       # 頁面轉圖片的縮放倍率，越大越清晰但越慢
BLOCK_SIZE = 24         # 視覺差異偵測的分塊像素大小
DIFF_THRESHOLD = 18     # 單一像素灰階差異超過此值視為「不同」（0-255）
MIN_DIFF_RATIO = 0.02   # 一個分塊內「不同像素」比例超過此值，該分塊才算有差異


@dataclass
class EmbeddedImage:
    hash: str
    width: int
    height: int
    ext: str
    data: bytes


@dataclass
class PageResult:
    index: int  # 1-based
    only_in: str = ""  # "old" / "new" / "" (兩邊都有)
    text_diff_html: str = ""
    text_changed: bool = False
    visual_changed: bool = False
    visual_box_count: int = 0
    old_image_rel: str = ""
    new_image_rel: str = ""
    diff_image_rel: str = ""
    added_images: list = field(default_factory=list)
    removed_images: list = field(default_factory=list)


def extract_page_text(page: "fitz.Page") -> str:
    return page.get_text("text")


def render_page(page: "fitz.Page", zoom: float = RENDER_ZOOM) -> Image.Image:
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)


def image_hash(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()


def extract_embedded_images(doc: "fitz.Document", page: "fitz.Page") -> dict[str, EmbeddedImage]:
    """回傳該頁所有內嵌點陣圖片（以內容雜湊為 key，避免重複計算同一張圖）。"""
    results: dict[str, EmbeddedImage] = {}
    for img in page.get_images(full=True):
        xref = img[0]
        try:
            base = doc.extract_image(xref)
        except Exception:
            continue
        data = base["image"]
        h = image_hash(data)
        if h not in results:
            results[h] = EmbeddedImage(
                hash=h,
                width=base.get("width", 0),
                height=base.get("height", 0),
                ext=base.get("ext", "png"),
                data=data,
            )
    return results


def diff_visual(img_old: Image.Image, img_new: Image.Image):
    """比對兩張頁面圖片的像素差異。

    回傳 (是否有差異, 差異區塊數, 標示紅框後的新頁面圖片)。
    """
    w = max(img_old.width, img_new.width)
    h = max(img_old.height, img_new.height)
    canvas_old = Image.new("RGB", (w, h), "white")
    canvas_new = Image.new("RGB", (w, h), "white")
    canvas_old.paste(img_old, (0, 0))
    canvas_new.paste(img_new, (0, 0))

    diff = ImageChops.difference(canvas_old.convert("L"), canvas_new.convert("L"))
    arr = np.asarray(diff)

    gy = h // BLOCK_SIZE + 1
    gx = w // BLOCK_SIZE + 1
    flagged = np.zeros((gy, gx), dtype=bool)
    for by in range(0, h, BLOCK_SIZE):
        for bx in range(0, w, BLOCK_SIZE):
            block = arr[by:by + BLOCK_SIZE, bx:bx + BLOCK_SIZE]
            if block.size == 0:
                continue
            if np.mean(block > DIFF_THRESHOLD) > MIN_DIFF_RATIO:
                flagged[by // BLOCK_SIZE, bx // BLOCK_SIZE] = True

    # 用 BFS 把相鄰的差異分塊合併成矩形框，避免同一處差異被拆成一堆小框
    visited = np.zeros_like(flagged)
    boxes = []
    for j in range(gy):
        for i in range(gx):
            if flagged[j, i] and not visited[j, i]:
                stack = [(j, i)]
                visited[j, i] = True
                min_j = max_j = j
                min_i = max_i = i
                while stack:
                    cy, cx = stack.pop()
                    min_j, max_j = min(min_j, cy), max(max_j, cy)
                    min_i, max_i = min(min_i, cx), max(max_i, cx)
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < gy and 0 <= nx < gx and flagged[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            stack.append((ny, nx))
                boxes.append((
                    min_i * BLOCK_SIZE, min_j * BLOCK_SIZE,
                    min((max_i + 1) * BLOCK_SIZE, w), min((max_j + 1) * BLOCK_SIZE, h),
                ))

    marked = canvas_new.copy()
    if boxes:
        draw = ImageDraw.Draw(marked)
        for (x0, y0, x1, y1) in boxes:
            draw.rectangle([x0, y0, x1, y1], outline=(230, 0, 0), width=3)

    return len(boxes) > 0, len(boxes), marked, canvas_old


def make_text_diff_html(text_old: str, text_new: str) -> tuple[str, bool]:
    lines_old = text_old.splitlines()
    lines_new = text_new.splitlines()
    sm = difflib.SequenceMatcher(a=lines_old, b=lines_new)
    changed = False
    rows = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        changed = True
        for line in lines_old[i1:i2]:
            rows.append(f'<div class="del">- {escape(line) or "&nbsp;"}</div>')
        for line in lines_new[j1:j2]:
            rows.append(f'<div class="add">+ {escape(line) or "&nbsp;"}</div>')
    if not rows:
        return "", changed
    return "\n".join(rows), changed


def compare_pdfs(old_path: Path, new_path: Path, out_dir: Path, zoom: float = RENDER_ZOOM) -> list[PageResult]:
    pages_dir = out_dir / "pages"
    images_dir = out_dir / "images"
    pages_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    doc_old = fitz.open(old_path)
    doc_new = fitz.open(new_path)

    n_old, n_new = len(doc_old), len(doc_new)
    n_common = min(n_old, n_new)

    results: list[PageResult] = []

    for idx in range(n_common):
        page_old = doc_old[idx]
        page_new = doc_new[idx]
        result = PageResult(index=idx + 1)

        text_old = extract_page_text(page_old)
        text_new = extract_page_text(page_new)
        result.text_diff_html, result.text_changed = make_text_diff_html(text_old, text_new)

        img_old_render = render_page(page_old, zoom)
        img_new_render = render_page(page_new, zoom)
        changed, box_count, marked, canvas_old = diff_visual(img_old_render, img_new_render)
        result.visual_changed = changed
        result.visual_box_count = box_count

        old_rel = f"pages/p{idx + 1:03d}_old.png"
        new_rel = f"pages/p{idx + 1:03d}_new_diff.png"
        canvas_old.save(out_dir / old_rel)
        marked.save(out_dir / new_rel)
        result.old_image_rel = old_rel
        result.diff_image_rel = new_rel

        images_old = extract_embedded_images(doc_old, page_old)
        images_new = extract_embedded_images(doc_new, page_new)
        removed_hashes = set(images_old) - set(images_new)
        added_hashes = set(images_new) - set(images_old)

        for h in sorted(removed_hashes):
            info = images_old[h]
            rel = f"images/p{idx + 1:03d}_removed_{h[:8]}.{info.ext}"
            (out_dir / rel).write_bytes(info.data)
            result.removed_images.append({"rel": rel, "w": info.width, "h": info.height})
        for h in sorted(added_hashes):
            info = images_new[h]
            rel = f"images/p{idx + 1:03d}_added_{h[:8]}.{info.ext}"
            (out_dir / rel).write_bytes(info.data)
            result.added_images.append({"rel": rel, "w": info.width, "h": info.height})

        results.append(result)

    # 頁數不同時，多出來的頁面整頁列為「僅存在於其中一份檔案」
    if n_old > n_common:
        for idx in range(n_common, n_old):
            page_old = doc_old[idx]
            result = PageResult(index=idx + 1, only_in="old")
            img = render_page(page_old, zoom)
            rel = f"pages/p{idx + 1:03d}_old_only.png"
            img.save(out_dir / rel)
            result.old_image_rel = rel
            results.append(result)
    if n_new > n_common:
        for idx in range(n_common, n_new):
            page_new = doc_new[idx]
            result = PageResult(index=idx + 1, only_in="new")
            img = render_page(page_new, zoom)
            rel = f"pages/p{idx + 1:03d}_new_only.png"
            img.save(out_dir / rel)
            result.diff_image_rel = rel
            results.append(result)

    doc_old.close()
    doc_new.close()
    return results


REPORT_CSS = """
body { font-family: -apple-system, "Noto Sans TC", "Microsoft JhengHei", sans-serif;
       margin: 0; padding: 0; background: #f5f6f8; color: #1a1a1a; }
header { background: #1f2933; color: #fff; padding: 20px 32px; }
header h1 { margin: 0 0 6px; font-size: 20px; }
header p { margin: 2px 0; color: #cbd2d9; font-size: 13px; }
main { padding: 24px 32px; max-width: 1200px; margin: 0 auto; }
.summary { background: #fff; border-radius: 8px; padding: 16px 20px; margin-bottom: 24px;
           box-shadow: 0 1px 3px rgba(0,0,0,.08); }
.summary table { border-collapse: collapse; width: 100%; font-size: 13px; }
.summary th, .summary td { text-align: left; padding: 6px 10px; border-bottom: 1px solid #eee; }
.summary a { color: #2563eb; text-decoration: none; }
.page-block { background: #fff; border-radius: 8px; padding: 20px; margin-bottom: 24px;
              box-shadow: 0 1px 3px rgba(0,0,0,.08); }
.page-block h2 { margin-top: 0; font-size: 16px; border-bottom: 1px solid #eee; padding-bottom: 8px; }
.badge { display: inline-block; font-size: 12px; padding: 2px 8px; border-radius: 10px; margin-left: 8px; }
.badge.changed { background: #fde8e8; color: #c53030; }
.badge.same { background: #e6fffa; color: #257953; }
.text-diff { font-family: "SFMono-Regular", Consolas, monospace; font-size: 13px;
             background: #fafafa; border: 1px solid #eee; border-radius: 6px; padding: 10px; overflow-x: auto; }
.text-diff .add { color: #087f23; background: #e9f9ee; }
.text-diff .del { color: #b00020; background: #fdecea; text-decoration: line-through; }
.no-diff { color: #888; font-size: 13px; }
.visual-row { display: flex; gap: 16px; flex-wrap: wrap; margin-top: 10px; }
.visual-row figure { margin: 0; flex: 1 1 320px; }
.visual-row img { max-width: 100%; border: 1px solid #ddd; border-radius: 4px; }
.visual-row figcaption { font-size: 12px; color: #666; margin-top: 4px; }
.img-list { display: flex; gap: 12px; flex-wrap: wrap; margin-top: 8px; }
.img-list figure { margin: 0; text-align: center; }
.img-list img { max-height: 90px; border: 1px solid #ddd; border-radius: 4px; background: #fff; }
.img-list figcaption { font-size: 11px; color: #666; }
.section-title { font-size: 13px; font-weight: 600; margin: 14px 0 4px; color: #333; }
"""


def render_report(results: list[PageResult], old_path: Path, new_path: Path, out_dir: Path) -> Path:
    total_pages = len(results)
    text_changed_pages = sum(1 for r in results if r.text_changed)
    visual_changed_pages = sum(1 for r in results if r.visual_changed)
    image_changed_pages = sum(1 for r in results if r.added_images or r.removed_images)
    only_pages = [r for r in results if r.only_in]

    summary_rows = []
    for r in results:
        if r.only_in:
            status = f'僅存在於{"舊檔" if r.only_in == "old" else "新檔"}'
        else:
            parts = []
            if r.text_changed:
                parts.append("文字差異")
            if r.visual_changed:
                parts.append(f"版面差異({r.visual_box_count}處)")
            if r.added_images or r.removed_images:
                parts.append(f"圖片差異(+{len(r.added_images)}/-{len(r.removed_images)})")
            status = "、".join(parts) if parts else "相同"
        summary_rows.append(
            f'<tr><td><a href="#page-{r.index}">第 {r.index} 頁</a></td><td>{escape(status)}</td></tr>'
        )

    page_sections = []
    for r in results:
        if r.only_in:
            side = "old_image_rel" if r.only_in == "old" else "diff_image_rel"
            img_rel = getattr(r, side)
            page_sections.append(f"""
            <section class="page-block" id="page-{r.index}">
              <h2>第 {r.index} 頁 <span class="badge changed">僅存在於{"舊檔" if r.only_in == "old" else "新檔"}</span></h2>
              <div class="visual-row">
                <figure><img src="{img_rel}"><figcaption>此頁內容</figcaption></figure>
              </div>
            </section>
            """)
            continue

        text_block = (
            f'<div class="text-diff">{r.text_diff_html}</div>'
            if r.text_diff_html else '<p class="no-diff">（文字內容相同）</p>'
        )

        def img_figures(items, label):
            if not items:
                return ""
            figs = "".join(
                f'<figure><img src="{it["rel"]}"><figcaption>{it["w"]}x{it["h"]}</figcaption></figure>'
                for it in items
            )
            return f'<div class="section-title">{label}（{len(items)} 張）</div><div class="img-list">{figs}</div>'

        images_block = img_figures(r.added_images, "新增的圖片／圖示") + img_figures(r.removed_images, "刪除的圖片／圖示")
        if not images_block:
            images_block = '<p class="no-diff">（內嵌圖片無差異）</p>'

        text_badge = '<span class="badge changed">有差異</span>' if r.text_changed else '<span class="badge same">相同</span>'
        visual_badge = (
            f'<span class="badge changed">{r.visual_box_count} 處差異</span>'
            if r.visual_changed else '<span class="badge same">相同</span>'
        )

        page_sections.append(f"""
        <section class="page-block" id="page-{r.index}">
          <h2>第 {r.index} 頁</h2>
          <div class="section-title">文字差異 {text_badge}</div>
          {text_block}
          <div class="section-title">版面／圖形視覺差異 {visual_badge}（紅框標示新檔中不同之處）</div>
          <div class="visual-row">
            <figure><img src="{r.old_image_rel}"><figcaption>舊檔</figcaption></figure>
            <figure><img src="{r.diff_image_rel}"><figcaption>新檔（差異已用紅框標示）</figcaption></figure>
          </div>
          <div class="section-title">內嵌圖片／圖示差異</div>
          {images_block}
        </section>
        """)

    html = f"""<!doctype html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8">
<title>PDF 比較報告</title>
<style>{REPORT_CSS}</style>
</head>
<body>
<header>
  <h1>PDF 比較報告</h1>
  <p>舊檔：{escape(str(old_path.name))}</p>
  <p>新檔：{escape(str(new_path.name))}</p>
</header>
<main>
  <div class="summary">
    <p>共 {total_pages} 頁比對結果；{text_changed_pages} 頁文字有差異、{visual_changed_pages} 頁版面有視覺差異、
    {image_changed_pages} 頁圖片／圖示有差異{f'、{len(only_pages)} 頁僅存在於單一檔案' if only_pages else ''}。</p>
    <table>
      <tr><th>頁碼</th><th>差異摘要</th></tr>
      {''.join(summary_rows)}
    </table>
  </div>
  {''.join(page_sections)}
</main>
</body>
</html>
"""
    report_path = out_dir / "report.html"
    report_path.write_text(html, encoding="utf-8")
    return report_path


def print_console_summary(results: list[PageResult]) -> None:
    print("\n=== PDF 差異摘要 ===")
    for r in results:
        if r.only_in:
            label = "舊檔" if r.only_in == "old" else "新檔"
            print(f"第 {r.index:>3} 頁：僅存在於{label}")
            continue
        bits = []
        bits.append("文字有差異" if r.text_changed else "文字相同")
        bits.append(f"版面差異 {r.visual_box_count} 處" if r.visual_changed else "版面相同")
        if r.added_images or r.removed_images:
            bits.append(f"圖片 +{len(r.added_images)}/-{len(r.removed_images)}")
        else:
            bits.append("圖片相同")
        print(f"第 {r.index:>3} 頁：" + "、".join(bits))
    print()


def main():
    parser = argparse.ArgumentParser(description="比較兩份 PDF 檔案的文字與圖片(圖示)差異")
    parser.add_argument("old_pdf", type=Path, help="舊版 PDF 檔案路徑")
    parser.add_argument("new_pdf", type=Path, help="新版 PDF 檔案路徑")
    parser.add_argument("-o", "--output", type=Path, default=Path("pdf_compare_report"),
                         help="輸出報告的資料夾（預設：pdf_compare_report）")
    parser.add_argument("--zoom", type=float, default=RENDER_ZOOM,
                         help="頁面轉圖片的縮放倍率，越大越清晰但速度越慢（預設 2.0）")
    args = parser.parse_args()

    if not args.old_pdf.exists():
        sys.exit(f"找不到檔案：{args.old_pdf}")
    if not args.new_pdf.exists():
        sys.exit(f"找不到檔案：{args.new_pdf}")

    args.output.mkdir(parents=True, exist_ok=True)

    print(f"比對中：{args.old_pdf.name}  vs  {args.new_pdf.name} ...")
    results = compare_pdfs(args.old_pdf, args.new_pdf, args.output, zoom=args.zoom)
    print_console_summary(results)

    report_path = render_report(results, args.old_pdf, args.new_pdf, args.output)
    print(f"報告已產生：{report_path.resolve()}")


if __name__ == "__main__":
    main()
