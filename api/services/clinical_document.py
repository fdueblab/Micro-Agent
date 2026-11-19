"""Bounded, page-addressable PDF extraction; OCR is evidence, never ground truth."""
import json
import shutil
import subprocess


def extract_document(folder):
    source = folder / "source.pdf"
    if not source.exists():
        return
    import pymupdf
    pages = folder / "pages"
    pages.mkdir(exist_ok=True)
    report = []
    ocr_count = 0
    with pymupdf.open(source) as document:
        for index, page in enumerate(document):
            if index >= 100:
                report.append({"warning": "超过 100 页，请分批提供目标章节"})
                break
            text = page.get_text(sort=True)
            (pages / f"page-{index+1}.txt").write_text(text, encoding="utf-8")
            item = {"page": index+1, "text": f"pages/page-{index+1}.txt", "ocr": "not_needed"}
            if (len(text.strip()) < 100 or page.get_images()) and ocr_count < 8:
                ocr_count += 1
                image = pages / f"page-{index+1}.png"
                page.get_pixmap(matrix=pymupdf.Matrix(1.5, 1.5)).save(image)
                item["image"] = image.name
                item["ocr"] = "unavailable"
                if shutil.which("tesseract"):
                    try:
                        result = subprocess.run(["tesseract", str(image), "stdout", "--psm", "11"],
                                                capture_output=True, timeout=20, encoding="utf-8", errors="replace")
                        if result.returncode == 0:
                            (pages / f"page-{index+1}-ocr.txt").write_text(result.stdout, encoding="utf-8")
                            item["ocr"] = "extracted_requires_review"
                    except subprocess.TimeoutExpired:
                        item["ocr"] = "timeout"
            report.append(item)
    (folder / "document_index.json").write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
