"""批量评测历史生成的算法模型：dev/batch_evaluate.py

扫描后端资源库 ioeb_backend/uploads/generated_algorithm/**/*_algorithm.py，
逐个执行评测系统的静态检查 + 独立子进程导入测试。

新登记的模型（2026-09 之后）同目录留存有配套资产，会自动复用：
- *_test.py            → Agent 自测试执行（agent_test_run）
- 同目录数据集文件      → holdout 评测与基线对比（csv/xlsx/json/txt）

文件名含连字符（如 ZWH--015-260728_algorithm.py）不是合法 Python 模块名，
评测前会复制到临时目录并转为合法模块名。

用法（在 Micro-Agent 根目录）：
    python dev/batch_evaluate.py                    # 默认扫描后端资源库
    python dev/batch_evaluate.py --dir <目录>        # 指定目录
    python dev/batch_evaluate.py --dataset <csv>     # 显式指定数据集（优先于同目录自动发现）
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import shutil
import sys
import tempfile
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from micro_agent.evaluation.evaluator import EvaluationInput, run_evaluation  # noqa: E402

DEFAULT_DIR = ROOT.parent / "ioeb_backend" / "uploads" / "generated_algorithm"

# 文件名 → 合法 Python 模块名
_INVALID = re.compile(r"[^0-9a-zA-Z_]")


def to_module_name(stem: str) -> str:
    name = _INVALID.sub("_", stem).strip("_").lower()
    if not name or name[0].isdigit():
        name = "alg_" + name
    return name


def guess_category(source: str) -> str:
    """从代码内容粗略推断算法类别（用于展示，不用于硬约束检查）。"""
    if re.search(r"classification_label|分类标签|classify", source):
        return "classification"
    if re.search(r"回归|regression|预测值|predicted_value", source):
        return "regression"
    if re.search(r"检测|detection|异常检测", source):
        return "detection"
    if re.search(r"聚类|cluster", source):
        return "clustering"
    if re.search(r"推荐|recommend", source):
        return "recommendation"
    return ""


_DATASET_EXTS = {".csv", ".xlsx", ".json", ".txt", ".tsv"}


def find_sibling_assets(code_path: Path) -> tuple[Path | None, Path | None]:
    """发现同目录留存的配套资产：测试文件与数据集。"""
    test_path = None
    dataset_path = None
    for f in sorted(code_path.parent.iterdir()):
        if f is code_path or not f.is_file():
            continue
        if f.suffix.lower() == ".py" and f.stem.endswith("_test"):
            test_path = test_path or f
        elif f.suffix.lower() in _DATASET_EXTS:
            dataset_path = dataset_path or f
    return test_path, dataset_path


async def evaluate_one(
    code_path: Path,
    work_root: Path,
    dataset_path: Path | None,
) -> dict:
    """评测单个模型：重命名为合法模块名后执行全部检查。"""
    source = code_path.read_text(encoding="utf-8", errors="replace")
    mod = to_module_name(code_path.stem)
    stage = work_root / mod
    stage.mkdir(parents=True, exist_ok=True)
    staged = stage / f"{mod}.py"
    shutil.copyfile(code_path, staged)

    # 自动复用同目录留存的测试文件与数据集
    sib_test, sib_dataset = find_sibling_assets(code_path)
    staged_test = None
    if sib_test is not None:
        staged_test = stage / f"{mod}_test.py"
        test_src = sib_test.read_text(encoding="utf-8", errors="replace")
        # 测试文件 import 的是生成时的模块名，改写为 staging 重命名后的模块名
        test_src = re.sub(
            r"^(\s*from\s+)[A-Za-z_][A-Za-z0-9_]*_algorithm(\s+import\b.*)$",
            rf"\g<1>{mod}\g<2>", test_src, flags=re.MULTILINE,
        )
        test_src = re.sub(
            r"^(\s*import\s+)[A-Za-z_][A-Za-z0-9_]*_algorithm\b",
            rf"\g<1>{mod}", test_src, flags=re.MULTILINE,
        )
        staged_test.write_text(test_src, encoding="utf-8")
    ds_path = dataset_path or sib_dataset

    inp = EvaluationInput(
        code_path=staged,
        test_path=staged_test,                 # 同目录留存的测试文件（如有）
        algorithm_category="",                 # 不做标签硬约束（当时参数未留存）
        category_params={},
        constraints=["single_file"],           # 平台单文件交付为已知约束
        dataset_path=ds_path,
        reference_texts=[],
        eval_dir=stage / "worker",
    )
    report = await run_evaluation(inp)
    report["source_file"] = str(code_path)
    report["module_name"] = mod
    report["guessed_category"] = guess_category(source)
    report["retained_test_file"] = str(sib_test) if sib_test else ""
    report["retained_dataset_file"] = str(ds_path) if (ds_path and dataset_path is None) else (
        str(dataset_path) if dataset_path else ""
    )
    return report


async def main() -> int:
    parser = argparse.ArgumentParser(description="批量评测历史生成的算法模型")
    parser.add_argument("--dir", default=str(DEFAULT_DIR), help="算法模型所在目录（递归扫描 *_algorithm.py）")
    parser.add_argument("--dataset", default=None, help="可选数据集路径（用于 holdout 评测）")
    parser.add_argument("--out-dir", default=str(ROOT / "workspace" / "temp" / "batch_eval_reports"),
                        help="逐模型 JSON 报告输出目录")
    args = parser.parse_args()

    scan_dir = Path(args.dir)
    if not scan_dir.is_dir():
        print(f"[错误] 目录不存在: {scan_dir}")
        return 2

    files = sorted(scan_dir.rglob("*_algorithm.py"))
    if not files:
        print(f"[错误] 未在 {scan_dir} 找到 *_algorithm.py 文件")
        return 2

    dataset_path = Path(args.dataset) if args.dataset else None
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"共发现 {len(files)} 个算法模型，开始评测...\n")
    reports: list[dict] = []
    with tempfile.TemporaryDirectory(prefix="batch_eval_") as td:
        work_root = Path(td)
        for i, f in enumerate(files, 1):
            name = f"{f.parent.name[:8]} / {f.name}"
            print(f"[{i}/{len(files)}] {name}")
            try:
                report = await evaluate_one(f, work_root, dataset_path)
            except Exception as e:  # noqa: BLE001
                report = {
                    "verdict": "error",
                    "verdict_label": "评测异常",
                    "summary": {},
                    "checks": [],
                    "source_file": str(f),
                    "error": str(e),
                }
            reports.append(report)
            s = report.get("summary") or {}
            print(
                f"    → {report.get('verdict_label')} | "
                f"passed={s.get('passed', '-')} warning={s.get('warning', '-')} "
                f"failed={s.get('failed', '-')} skipped={s.get('skipped', '-')} | "
                f"类别推断: {report.get('guessed_category') or '未知'}"
            )
            for c in report.get("checks", []):
                if c.get("status") in ("failed", "warning"):
                    print(f"      [{c['status_label']}] {c['name']}: {c['details'][:100]}")

    # 保存逐模型完整报告
    stamp = uuid.uuid4().hex[:8]
    summary_path = out_dir / f"batch_summary_{stamp}.json"
    summary_path.write_text(
        json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # 汇总统计
    verdicts: dict[str, int] = {}
    problem_counts: dict[str, int] = {}
    for r in reports:
        v = r.get("verdict", "error")
        verdicts[v] = verdicts.get(v, 0) + 1
        for c in r.get("checks", []):
            if c.get("status") == "failed":
                key = f"{c['name']}"
                problem_counts[key] = problem_counts.get(key, 0) + 1

    print("\n" + "=" * 60)
    print(f"批量评测完成：共 {len(reports)} 个模型")
    for v, n in sorted(verdicts.items(), key=lambda x: -x[1]):
        print(f"  {v}: {n} 个")
    if problem_counts:
        print("\n未通过项分布（问题热点）：")
        for name, n in sorted(problem_counts.items(), key=lambda x: -x[1]):
            print(f"  {name}: {n} 个模型")
    print(f"\n完整报告已保存: {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
