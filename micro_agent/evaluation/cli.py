"""评测 CLI 入口。

用法（在 Micro-Agent 根目录）：
    python -m micro_agent.evaluation.cli --code temp/xxx_algorithm.py ...

也支持以脚本绝对路径运行（自动引导 sys.path）：
    python3 /path/to/Micro-Agent/micro_agent/evaluation/cli.py --code ...

参数支持 --inputs <json>（由生成流程写入，含 category/params/constraints/
dataset_path/references_dir），命令行显式参数优先。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

# 以脚本方式运行时，将仓库根目录加入 sys.path
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def _read_reference_file(path: Path) -> str:
    """读取参考资料文本：复用 api 层的提取逻辑（PDF/DOCX/ZIP/代码等）。"""
    try:
        from api.services.files import read_reference_text
        return read_reference_text(str(path))
    except Exception:  # noqa: BLE001 — api 层不可用时退化为纯文本读取
        try:
            return path.read_text(encoding="utf-8", errors="replace")[:4000]
        except Exception:  # noqa: BLE001
            return ""


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="算法模型确定性评测（反作弊/约束/契约/数据集/基线/IP）")
    p.add_argument("--code", required=True, help="被测算法 .py 文件路径")
    p.add_argument("--test", default=None, help="Agent 编写的测试文件路径")
    p.add_argument("--category", default="", help="算法类别（classification/...）")
    p.add_argument("--params-json", default=None, help="类别参数 JSON 字符串")
    p.add_argument("--constraints", default=None, help="约束列表，逗号分隔（如 no_gpu,single_file）")
    p.add_argument("--dataset", default=None, help="数据集文件路径（holdout 评测）")
    p.add_argument("--references-dir", default=None, help="参考资料目录（IP 相似度比对）")
    p.add_argument("--references-file", action="append", default=None, help="参考资料文件（可多次）")
    p.add_argument("--inputs", default=None, help="输入元数据 JSON 文件（生成流程自动写入）")
    p.add_argument("--out", default=None, help="完整报告 JSON 输出文件路径")
    return p


def resolve_input(args: argparse.Namespace):
    """合并 --inputs 文件与命令行参数（命令行优先）。"""
    meta: dict = {}
    if args.inputs:
        inputs_path = Path(args.inputs)
        if inputs_path.exists():
            meta = json.loads(inputs_path.read_text(encoding="utf-8"))
        else:
            print(f"[警告] inputs 文件不存在: {inputs_path}", file=sys.stderr)

    def pick(cli_val, meta_key, default=None):
        # 命令行参数仅非空时生效，否则回退到 inputs 元数据
        return cli_val if cli_val else meta.get(meta_key, default)

    params_raw = pick(args.params_json, "category_params", None)
    if isinstance(params_raw, dict):
        category_params = params_raw
    elif params_raw:
        try:
            category_params = json.loads(params_raw)
        except json.JSONDecodeError:
            category_params = {}
    else:
        category_params = {}

    constraints_raw = pick(args.constraints, "constraints", None)
    if isinstance(constraints_raw, list):
        constraints = [str(c) for c in constraints_raw]
    elif constraints_raw:
        constraints = [c.strip() for c in str(constraints_raw).split(",") if c.strip()]
    else:
        constraints = list(category_params.get("constraints") or [])

    dataset = pick(args.dataset, "dataset_path", None)
    references_dir = pick(args.references_dir, "references_dir", None)

    # 参考资料文本收集
    reference_texts: list[tuple[str, str]] = []
    ref_paths: list[Path] = []
    if references_dir and Path(references_dir).is_dir():
        ref_paths.extend(sorted(Path(references_dir).iterdir()))
    if args.references_file:
        ref_paths.extend(Path(f) for f in args.references_file)
    seen: set[Path] = set()
    for rp in ref_paths:
        if rp in seen or not rp.is_file():
            continue
        seen.add(rp)
        text = _read_reference_file(rp)
        if text.strip():
            reference_texts.append((rp.name, text))

    from micro_agent.evaluation.evaluator import EvaluationInput

    return EvaluationInput(
        code_path=Path(args.code),
        test_path=Path(args.test) if args.test else None,
        algorithm_category=str(pick(args.category, "algorithm_category", "") or ""),
        category_params=category_params,
        constraints=constraints,
        dataset_path=Path(dataset) if dataset else None,
        reference_texts=reference_texts,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        inp = resolve_input(args)
    except Exception as e:  # noqa: BLE001
        print(json.dumps({"error": f"参数解析失败: {e}"}, ensure_ascii=False))
        return 2

    from micro_agent.evaluation.evaluator import run_evaluation

    try:
        report = asyncio.run(run_evaluation(inp))
    except FileNotFoundError as e:
        print(json.dumps({"error": str(e)}, ensure_ascii=False))
        return 2
    except Exception as e:  # noqa: BLE001
        print(json.dumps({"error": f"评测执行失败: {e}"}, ensure_ascii=False))
        return 1

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    # stdout 输出紧凑摘要 + 完整 JSON（Agent 可直接解析）
    print("=== 评测摘要 ===")
    print(f"结论: {report.get('verdict_label')} | {json.dumps(report.get('summary'), ensure_ascii=False)}")
    for c in report.get("checks", []):
        if c.get("status") in ("failed", "warning"):
            print(f"[{c.get('status_label')}] {c.get('name')}: {c.get('details')}")
    print("=== 完整报告 JSON ===")
    print(json.dumps(report, ensure_ascii=False))

    # 退出码：不合格/需复核返回 1，合格返回 0（便于脚本判断）
    return 0 if report.get("verdict") == "qualified" else 1


if __name__ == "__main__":
    sys.exit(main())
