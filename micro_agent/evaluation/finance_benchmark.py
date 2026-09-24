"""金融垂域评测基准引擎：标准想定上评测算法模型，与金融规则/朴素基线对比。

评测流程:
    1. 加载想定(scenarios.json)与合成数据集(CSV)
    2. 隔离子进程逐行调用被评算法的 main_process(仅喂 input_columns, 基线辅助列不喂)
    3. 计算垂域指标: 分类(PR-AUC/KS/Recall@P/Accuracy) 回归(MAPE/方向准确率)
    4. 同口径计算想定内置基线(大额阈值规则/朴素基线等)
    5. 结论: must_beat_baseline 指标须优于基线 + 期望阈值 + 模型崩溃率<10%

用法:
    python -m micro_agent.evaluation.finance_benchmark --code model.py --scenario S1_suspicious_txn
    python -m micro_agent.evaluation.finance_benchmark --code model.py --all --out report.json
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_BENCHMARK_DIR = Path(__file__).resolve().parents[2] / "workspace" / "benchmarks" / "finance"

_RUNNER = r'''
import csv, importlib, json, sys, traceback

tmpdir, modname, ds, input_cols_str, output_key, score_key = sys.argv[1:7]
input_cols = [c for c in input_cols_str.split(",") if c]
sys.path.insert(0, tmpdir)

def _convert(v: str):
    """CSV 字符串尽力转数值, 失败保留原字符串(脏值防御考察点)。"""
    try:
        f = float(v)
        return int(f) if f == int(f) and "." not in v and "e" not in v.lower() else f
    except (TypeError, ValueError):
        return v

mod = importlib.import_module(modname)
results = []
with open(ds, encoding="utf-8") as f:
    for i, row in enumerate(csv.DictReader(f)):
        inp = {k: _convert(row.get(k, "")) for k in input_cols}
        try:
            res = mod.main_process(inp)
            if not isinstance(res, dict):
                raise TypeError(f"main_process 返回非 dict: {type(res).__name__}")
            score = res.get(score_key)
            results.append({"i": i, "ok": True, "label": str(res.get(output_key)),
                            "score": float(score) if isinstance(score, (int, float)) else None})
        except Exception as e:
            results.append({"i": i, "ok": False, "err": f"{type(e).__name__}: {e}"})
print("@@RESULT@@")
print(json.dumps(results, ensure_ascii=False))
'''


# ---------- 指标 ----------

def _pr_auc(y: np.ndarray, s: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score
    return float(average_precision_score(y, s))


def _ks(y: np.ndarray, s: np.ndarray) -> float:
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y, s)
    return float(np.max(tpr - fpr)) if len(tpr) else 0.0


def _recall_at_precision(y: np.ndarray, s: np.ndarray, p_min: float) -> float:
    """精确率 >= p_min 前提下的最大召回(无满足阈值时返回 0)。"""
    best = 0.0
    for th in np.unique(s)[::-1]:
        pred = s >= th
        tp = int(np.sum(pred & (y == 1)))
        fp = int(np.sum(pred & (y == 0)))
        if tp + fp == 0:
            continue
        precision = tp / (tp + fp)
        if precision >= p_min:
            best = max(best, tp / max(int(np.sum(y == 1)), 1))
    return float(best)


def _mape(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - true) / np.abs(true)) * 100)


def _direction_acc(pred: np.ndarray, last: np.ndarray, true: np.ndarray) -> float:
    return float(np.mean(np.sign(pred - last) == np.sign(true - last)))


# ---------- 基线 ----------

def _safe_env(row: dict) -> dict:
    def contains_any(text: str, keys: list) -> bool:
        return any(k.lower() in str(text).lower() for k in keys)
    return {"__builtins__": {}, "contains_any": contains_any, **row}


def _eval_rule(expr: str, df: pd.DataFrame) -> np.ndarray:
    """在数据框上按行求值基线规则表达式(受限环境)。"""
    out = []
    for row in df.to_dict("records"):
        try:
            out.append(bool(eval(expr, _safe_env(row))))
        except Exception:
            out.append(False)
    return np.array(out, dtype=float)


def _to_float(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


# ---------- 主流程 ----------

def _run_model(code_path: Path, scenario: dict, dataset: Path) -> list[dict]:
    """隔离子进程逐行调用被评算法。"""
    with tempfile.TemporaryDirectory(prefix="finbench_") as td:
        tmp = Path(td)
        modname = f"finbench_{uuid.uuid4().hex[:8]}"
        shutil.copy(code_path, tmp / f"{modname}.py")
        (tmp / "_runner.py").write_text(_RUNNER, encoding="utf-8")
        proc = subprocess.run(
            [sys.executable, str(tmp / "_runner.py"), str(tmp), modname, str(dataset),
             ",".join(scenario["input_columns"]), scenario["output_key"],
             scenario.get("score_key", "")],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=600,
        )
        if "@@RESULT@@" not in proc.stdout:
            raise RuntimeError(f"评测子进程失败: {proc.stderr[-2000:]}")
        return json.loads(proc.stdout.split("@@RESULT@@", 1)[1])


def _classification_scores(results: list[dict], df: pd.DataFrame,
                           scenario: dict) -> tuple[np.ndarray, np.ndarray]:
    """模型正类得分(非正类标签取 1-confidence)与真值。"""
    pos = scenario["positive_class"]
    y, s = [], []
    lab_col = scenario["label_column"]
    for r in results:
        truth = 1 if str(df.iloc[r["i"]][lab_col]) == pos else 0
        if r["ok"] and r["score"] is not None:
            score = r["score"] if r["label"] == pos else 1.0 - r["score"]
        elif r["ok"]:
            score = 1.0 if r["label"] == pos else 0.0
        else:
            score = 0.0  # 崩溃行按最低分计, 同时计入错误率
        y.append(truth)
        s.append(score)
    return np.array(y, dtype=float), np.array(s, dtype=float)


def evaluate_scenario(code_path: Path, scenario: dict, data_dir: Path) -> dict:
    df = pd.read_csv(data_dir / scenario["dataset"])
    results = _run_model(code_path, scenario, data_dir / scenario["dataset"])
    n_err = sum(1 for r in results if not r["ok"])
    n = len(results)

    metrics_cfg = scenario["metrics"]
    is_regression = "target_column" in scenario

    # ---- 模型指标 ----
    model_m: dict[str, float] = {}
    if is_regression:
        pred = np.array([r.get("label") if r["ok"] else None for r in results], dtype=object)
        ok_mask = np.array([r["ok"] and r["label"] is not None for r in results])
        p = np.array([_to_float(x) for x in pred[ok_mask]])
        true = df.loc[np.where(ok_mask)[0], scenario["target_column"]].to_numpy(dtype=float)
        last = df.loc[np.where(ok_mask)[0], "last_close"].to_numpy(dtype=float)
        valid = ~(np.isnan(p) | np.isnan(true))
        if not valid.any():
            raise RuntimeError("模型无有效回归输出")
        model_m["mape"] = _mape(p[valid], true[valid])
        model_m["direction_accuracy"] = _direction_acc(p[valid], last[valid], true[valid])
        model_m["coverage"] = float(valid.mean())
    else:
        y, s = _classification_scores(results, df, scenario)
        if "pr_auc" in metrics_cfg:
            model_m["pr_auc"] = _pr_auc(y, s)
        if "ks" in metrics_cfg:
            model_m["ks"] = _ks(y, s)
        if "recall_at_precision_30" in metrics_cfg:
            model_m["recall_at_precision_30"] = _recall_at_precision(y, s, 0.30)
        if "recall_at_precision_50" in metrics_cfg:
            model_m["recall_at_precision_50"] = _recall_at_precision(y, s, 0.50)
        if "accuracy" in metrics_cfg:
            lab_col, pos = scenario["label_column"], scenario["positive_class"]
            correct = sum(1 for r in results if r["ok"] and (
                (r["label"] == pos) == (str(df.iloc[r["i"]][lab_col]) == pos)))
            model_m["accuracy"] = correct / n

    # ---- 基线指标(同口径) ----
    base = scenario["baseline"]
    base_m: dict[str, float] = {}
    if base["type"] == "naive":
        true = df[scenario["target_column"]].to_numpy(dtype=float)
        last = df["last_close"].to_numpy(dtype=float)
        base_m["mape"] = _mape(last, true)
        base_m["direction_accuracy"] = 0.0  # 朴素基线方向准确率恒为 0(无变化预测)
    else:
        s_base = _eval_rule(base["expr"], df)
        y = (df[scenario["label_column"]] == scenario["positive_class"]).to_numpy(dtype=int)
        if "pr_auc" in metrics_cfg:
            base_m["pr_auc"] = _pr_auc(y, s_base)
        if "ks" in metrics_cfg:
            base_m["ks"] = _ks(y, s_base)
        if "recall_at_precision_30" in metrics_cfg:
            base_m["recall_at_precision_30"] = _recall_at_precision(y, s_base, 0.30)
        if "recall_at_precision_50" in metrics_cfg:
            base_m["recall_at_precision_50"] = _recall_at_precision(y, s_base, 0.50)
        if "accuracy" in metrics_cfg:
            base_m["accuracy"] = float(np.mean((s_base >= 0.5) == (y == 1)))

    # ---- 结论 ----
    checks: list[dict] = []
    exp = scenario.get("expectations", {})
    lower_better = {"mape"}
    for m in exp.get("must_beat_baseline", []):
        if m not in model_m or m not in base_m:
            continue
        better = model_m[m] < base_m[m] if m in lower_better else model_m[m] > base_m[m]
        checks.append({"name": f"{m} 优于基线", "model": round(model_m[m], 4),
                       "baseline": round(base_m[m], 4), "pass": bool(better)})
    if "min_ks" in exp and "ks" in model_m:
        checks.append({"name": f"KS >= {exp['min_ks']}", "model": round(model_m["ks"], 4),
                       "baseline": "-", "pass": model_m["ks"] >= exp["min_ks"]})
    if "max_mape" in exp and "mape" in model_m:
        checks.append({"name": f"MAPE <= {exp['max_mape']}%", "model": round(model_m["mape"], 4),
                       "baseline": "-", "pass": model_m["mape"] <= exp["max_mape"]})
    if "min_direction_accuracy" in exp and "direction_accuracy" in model_m:
        checks.append({"name": f"方向准确率 >= {exp['min_direction_accuracy']}",
                       "model": round(model_m["direction_accuracy"], 4), "baseline": "-",
                       "pass": model_m["direction_accuracy"] >= exp["min_direction_accuracy"]})
    if "mape_within_baseline_ratio" in exp and "mape" in model_m and "mape" in base_m:
        ratio = model_m["mape"] / max(base_m["mape"], 1e-9)
        checks.append({"name": f"MAPE 不超朴素基线 {exp['mape_within_baseline_ratio']}倍",
                       "model": f"{ratio:.3f}x", "baseline": round(base_m["mape"], 4),
                       "pass": ratio <= exp["mape_within_baseline_ratio"]})
    err_rate = n_err / max(n, 1)
    checks.append({"name": "崩溃率 < 10%", "model": f"{err_rate:.1%}", "baseline": "-",
                   "pass": err_rate < 0.10})

    verdict = "达标" if all(c["pass"] for c in checks) else "未达标"
    return {
        "scenario": scenario["id"], "name": scenario["name"],
        "n_rows": n, "n_model_errors": n_err,
        "model_metrics": {k: round(v, 4) for k, v in model_m.items()},
        "baseline": {"description": base["description"],
                     "metrics": {k: round(v, 4) for k, v in base_m.items()}},
        "checks": checks, "verdict": verdict,
    }


def load_scenarios() -> list[dict]:
    cfg = json.loads((_BENCHMARK_DIR / "scenarios.json").read_text(encoding="utf-8"))
    return cfg["scenarios"]


def run_benchmark(code_path: Path, scenario_ids: list[str] | None = None,
                  out_path: Path | None = None) -> list[dict]:
    data_dir = _BENCHMARK_DIR / "datasets"
    if not (data_dir / "s1_suspicious_txn.csv").exists():
        print("数据集不存在, 先运行 gen_datasets.py 生成")
        sys.exit(2)
    reports = []
    for sc in load_scenarios():
        if scenario_ids and sc["id"] not in scenario_ids:
            continue
        print(f"评测想定 {sc['id']} ({sc['name']}) ...", flush=True)
        try:
            reports.append(evaluate_scenario(code_path, sc, data_dir))
        except Exception as e:
            reports.append({"scenario": sc["id"], "verdict": "执行失败", "error": str(e)})
    # 摘要
    print("\n" + "=" * 72)
    print(f"{'想定':<24}{'结论':<6}{'崩溃':<8}模型指标 / 基线")
    print("-" * 72)
    for r in reports:
        if r.get("verdict") == "执行失败":
            print(f"{r['scenario']:<24}{r['verdict']:<6}{r.get('error', '')[:60]}")
            continue
        mm = r["model_metrics"]; bm = r["baseline"]["metrics"]
        cmp_str = " | ".join(f"{k}={mm[k]}(基线{bm.get(k, '-')})" for k in list(mm)[:3])
        print(f"{r['scenario']:<24}{r['verdict']:<6}{r['n_model_errors']}/{r['n_rows']:<6}{cmp_str}")
        for c in r["checks"]:
            mark = "√" if c["pass"] else "×"
            print(f"    [{mark}] {c['name']}: {c['model']} vs {c['baseline']}")
    if out_path:
        Path(out_path).write_text(json.dumps(reports, ensure_ascii=False, indent=2),
                                  encoding="utf-8")
        print(f"\n完整报告: {out_path}")
    return reports


def main():
    ap = argparse.ArgumentParser(description="金融垂域评测基准")
    ap.add_argument("--code", required=True, help="被评算法 .py 文件")
    ap.add_argument("--scenario", help="想定 id (缺省全部)")
    ap.add_argument("--all", action="store_true", help="评测全部想定")
    ap.add_argument("--out", help="报告 JSON 输出路径")
    args = ap.parse_args()
    ids = [args.scenario] if args.scenario else None
    run_benchmark(Path(args.code), ids, Path(args.out) if args.out else None)


if __name__ == "__main__":
    main()
