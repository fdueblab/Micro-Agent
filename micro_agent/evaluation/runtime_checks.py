"""运行时检查：在隔离子进程中真实执行生成代码。

1. import_check   — 干净子进程导入模块 + **属性探测**：静态收集代码中
                    （含函数体内）对导入模块的属性访问链，在子进程中逐条
                    触发，暴露「导入成功但实际调用会崩」的断链
2. agent_test_run — 执行 Agent 自己编写的测试文件（真实执行，不可伪造）
3. perturbation   — 扰动鲁棒性：向 main_process 喂 5 类脏输入
                    （空/None/类型错乱/极端值/垃圾键），统计崩溃数
4. holdout        — 在真实数据集上逐行调用 main_process，计算指标并与
                    trivial 基线（多数类/均值）对比，检测「测试过拟合」
                    与「不如基线」的无效模型
"""

from __future__ import annotations

import ast
import asyncio
import json
import sys
from pathlib import Path

from micro_agent.evaluation.schema import (
    CATEGORY_RUNTIME,
    CheckResult,
    STATUS_FAILED,
    STATUS_PASSED,
    STATUS_SKIPPED,
    STATUS_WARNING,
)

_LABEL_COL_NAMES = {"label", "target", "class", "y", "category", "标签", "类别", "分类"}
_VALUE_COL_NAMES = {"label", "target", "y", "value", "actual", "true", "目标值", "真实值", "实际值"}

_CLASSIFICATION_PRED_KEYS = [
    "classification_label", "label", "labels", "prediction", "predicted_label",
    "category", "risk_level", "class", "结论", "标签", "风险等级",
]
_REGRESSION_PRED_KEYS = [
    "prediction", "predicted_value", "value", "output", "result", "score", "预测值",
]

# holdout worker：独立脚本，在子进程中运行，通过 stdout JSON 返回结果。
# 使用 @TOKEN@ 占位符注入参数（避免 .format 与 Python 花括号冲突）。
_HOLDOUT_WORKER = r'''
import sys, json, csv, math

MOD_DIR = @MOD_DIR@
MOD_NAME = @MOD_NAME@
DATASET = @DATASET@
CATEGORY = @CATEGORY@
LABELS = @LABELS@
MAX_ROWS = 200

result = {"status": "skipped", "reason": "", "metrics": {}, "baseline": {}, "verdict": "", "errors": []}


def fail(reason):
    result["reason"] = reason
    print(json.dumps(result, ensure_ascii=False))
    sys.exit(0)


def load_rows():
    if DATASET.endswith(".csv") or DATASET.endswith(".txt"):
        with open(DATASET, "r", encoding="utf-8", errors="replace") as f:
            return list(csv.DictReader(f))
    if DATASET.endswith(".json"):
        with open(DATASET, "r", encoding="utf-8", errors="replace") as f:
            data = json.load(f)
        if isinstance(data, list):
            return [r for r in data if isinstance(r, dict)]
        if isinstance(data, dict):
            for key in ("data", "rows", "records", "items"):
                if isinstance(data.get(key), list):
                    return [r for r in data[key] if isinstance(r, dict)]
        return []
    if DATASET.endswith((".xlsx", ".xls")):
        try:
            import openpyxl
            wb = openpyxl.load_workbook(DATASET, read_only=True, data_only=True)
            ws = wb.active
            rows_iter = list(ws.iter_rows(values_only=True))
            wb.close()
            if not rows_iter:
                return []
            header = [str(c) if c is not None else "" for c in rows_iter[0]]
            out = []
            for r in rows_iter[1:]:
                out.append({header[i]: ("" if v is None else v) for i, v in enumerate(r)})
            return out
        except Exception:
            return []
    return []


def to_float(v):
    try:
        return float(str(v).strip())
    except (TypeError, ValueError):
        return None


rows = load_rows()
if not rows:
    fail("数据集无法解析为结构化数据行（支持 CSV/JSON/Excel/TXT）")
rows = rows[:MAX_ROWS]

# ── 标签列检测 ──
label_col = None
if CATEGORY == "classification":
    for col in rows[0].keys():
        if str(col).strip().lower() in _LABEL_NAMES:
            label_col = col
            break
    if label_col is None and LABELS:
        for col in rows[0].keys():
            vals = [str(r.get(col, "")) for r in rows[:50]]
            hit = sum(1 for v in vals if v in LABELS)
            if vals and hit / len(vals) >= 0.8:
                label_col = col
                break
    if label_col is None:
        fail("未在数据集中找到标签列（按列名或标签取值匹配均失败）")
else:
    for col in rows[0].keys():
        if str(col).strip().lower() in _VALUE_NAMES:
            vals = [to_float(r.get(col)) for r in rows[:50]]
            if vals and sum(1 for v in vals if v is not None) / len(vals) >= 0.8:
                label_col = col
                break
    if label_col is None:
        fail("未在数据集中找到数值型目标列")

# ── 加载被测模块 ──
sys.path.insert(0, MOD_DIR)
try:
    import importlib
    mod = importlib.import_module(MOD_NAME)
except Exception as e:
    fail(f"导入被测模块失败: {e}")

fn = getattr(mod, "main_process", None)
if fn is None:
    fail("被测模块缺少 main_process")

import inspect
sig = inspect.signature(fn)
params = [p for p in sig.parameters.values() if p.kind in (
    inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)]
if not params:
    fail("main_process 不接受任何参数，无法基于数据行逐行评测")


def call_with_row(row):
    # 策略 A：单参数（整行 dict）
    if len(params) == 1:
        return fn(dict(row))
    # 策略 B：按参数名传 kwargs
    kwargs = {k: v for k, v in row.items() if k in sig.parameters}
    missing = [p.name for p in params
               if p.default is inspect.Parameter.empty and p.name not in kwargs]
    if missing:
        raise TypeError("参数无法从数据行映射: " + ",".join(missing))
    return fn(**kwargs)


def extract_prediction(res):
    if isinstance(res, str):
        return res
    if isinstance(res, dict):
        keys = _CLS_KEYS if CATEGORY == "classification" else _REG_KEYS
        for k in keys:
            if k in res:
                v = res[k]
                if isinstance(v, list):
                    v = v[0] if v else None
                if v is None:
                    continue
                return v if isinstance(v, str) else str(v)
        for v in res.values():
            if isinstance(v, str):
                return v
    return None


# ── 逐行评测 ──
true_labels, pred_labels, errors = [], [], []
for row in rows:
    truth = row.get(label_col)
    try:
        res = call_with_row(row)
    except Exception as e:
        errors.append(str(e)[:120])
        continue
    pred = extract_prediction(res)
    if pred is None:
        errors.append("无法从返回值提取预测结果")
        continue
    if CATEGORY == "regression":
        pred = to_float(pred)
        truth_v = to_float(truth)
        if pred is None or truth_v is None:
            errors.append("预测值/真实值非数值")
            continue
        true_labels.append(truth_v)
        pred_labels.append(pred)
    else:
        true_labels.append(str(truth).strip())
        pred_labels.append(str(pred).strip())

usable = len(true_labels)
if usable < 10 or usable < len(rows) * 0.5:
    fail(f"main_process 调用成功率过低（{usable}/{len(rows)} 行成功），错误示例: {errors[:2]}")

# ── 指标与基线 ──
metrics, baseline, verdict = {}, {}, ""
if CATEGORY == "classification":
    correct = sum(1 for t, p in zip(true_labels, pred_labels) if t == p)
    acc = correct / usable
    from collections import Counter
    majority, _ = Counter(true_labels).most_common(1)[0]
    base_acc = sum(1 for t in true_labels if t == majority) / usable
    labels_present = sorted(set(true_labels) | set(pred_labels))
    f1s = []
    per_label = {}
    for lb in labels_present:
        tp = sum(1 for t, p in zip(true_labels, pred_labels) if t == lb and p == lb)
        fp = sum(1 for t, p in zip(true_labels, pred_labels) if t != lb and p == lb)
        fn_ = sum(1 for t, p in zip(true_labels, pred_labels) if t == lb and p != lb)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn_) if tp + fn_ else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        f1s.append(f1)
        per_label[lb] = {"precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4)}
    metrics = {"accuracy": round(acc, 4), "macro_f1": round(sum(f1s) / len(f1s), 4),
               "evaluated_rows": usable, "per_label": per_label}
    baseline = {"strategy": "多数类预测", "majority_class": majority, "accuracy": round(base_acc, 4)}
    if acc > base_acc + 0.02:
        verdict = "beats_baseline"
    elif acc >= base_acc - 0.02:
        verdict = "matches_baseline"
    else:
        verdict = "below_baseline"
else:
    mean_t = sum(true_labels) / usable
    mae = sum(abs(t - p) for t, p in zip(true_labels, pred_labels)) / usable
    rmse = math.sqrt(sum((t - p) ** 2 for t, p in zip(true_labels, pred_labels)) / usable)
    ss_res = sum((t - p) ** 2 for t, p in zip(true_labels, pred_labels))
    ss_tot = sum((t - mean_t) ** 2 for t in true_labels)
    r2 = 1 - ss_res / ss_tot if ss_tot else 0.0
    base_mae = sum(abs(t - mean_t) for t in true_labels) / usable
    base_rmse = math.sqrt(sum((t - mean_t) ** 2 for t in true_labels) / usable)
    metrics = {"mae": round(mae, 4), "rmse": round(rmse, 4), "r2": round(r2, 4), "evaluated_rows": usable}
    baseline = {"strategy": "均值预测", "mae": round(base_mae, 4), "rmse": round(base_rmse, 4)}
    if mae <= base_mae * 0.98:
        verdict = "beats_baseline"
    elif mae <= base_mae * 1.02:
        verdict = "matches_baseline"
    else:
        verdict = "below_baseline"

result.update({"status": "completed", "metrics": metrics, "baseline": baseline,
               "verdict": verdict, "label_column": label_col, "errors": errors[:5]})
print(json.dumps(result, ensure_ascii=False))
'''

_WORKER_CONSTS = (
    f"_LABEL_NAMES = {_LABEL_COL_NAMES!r}\n"
    f"_VALUE_NAMES = {_VALUE_COL_NAMES!r}\n"
    f"_CLS_KEYS = {_CLASSIFICATION_PRED_KEYS!r}\n"
    f"_REG_KEYS = {_REGRESSION_PRED_KEYS!r}\n"
)


async def _run_subprocess(args: list[str], timeout: float, cwd: str | Path | None = None) -> tuple[int, str, str]:
    """运行子进程，返回 (returncode, stdout, stderr)。超时返回 (-1, '', 超时说明)。"""
    try:
        proc = await asyncio.create_subprocess_exec(
            *args,
            cwd=str(cwd) if cwd else None,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return -1, "", f"子进程执行超时（{timeout:.0f}s）"
        return (
            proc.returncode or 0,
            stdout.decode("utf-8", errors="replace"),
            stderr.decode("utf-8", errors="replace"),
        )
    except Exception as e:  # noqa: BLE001
        return -1, "", f"子进程启动失败: {e}"


def _module_name(code_path: Path) -> str:
    return code_path.stem


# 属性探测跳过的顶层模块（标准库工具属性，探测无意义且拖慢）
_PROBE_SKIP_TOP = {"__future__"}


def _collect_attribute_probes(source: str, max_probes: int = 80) -> list[str]:
    """静态收集代码中对导入模块的属性访问链（含函数体内的）。

    将 `import mediapipe as mp` + `mp.solutions.pose`（无论出现在模块级
    还是函数体内）统一转换为绝对链 "mediapipe.solutions.pose"，
    供子进程导入后逐条触发，暴露「导入成功但实际访问会崩」的断链。
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    # 1) 绑定名 -> 模块路径（parts）
    alias_map: dict[str, list[str]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                parts = a.name.split(".")
                # `import X.Y` 绑定顶层名 X；`import X.Y as z` 绑定整个 X.Y
                alias_map[a.asname or parts[0]] = parts if a.asname else parts[:1]
        elif isinstance(node, ast.ImportFrom):
            if node.level and node.level > 0:
                continue  # 相对导入指向本地模块，由 single_file 约束检查负责
            mod = (node.module or "").split(".")
            if not mod or mod[0] in _PROBE_SKIP_TOP:
                continue
            for a in node.names:
                if a.name == "*":
                    continue
                alias_map[a.asname or a.name] = mod + [a.name]

    if not alias_map:
        return []

    # 2) 收集「绑定名.属性链」并转为绝对链（walk 会同时产出内外层链，set 去重）
    probe_set: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute):
            continue
        attrs: list[str] = []
        cur: ast.AST = node
        while isinstance(cur, ast.Attribute):
            attrs.append(cur.attr)
            cur = cur.value
        if not isinstance(cur, ast.Name):
            continue
        base = alias_map.get(cur.id)
        if not base:
            continue
        probe_set.add(".".join(base + list(reversed(attrs))))

    return sorted(probe_set)[:max_probes]


# 子进程探测脚本模板：先导入被测模块，再逐条触发属性访问链。
# 每条链先尝试作为子模块 import（覆盖真子模块与懒加载包），
# 失败再逐级 getattr（覆盖函数/类/常量属性）。
_PROBE_SCRIPT = """import sys, json, importlib
sys.path.insert(0, {mod_dir!r})
import {mod}
failures = []
for dotted in {probes!r}:
    try:
        importlib.import_module(dotted)
        continue
    except ImportError:
        pass
    except Exception as e:
        failures.append(dotted + ': ' + type(e).__name__ + ': ' + str(e)[:80])
        continue
    parts = dotted.split('.')
    try:
        obj = importlib.import_module(parts[0])
        for p in parts[1:]:
            obj = getattr(obj, p)
    except Exception as e:
        failures.append(dotted + ': ' + type(e).__name__ + ': ' + str(e)[:80])
print(json.dumps({{'probed': len({probes!r}), 'failures': failures}}))
"""


async def check_import(code_path: Path) -> CheckResult:
    mod_dir = str(code_path.parent)
    mod = _module_name(code_path)
    try:
        source = code_path.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        return CheckResult(
            id="import_check", name="导入测试（独立子进程）", category=CATEGORY_RUNTIME,
            status=STATUS_FAILED, details=f"无法读取源码: {e}",
        )
    probes = _collect_attribute_probes(source)
    script = _PROBE_SCRIPT.format(mod_dir=mod_dir, mod=mod, probes=probes)
    rc, out, err = await _run_subprocess(
        [sys.executable, "-c", script], timeout=60, cwd=mod_dir
    )

    payload = None
    for line in reversed(out.strip().splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                payload = json.loads(line)
                break
            except json.JSONDecodeError:
                continue

    if payload is not None and not payload.get("failures"):
        return CheckResult(
            id="import_check", name="导入测试（独立子进程）", category=CATEGORY_RUNTIME,
            status=STATUS_PASSED,
            details=f"模块导入成功；属性探测 {payload.get('probed', 0)} 项全部可用",
        )
    if payload is not None:
        failures = payload.get("failures") or []
        return CheckResult(
            id="import_check", name="导入测试（独立子进程）", category=CATEGORY_RUNTIME,
            status=STATUS_FAILED,
            details=f"模块可导入，但存在不可用属性（实际调用会崩溃）:",
            evidence=[f"探测失败: {f}" for f in failures[:10]],
        )
    # 子进程未产出 JSON：导入本身失败或超时
    detail_lines = (err or out).strip().splitlines()
    detail = detail_lines[-1] if detail_lines else "未知错误"
    return CheckResult(
        id="import_check", name="导入测试（独立子进程）", category=CATEGORY_RUNTIME,
        status=STATUS_FAILED, details=f"导入失败: {detail[:200]}",
    )


async def check_agent_tests(test_path: Path | None) -> CheckResult:
    if test_path is None or not Path(test_path).exists():
        return CheckResult(
            id="agent_test_run", name="自测试文件执行", category=CATEGORY_RUNTIME,
            status=STATUS_SKIPPED, details="未提供测试文件",
        )
    rc, out, err = await _run_subprocess(
        [sys.executable, str(test_path)], timeout=90, cwd=str(test_path.parent)
    )
    combined = (out + "\n" + err).strip()
    tail = "\n".join(combined.splitlines()[-8:])
    if rc == 0:
        snippet = ""
        for line in out.splitlines():
            if "通过" in line:
                snippet = line.strip()[:80]
                break
        return CheckResult(
            id="agent_test_run", name="自测试文件执行", category=CATEGORY_RUNTIME,
            status=STATUS_PASSED, details=snippet or "测试文件退出码 0",
        )
    return CheckResult(
        id="agent_test_run", name="自测试文件执行", category=CATEGORY_RUNTIME,
        status=STATUS_FAILED,
        details=f"测试文件执行失败（退出码 {rc}）:\n{tail[:800]}",
    )


# 扰动鲁棒性 worker：独立脚本，在子进程中运行，通过 stdout JSON 返回结果。
# 使用 @TOKEN@ 占位符注入参数（与 _HOLDOUT_WORKER 一致）。
_PERTURB_WORKER = r'''
import sys, json

MOD_DIR = @MOD_DIR@
MOD_NAME = @MOD_NAME@

try:
    mod = __import__(MOD_NAME)
except Exception as e:
    print(json.dumps({"status": "skipped",
                      "reason": "模块导入失败: " + type(e).__name__}, ensure_ascii=False))
    sys.exit(0)

entry = getattr(mod, "main_process", None)
if entry is None:
    print(json.dumps({"status": "skipped", "reason": "无 main_process 入口"}, ensure_ascii=False))
    sys.exit(0)

INPUTS = [
    ("空输入", {}),
    ("None值", {"amount": None, "text": None, "value": None, "data": None}),
    ("错误类型", {"amount": "abc", "text": 12345, "value": [], "data": {}}),
    ("极端数值", {"amount": 1e18, "value": -1e18, "count": 1000000000}),
    ("垃圾键", {"???!!!": "garbage", "": ""}),
]

results = []
for label, row in INPUTS:
    try:
        r = entry(row)
        ok = isinstance(r, dict)
        err = "" if ok else "返回非字典: " + type(r).__name__
        results.append({"label": label, "ok": ok, "err": err})
    except Exception as e:
        results.append({"label": label, "ok": False,
                        "err": type(e).__name__ + ": " + str(e)[:60]})

print(json.dumps({"status": "completed", "results": results}, ensure_ascii=False))
'''


async def check_perturbation(code_path: Path) -> CheckResult:
    """扰动鲁棒性检查：向 main_process 喂脏输入（空/None/类型错乱/极端值/垃圾键）。

    用户上传的数据集常含脏值，逐行调用场景下一次崩溃即整批失败。
    判定：≥3/5 崩溃 → failed（鲁棒性严重不足）；1~2/5 → warning；0 → passed。
    模块导入失败时由 import_check 负责，此处 skipped。
    """
    mod_dir = str(code_path.parent)
    mod = _module_name(code_path)
    script = _PERTURB_WORKER.replace("@MOD_DIR@", repr(mod_dir)).replace("@MOD_NAME@", repr(mod))
    rc, out, err = await _run_subprocess(
        [sys.executable, "-c", script], timeout=60, cwd=mod_dir
    )

    payload = None
    for line in reversed((out or "").strip().splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                payload = json.loads(line)
                break
            except json.JSONDecodeError:
                continue

    if payload is None:
        detail = (err or out).strip().splitlines()
        return CheckResult(
            id="perturbation_robustness", name="扰动鲁棒性检查", category=CATEGORY_RUNTIME,
            status=STATUS_FAILED,
            details=f"扰动测试子进程异常（超时或崩溃）: {(detail[-1] if detail else '未知')[:150]}",
        )
    if payload.get("status") == "skipped":
        return CheckResult(
            id="perturbation_robustness", name="扰动鲁棒性检查", category=CATEGORY_RUNTIME,
            status=STATUS_SKIPPED, details=f"跳过: {payload.get('reason', '')}",
        )

    results = payload.get("results") or []
    bad = [r for r in results if not r.get("ok")]
    evidence = [f"{r['label']}: {r.get('err', '')[:70]}" for r in bad]
    if len(bad) >= 3:
        return CheckResult(
            id="perturbation_robustness", name="扰动鲁棒性检查", category=CATEGORY_RUNTIME,
            status=STATUS_FAILED,
            details=f"5 类脏输入中 {len(bad)} 类导致崩溃/非法返回，鲁棒性严重不足",
            evidence=evidence[:5],
        )
    if bad:
        return CheckResult(
            id="perturbation_robustness", name="扰动鲁棒性检查", category=CATEGORY_RUNTIME,
            status=STATUS_WARNING,
            details=f"5 类脏输入中 {len(bad)} 类导致崩溃/非法返回，建议对脏值优雅降级",
            evidence=evidence[:5],
        )
    return CheckResult(
        id="perturbation_robustness", name="扰动鲁棒性检查", category=CATEGORY_RUNTIME,
        status=STATUS_PASSED, details="空输入/None/类型错乱/极端值/垃圾键 5 类脏输入全部优雅处理",
    )


async def check_holdout(
    code_path: Path,
    dataset_path: Path | None,
    *,
    algorithm_category: str,
    labels: list[str],
    eval_dir: Path,
) -> tuple[CheckResult, dict | None]:
    """holdout 评测：真实数据集上逐行调用 + trivial 基线对比。"""
    if dataset_path is None or not Path(dataset_path).exists():
        return CheckResult(
            id="holdout_evaluation", name="数据集真实评测与基线对比", category=CATEGORY_RUNTIME,
            status=STATUS_SKIPPED, details="未提供数据集",
        ), None
    if algorithm_category not in ("classification", "regression"):
        return CheckResult(
            id="holdout_evaluation", name="数据集真实评测与基线对比", category=CATEGORY_RUNTIME,
            status=STATUS_SKIPPED,
            details=f"类别 {algorithm_category or '未知'} 暂不支持自动数据集评测",
        ), None

    eval_dir.mkdir(parents=True, exist_ok=True)
    worker_path = eval_dir / "holdout_worker.py"
    worker_src = (
        _WORKER_CONSTS
        + _HOLDOUT_WORKER
        .replace("@MOD_DIR@", json.dumps(str(code_path.parent)))
        .replace("@MOD_NAME@", json.dumps(_module_name(code_path)))
        .replace("@DATASET@", json.dumps(str(dataset_path)))
        .replace("@CATEGORY@", json.dumps(algorithm_category))
        .replace("@LABELS@", json.dumps([str(l) for l in labels], ensure_ascii=False))
    )
    worker_path.write_text(worker_src, encoding="utf-8")

    rc, out, err = await _run_subprocess(
        [sys.executable, str(worker_path)], timeout=150
    )
    payload = None
    for line in reversed(out.strip().splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                payload = json.loads(line)
                break
            except json.JSONDecodeError:
                continue
    if payload is None:
        return CheckResult(
            id="holdout_evaluation", name="数据集真实评测与基线对比", category=CATEGORY_RUNTIME,
            status=STATUS_FAILED,
            details=f"评测脚本未能产出结果（rc={rc}）: {(err or out).strip()[:300]}",
        ), None

    if payload.get("status") != "completed":
        return CheckResult(
            id="holdout_evaluation", name="数据集真实评测与基线对比", category=CATEGORY_RUNTIME,
            status=STATUS_SKIPPED,
            details=f"跳过原因: {payload.get('reason', '未知')}",
        ), payload

    verdict = payload.get("verdict", "")
    metrics = payload.get("metrics") or {}
    baseline = payload.get("baseline") or {}
    if verdict == "beats_baseline":
        status, summary = STATUS_PASSED, "模型显著优于基线"
    elif verdict == "matches_baseline":
        status, summary = STATUS_WARNING, "模型与基线持平（可能未学到有效规律）"
    else:
        status, summary = STATUS_FAILED, "模型劣于基线（无效模型）"

    if algorithm_category == "classification":
        detail = (
            f"{summary}: accuracy={metrics.get('accuracy')} vs 基线({baseline.get('strategy')})"
            f"={baseline.get('accuracy')}，macro_f1={metrics.get('macro_f1')}，"
            f"评测 {metrics.get('evaluated_rows')} 行（标签列: {payload.get('label_column')}）"
        )
    else:
        detail = (
            f"{summary}: MAE={metrics.get('mae')} vs 基线 MAE={baseline.get('mae')}，"
            f"RMSE={metrics.get('rmse')}，R²={metrics.get('r2')}，"
            f"评测 {metrics.get('evaluated_rows')} 行"
        )
    return CheckResult(
        id="holdout_evaluation", name="数据集真实评测与基线对比", category=CATEGORY_RUNTIME,
        status=status, details=detail,
    ), payload
