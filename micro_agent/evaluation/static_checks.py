"""静态检查：基于 AST 的确定性检查，不执行代码。

覆盖四类：
1. 反作弊扫描：random 核心逻辑、占位符注释、空壳实现、torch 仅用于随机张量
2. 技术约束校验：no_gpu / no_llm / no_training / pretrained_only / rule_based / single_file
3. 输出契约：main_process 入口、docstring、分类标签集、未使用依赖
4. 工程质量：依赖声明一致性（虚假/缺失声明）、外部命令依赖（subprocess 未保护）、
   吞异常兜底（except Exception 静默返回默认值）
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

from micro_agent.evaluation.schema import (
    CATEGORY_BASIC,
    CATEGORY_CHEATING,
    CATEGORY_CONTRACT,
    CATEGORY_QUALITY,
    CATEGORY_CONSTRAINT,
    CheckResult,
    STATUS_FAILED,
    STATUS_PASSED,
    STATUS_SKIPPED,
    STATUS_WARNING,
)

# ── 常量表 ──────────────────────────────────────────────────────────────

_RANDOM_FUNCS = {
    "choice", "randint", "uniform", "shuffle", "sample", "random",
    "randrange", "gauss", "normalvariate", "triangular",
}

_LLM_SDK_MODULES = {
    "openai", "dashscope", "anthropic", "zhipuai", "qianfan",
    "google.generativeai", "genai", "cohere", "llama_cpp",
}

_TRAINING_CALL_ATTRS = {"fit", "train", "backward", "train_step", "fit_transform"}

_MODEL_LOAD_FUNC_NAMES = {"from_pretrained", "YOLO", "torch.hub.load"}

_PLACEHOLDER_PATTERNS = [
    (re.compile(r"在实际应用中应该"), "「在实际应用中应该…」式注释代替实现"),
    (re.compile(r"模拟实现|简化模拟|此处模拟"), "「模拟…」式注释代替实现"),
    (re.compile(r"placeholder", re.IGNORECASE), "placeholder 占位符"),
    (re.compile(r"待实现|未实现|暂未实现"), "待实现标记"),
    (re.compile(r"此处省略|省略部分实现|代码省略"), "省略实现"),
    (re.compile(r"demo实现|示例实现|仅为演示"), "演示性实现"),
]

_HEAVY_IMPORTS = {
    "torch", "tensorflow", "keras", "transformers", "paddle",
    "mxnet", "jittor", "mindspore",
}

_TORCH_RANDOM_ONLY_REFS = {
    "rand", "randn", "randint", "randperm", "manual_seed",
    "Generator", "default_generator", "rand_like", "randn_like",
}

# main_process 核心入口名
_ENTRY_FUNC = "main_process"

# pip 包名 -> 导入名 映射（两者不一致的常见第三方库）
_PIP_TO_IMPORT = {
    "opencv-python": "cv2",
    "opencv-python-headless": "cv2",
    "opencv-contrib-python": "cv2",
    "scikit-learn": "sklearn",
    "scikit-image": "skimage",
    "pillow": "PIL",
    "pyyaml": "yaml",
    "beautifulsoup4": "bs4",
    "python-dateutil": "dateutil",
    "protobuf": "google.protobuf",
    "python-dotenv": "dotenv",
    "imbalanced-learn": "imblearn",
    "scipy": "scipy",
}

# subprocess / os 层面的外部命令调用
_EXTERNAL_CMD_ATTRS = {"run", "call", "check_call", "check_output", "Popen", "system", "popen"}
_EXTERNAL_CMD_BASES = {"subprocess", "sp", "os", "commands"}

# 重型资源构造调用（在 main_process 可达函数体内每次调用都会重建 → 性能缺陷）
_HEAVY_INIT_ATTRS = _MODEL_LOAD_FUNC_NAMES | {
    # mediapipe solutions 系列（旧版 API）
    "Pose", "FaceDetection", "FaceMesh", "Hands", "Holistic",
    "SelfieSegmentation", "ObjectDetection", "ImageClassifier",
    "ImageEmbedder", "TextClassifier", "TextEmbedder", "VideoClassifier",
    # opencv / onnx / hf
    "CascadeClassifier", "InferenceSession", "load_model", "readNet",
    "dnn_DetectionModel", "pipeline",
}

# 复杂度阈值（基于存量 14 模型证据标定：最大圈复杂度分布 3~24，最长函数 68 行，最深嵌套 5）
_COMPLEXITY_CC_WARN = 15      # 圈复杂度超过 → 警告
_COMPLEXITY_CC_FAIL = 25      # 圈复杂度超过 → 未通过
_COMPLEXITY_LEN_WARN = 80     # 函数超过该行数 → 警告
_COMPLEXITY_NEST_WARN = 6     # 嵌套深度达到 → 警告

# 结果缓存装饰器：被缓存的函数内构造重型资源不算性能缺陷
_CACHE_DECORATORS = {"lru_cache", "cache", "cached_property"}


# ── AST 辅助 ────────────────────────────────────────────────────────────

class _ModuleIndex:
    """对单个模块源码建立索引：函数定义、调用图、导入、字符串常量。"""

    def __init__(self, source: str):
        self.source = source
        self.tree = ast.parse(source)
        self.functions: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = {}
        self.call_graph: dict[str, set[str]] = {}
        self.random_calls: list[tuple[int, str]] = []   # (lineno, 函数名)
        self.training_calls: list[tuple[int, str]] = []  # (lineno, 属性名)
        self.string_literals: set[str] = set()
        self.string_list_literals: list[tuple[int, list[str]]] = []
        self.imported_bindings: dict[str, int] = {}  # 绑定名 -> 出现次数（导入语句计 1）
        self.import_toplevel: list[str] = []          # 顶层模块名
        self.relative_imports: list[int] = []
        self.llm_imports: list[str] = []
        self.torch_refs: set[str] = set()
        self._index()

    def _index(self) -> None:
        for node in ast.walk(self.tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self.functions[node.name] = node
                self.call_graph.setdefault(node.name, set())
                for sub in ast.walk(node):
                    if isinstance(sub, ast.Call):
                        fname = self._callee_name(sub)
                        if fname:
                            self.call_graph[node.name].add(fname)
                        attr = self._call_attr(sub)
                        if attr:
                            if attr in _RANDOM_FUNCS and self._is_random_source(sub):
                                self.random_calls.append((sub.lineno, node.name))
                            if attr in _TRAINING_CALL_ATTRS:
                                self.training_calls.append((sub.lineno, attr))
                        if self._is_torch_attr_call(sub):
                            self.torch_refs.add(attr or "")
                        if self._func_name_in(sub, _MODEL_LOAD_FUNC_NAMES):
                            self.training_calls.append((sub.lineno, self._callee_name(sub) or ""))
                    elif isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                        self.string_literals.add(sub.value)
                    elif isinstance(sub, (ast.List, ast.Tuple)):
                        strs = [
                            e.value for e in sub.elts
                            if isinstance(e, ast.Constant) and isinstance(e.value, str)
                        ]
                        if len(strs) == len(sub.elts) and len(strs) >= 2:
                            self.string_list_literals.append((sub.lineno, strs))
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    self.import_toplevel.append(alias.name.split(".")[0])
                    binding = alias.asname or alias.name.split(".")[0]
                    self.imported_bindings[binding] = self.imported_bindings.get(binding, 0) + 1
            elif isinstance(node, ast.ImportFrom):
                if node.level and node.level > 0:
                    self.relative_imports.append(node.lineno)
                mod = (node.module or "").split(".")[0]
                if mod:
                    self.import_toplevel.append(mod)
                    if mod in _LLM_SDK_MODULES:
                        self.llm_imports.append(mod)
                for alias in node.names:
                    binding = alias.asname or alias.name
                    self.imported_bindings[binding] = self.imported_bindings.get(binding, 0) + 1
                    # from random import choice / from torch import rand
                    if mod == "random" and alias.name in _RANDOM_FUNCS:
                        # 记为随机调用来源（在任意函数中出现 Name 调用时判定）
                        pass
            elif isinstance(node, ast.Call):
                # 模块级调用（不在函数内）也已在上面的函数遍历外，这里补充 torch 属性引用
                attr = self._call_attr(node)
                if self._is_torch_attr_call(node) and attr:
                    self.torch_refs.add(attr)

        # from random import X 后，Name 调用 X() 也算随机调用
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in _RANDOM_FUNCS:
                    # 判断是否 import 自 random
                    for imp in ast.walk(self.tree):
                        if isinstance(imp, ast.ImportFrom) and (imp.module or "") == "random":
                            if any(a.name == node.func.id for a in imp.names):
                                self.random_calls.append((node.lineno, self._enclosing_function(node)))

    # -- 辅助方法 --
    def _enclosing_function(self, target: ast.AST) -> str:
        for name, fn in self.functions.items():
            for sub in ast.walk(fn):
                if sub is target:
                    return name
        return "<module>"

    @staticmethod
    def _callee_name(call: ast.Call) -> str | None:
        f = call.func
        if isinstance(f, ast.Name):
            return f.id
        if isinstance(f, ast.Attribute):
            return f.attr
        return None

    @staticmethod
    def _call_attr(call: ast.Call) -> str | None:
        f = call.func
        return f.attr if isinstance(f, ast.Attribute) else None

    @staticmethod
    def _func_name_in(call: ast.Call, names: set[str]) -> bool:
        f = call.func
        if isinstance(f, ast.Name):
            return f.id in names
        if isinstance(f, ast.Attribute):
            return f.attr in names
        return False

    @staticmethod
    def _is_random_source(call: ast.Call) -> bool:
        """判断 Call 是否来自 random / np.random / rng 等随机源。"""
        f = call.func
        if not isinstance(f, ast.Attribute):
            return False
        # random.choice / rng.choice
        if isinstance(f.value, ast.Name) and f.value.id in ("random", "rng", "rdm"):
            return True
        # np.random.choice / numpy.random.choice
        if isinstance(f.value, ast.Attribute) and f.value.attr == "random":
            return True
        if isinstance(f.value, ast.Name) and f.value.id in ("np", "numpy"):
            # np.random 情况已覆盖；np.choice 不存在
            return False
        return False

    @staticmethod
    def _is_torch_attr_call(call: ast.Call) -> bool:
        f = call.func
        if not isinstance(f, ast.Attribute):
            return False
        if isinstance(f.value, ast.Name) and f.value.id in ("torch", "th"):
            return True
        if isinstance(f.value, ast.Attribute) and f.value.attr in ("torch", "th"):
            return True
        return False

    def reachable_from_entry(self) -> set[str]:
        """从 main_process 出发可达的函数集合（含自身）。"""
        if _ENTRY_FUNC not in self.functions:
            return set()
        visited = {_ENTRY_FUNC}
        queue = [_ENTRY_FUNC]
        while queue:
            cur = queue.pop()
            for callee in self.call_graph.get(cur, ()):
                if callee in self.functions and callee not in visited:
                    visited.add(callee)
                    queue.append(callee)
        return visited


# ── 检查实现 ────────────────────────────────────────────────────────────

def check_syntax(source: str) -> CheckResult | None:
    """语法编译检查。失败时返回 failed（其余 AST 检查由调用方跳过）。"""
    try:
        compile(source, "<algorithm>", "exec")
        return CheckResult(
            id="syntax_compile", name="语法编译检查", category=CATEGORY_BASIC,
            status=STATUS_PASSED, details="compile() 通过",
        )
    except SyntaxError as e:
        return CheckResult(
            id="syntax_compile", name="语法编译检查", category=CATEGORY_BASIC,
            status=STATUS_FAILED, details=f"语法错误: 第 {e.lineno} 行 {e.msg}",
        )


def check_random_core(idx: _ModuleIndex) -> CheckResult:
    """反作弊：random 是否被用作核心决策逻辑（main_process 可达函数内）。"""
    if not idx.random_calls:
        return CheckResult(
            id="anti_random_core", name="随机数核心逻辑检测", category=CATEGORY_CHEATING,
            status=STATUS_PASSED, details="未发现 random 随机决策逻辑",
        )
    reachable = idx.reachable_from_entry()
    core_hits = [f"{ln}:{fn}" for ln, fn in idx.random_calls if fn in reachable or fn == "<module>"]
    if core_hits:
        return CheckResult(
            id="anti_random_core", name="随机数核心逻辑检测", category=CATEGORY_CHEATING,
            status=STATUS_FAILED,
            details=f"main_process 可达路径中使用了随机决策（行:函数 {', '.join(core_hits[:5])}），"
                    "禁止用 random 作为分类/检测/预测的核心逻辑",
            evidence=core_hits,
        )
    return CheckResult(
        id="anti_random_core", name="随机数核心逻辑检测", category=CATEGORY_CHEATING,
        status=STATUS_WARNING,
        details=f"random 调用位于核心路径之外（{', '.join(f'{ln}:{fn}' for ln, fn in idx.random_calls[:3])}），请人工确认用途",
    )


def check_placeholders(source: str) -> CheckResult:
    """反作弊：占位符注释/字符串。"""
    hits: list[str] = []
    for lineno, line in enumerate(source.splitlines(), 1):
        for pattern, label in _PLACEHOLDER_PATTERNS:
            if pattern.search(line):
                hits.append(f"第 {lineno} 行: {label}")
                break
    if hits:
        return CheckResult(
            id="anti_placeholder", name="占位实现检测", category=CATEGORY_CHEATING,
            status=STATUS_FAILED,
            details="发现用注释/占位符代替真实实现的痕迹",
            evidence=hits[:10],
        )
    return CheckResult(
        id="anti_placeholder", name="占位实现检测", category=CATEGORY_CHEATING,
        status=STATUS_PASSED, details="未发现占位符实现",
    )


def check_stubs(idx: _ModuleIndex) -> CheckResult:
    """反作弊：函数体为 pass/... 的空壳实现。"""
    stubs: list[str] = []
    for name, fn in idx.functions.items():
        body = list(fn.body)
        # 跳过 docstring
        if (body and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)):
            body = body[1:]
        if not body:
            stubs.append(name)
            continue
        only_stub = all(
            (isinstance(s, ast.Pass)
             or (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant)
                 and s.value.value is Ellipsis))
            for s in body
        )
        if only_stub:
            stubs.append(name)
    if stubs:
        return CheckResult(
            id="anti_stub", name="空壳实现检测", category=CATEGORY_CHEATING,
            status=STATUS_FAILED,
            details=f"以下函数体为 pass/省略号空实现: {', '.join(stubs[:8])}",
            evidence=stubs,
        )
    return CheckResult(
        id="anti_stub", name="空壳实现检测", category=CATEGORY_CHEATING,
        status=STATUS_PASSED, details="所有函数均有真实实现",
    )


def check_torch_random_only(idx: _ModuleIndex) -> CheckResult:
    """反作弊：声明了 torch 但仅用于生成随机张量。"""
    if "torch" not in idx.import_toplevel or not idx.torch_refs:
        return CheckResult(
            id="torch_random_only", name="深度学习依赖真实性检测", category=CATEGORY_CHEATING,
            status=STATUS_SKIPPED, details="未导入 torch 或无 torch 调用",
        )
    non_random = idx.torch_refs - _TORCH_RANDOM_ONLY_REFS
    if not non_random:
        return CheckResult(
            id="torch_random_only", name="深度学习依赖真实性检测", category=CATEGORY_CHEATING,
            status=STATUS_FAILED,
            details="导入了 torch 但全部调用仅为随机张量生成（rand/randn 等），"
                    "属于与推理逻辑无关的虚假依赖",
        )
    return CheckResult(
        id="torch_random_only", name="深度学习依赖真实性检测", category=CATEGORY_CHEATING,
        status=STATUS_PASSED,
        details=f"torch 存在真实推理用途: {', '.join(sorted(non_random)[:6])}",
    )


# ── 约束检查 ────────────────────────────────────────────────────────────

def check_constraint_no_gpu(source: str) -> CheckResult:
    patterns = [r"\.cuda\s*\(", r"torch\.cuda", r"device\s*=\s*['\"]cuda", r"CUDA_VISIBLE_DEVICES"]
    hits = [
        f"第 {i} 行: {line.strip()[:60]}"
        for i, line in enumerate(source.splitlines(), 1)
        if any(re.search(p, line) for p in patterns)
    ]
    if hits:
        return CheckResult(
            id="constraint_no_gpu", name="约束校验: 纯 CPU 运行", category=CATEGORY_CONSTRAINT,
            status=STATUS_FAILED, details="代码中出现 GPU/CUDA 相关调用", evidence=hits[:6],
        )
    return CheckResult(
        id="constraint_no_gpu", name="约束校验: 纯 CPU 运行", category=CATEGORY_CONSTRAINT,
        status=STATUS_PASSED, details="未发现 GPU 依赖",
    )


def check_constraint_no_llm(idx: _ModuleIndex) -> CheckResult:
    if idx.llm_imports:
        return CheckResult(
            id="constraint_no_llm", name="约束校验: 不使用 LLM", category=CATEGORY_CONSTRAINT,
            status=STATUS_FAILED,
            details=f"导入了 LLM SDK: {', '.join(sorted(set(idx.llm_imports)))}",
        )
    return CheckResult(
        id="constraint_no_llm", name="约束校验: 不使用 LLM", category=CATEGORY_CONSTRAINT,
        status=STATUS_PASSED, details="未发现 LLM SDK 依赖",
    )


def check_constraint_no_training(idx: _ModuleIndex, *, pretrained_only: bool = False) -> CheckResult:
    cid = "constraint_pretrained_only" if pretrained_only else "constraint_no_training"
    cname = "约束校验: 仅预训练推理" if pretrained_only else "约束校验: 不进行训练"
    if idx.training_calls:
        hits = [f"第 {ln} 行: .{attr}()" for ln, attr in idx.training_calls]
        return CheckResult(
            id=cid, name=cname, category=CATEGORY_CONSTRAINT,
            status=STATUS_FAILED, details="发现训练/微调类调用", evidence=hits[:6],
        )
    return CheckResult(
        id=cid, name=cname, category=CATEGORY_CONSTRAINT,
        status=STATUS_PASSED, details="未发现 fit/train/backward 调用",
    )


def check_constraint_rule_based(source: str) -> CheckResult:
    patterns = [
        r"torch\.load\s*\(", r"joblib\.load\s*\(", r"pickle\.load\s*\(",
        r"from_pretrained\s*\(", r"\bYOLO\s*\(", r"torch\.hub\.load", r"load_model\s*\(",
    ]
    hits = [
        f"第 {i} 行: {line.strip()[:60]}"
        for i, line in enumerate(source.splitlines(), 1)
        if any(re.search(p, line) for p in patterns)
    ]
    if hits:
        return CheckResult(
            id="constraint_rule_based", name="约束校验: 纯规则方法", category=CATEGORY_CONSTRAINT,
            status=STATUS_FAILED, details="发现预训练模型/权重加载调用", evidence=hits[:6],
        )
    return CheckResult(
        id="constraint_rule_based", name="约束校验: 纯规则方法", category=CATEGORY_CONSTRAINT,
        status=STATUS_PASSED, details="未发现模型权重加载",
    )


def check_constraint_single_file(idx: _ModuleIndex, code_dir: Path) -> CheckResult:
    hits: list[str] = []
    if idx.relative_imports:
        hits.append(f"存在相对导入（行 {', '.join(map(str, idx.relative_imports[:3]))}）")
    for top in set(idx.import_toplevel):
        if (code_dir / f"{top}.py").exists() or (code_dir / top / "__init__.py").exists():
            hits.append(f"导入同目录本地模块: {top}")
    if hits:
        return CheckResult(
            id="constraint_single_file", name="约束校验: 单文件实现", category=CATEGORY_CONSTRAINT,
            status=STATUS_FAILED, details="代码依赖了其他本地文件", evidence=hits,
        )
    return CheckResult(
        id="constraint_single_file", name="约束校验: 单文件实现", category=CATEGORY_CONSTRAINT,
        status=STATUS_PASSED, details="无本地模块依赖",
    )


# ── 契约与规范检查 ──────────────────────────────────────────────────────

def check_entry_contract(idx: _ModuleIndex) -> CheckResult:
    fn = idx.functions.get(_ENTRY_FUNC)
    if fn is None:
        return CheckResult(
            id="entry_contract", name="main_process 入口契约", category=CATEGORY_CONTRACT,
            status=STATUS_FAILED, details="缺少 main_process 主入口函数",
        )
    problems: list[str] = []
    has_doc = (
        fn.body and isinstance(fn.body[0], ast.Expr)
        and isinstance(fn.body[0].value, ast.Constant)
        and isinstance(fn.body[0].value.value, str)
    )
    if not has_doc:
        problems.append("缺少 Google 风格 docstring")
    if fn.returns is None:
        problems.append("缺少返回值类型注解")
    unannotated = [
        a.arg for a in fn.args.args
        if a.annotation is None and a.default is None
    ]
    if unannotated:
        problems.append(f"参数缺少类型注解: {', '.join(unannotated)}")
    if problems:
        return CheckResult(
            id="entry_contract", name="main_process 入口契约", category=CATEGORY_CONTRACT,
            status=STATUS_WARNING,
            details="；".join(problems),
        )
    return CheckResult(
        id="entry_contract", name="main_process 入口契约", category=CATEGORY_CONTRACT,
        status=STATUS_PASSED, details="入口函数、docstring、类型注解齐全",
    )


def check_label_contract(idx: _ModuleIndex, labels: list[str]) -> CheckResult:
    """分类标签契约：用户指定标签必须全部出现，且不得额外引入未定义标签。"""
    if not labels:
        return CheckResult(
            id="label_contract", name="分类标签契约", category=CATEGORY_CONTRACT,
            status=STATUS_SKIPPED, details="用户未指定分类标签",
        )
    user_set = {str(l).strip() for l in labels}
    missing = [l for l in user_set if l not in idx.string_literals]
    extra: list[str] = []
    # 候选标签列表：字符串列表字面量中若多数元素命中用户标签但含未定义项，则视为越界
    for _, strs in idx.string_list_literals:
        sset = set(strs)
        overlap = len(sset & user_set)
        if overlap >= max(2, len(sset) // 2):
            extra.extend(sorted(sset - user_set))
    extra = sorted(set(extra))
    if missing or extra:
        parts = []
        if missing:
            parts.append(f"缺少标签: {', '.join(missing)}")
        if extra:
            parts.append(f"引入未定义标签: {', '.join(extra[:8])}")
        return CheckResult(
            id="label_contract", name="分类标签契约", category=CATEGORY_CONTRACT,
            status=STATUS_FAILED, details="；".join(parts),
        )
    return CheckResult(
        id="label_contract", name="分类标签契约", category=CATEGORY_CONTRACT,
        status=STATUS_PASSED,
        details=f"标签集与用户定义一致（{len(user_set)} 类）",
    )


def check_unused_imports(source: str, idx: _ModuleIndex) -> CheckResult:
    """未使用导入检测；重度依赖（torch 等）未使用时特别提示。"""
    unused: list[str] = []
    for binding in idx.imported_bindings:
        # 统计绑定名在源码中除导入行外的出现次数
        uses = len(re.findall(rf"\b{re.escape(binding)}\b", source)) - idx.imported_bindings[binding]
        if uses <= 0:
            unused.append(binding)
    heavy_unused = [u for u in unused if u in _HEAVY_IMPORTS]
    if heavy_unused:
        return CheckResult(
            id="unused_imports", name="依赖相关性检测", category=CATEGORY_QUALITY,
            status=STATUS_FAILED,
            details=f"导入了重度依赖但未实际使用: {', '.join(heavy_unused)}（违反「禁止无关依赖」要求）",
        )
    if unused:
        return CheckResult(
            id="unused_imports", name="依赖相关性检测", category=CATEGORY_QUALITY,
            status=STATUS_WARNING, details=f"存在未使用的导入: {', '.join(unused[:8])}",
        )
    return CheckResult(
        id="unused_imports", name="依赖相关性检测", category=CATEGORY_QUALITY,
        status=STATUS_PASSED, details="所有导入均有使用",
    )


def _parse_declared_deps(source: str) -> list[tuple[str, int]]:
    """从模块 docstring 的「依赖清单」段解析声明的 pip 依赖名。

    支持两种格式：
      依赖清单:
      opencv-python>=4.8.0
      mediapipe>=0.10.0
    或单行逗号分隔：依赖清单: numpy>=1.0, pandas
    """
    deps: list[tuple[str, int]] = []
    in_section = False
    # 「无依赖」占位写法：None / 无 / - / n/a
    _no_dep = {"none", "无", "-", "n/a", "na"}
    for lineno, line in enumerate(source.splitlines(), 1):
        stripped = line.strip().strip('"').strip("'").strip()
        m = re.match(r"^依赖清单\s*[:：]\s*(.*)$", stripped)
        if m:
            in_section = True
            for part in m.group(1).split(","):
                name = re.split(r"[<>=!~;\[\s(]", part.strip())[0]
                if name and name.lower() not in _no_dep:
                    deps.append((name, lineno))
            continue
        if not in_section:
            continue
        if not stripped or stripped.startswith('"""') or stripped == "'''":
            break
        name = re.split(r"[<>=!~;\[\s,#(]", stripped)[0]
        if not name:
            break
        if name.lower() in _no_dep:
            break
        deps.append((name, lineno))
    return deps


def check_dependency_declaration(
    source: str, idx: _ModuleIndex, code_dir: Path
) -> CheckResult:
    """依赖声明一致性：头部「依赖清单」与实际 import 双向比对。

    - 声明但未 import 且未以命令行方式使用 → 虚假依赖（failed）
    - 使用了第三方库但未声明 → 缺失声明（warning）
    """
    declared = _parse_declared_deps(source)
    imported_lower = {m.lower() for m in idx.import_toplevel}

    def is_local_or_stdlib(top: str) -> bool:
        if top in sys.stdlib_module_names:
            return True
        return (code_dir / f"{top}.py").exists() or (code_dir / top / "__init__.py").exists()

    third_party = sorted({
        m for m in set(idx.import_toplevel) if not is_local_or_stdlib(m.split(".")[0])
    })

    # 1) 声明但未使用（虚假依赖）；以命令行方式使用（名字出现在字符串中）除外
    false_deps: list[str] = []
    cli_used: list[str] = []
    covered_imports: set[str] = set()
    for name, ln in declared:
        import_name = _PIP_TO_IMPORT.get(name.lower(), name)
        top = import_name.split(".")[0].lower()
        if top in imported_lower:
            covered_imports.add(top)
            continue
        if name in idx.string_literals or import_name in idx.string_literals:
            cli_used.append(f"{name}（第 {ln} 行声明，仅作命令行工具使用）")
            continue
        false_deps.append(f"第 {ln} 行: {name}")

    # 2) 实际使用但未声明
    undeclared = [m for m in third_party if m.lower() not in covered_imports]

    if false_deps:
        parts = [f"虚假依赖声明: {', '.join(false_deps[:6])}"]
        if cli_used:
            parts.append("命令行工具声明: " + "; ".join(cli_used[:3]))
        return CheckResult(
            id="dependency_declaration", name="依赖声明一致性", category=CATEGORY_QUALITY,
            status=STATUS_FAILED,
            details="；".join(parts) + "。声明的依赖必须被代码真实使用",
            evidence=false_deps,
        )
    if not declared and third_party:
        return CheckResult(
            id="dependency_declaration", name="依赖声明一致性", category=CATEGORY_QUALITY,
            status=STATUS_WARNING,
            details=f"使用了第三方库但头部缺少「依赖清单」声明: {', '.join(third_party[:8])}",
        )
    if undeclared:
        return CheckResult(
            id="dependency_declaration", name="依赖声明一致性", category=CATEGORY_QUALITY,
            status=STATUS_WARNING,
            details=f"使用了但未声明的依赖: {', '.join(undeclared[:8])}",
        )
    if not declared:
        return CheckResult(
            id="dependency_declaration", name="依赖声明一致性", category=CATEGORY_QUALITY,
            status=STATUS_PASSED, details="无第三方依赖，无需声明",
        )
    return CheckResult(
        id="dependency_declaration", name="依赖声明一致性", category=CATEGORY_QUALITY,
        status=STATUS_PASSED,
        details=f"依赖声明与实际使用一致（{len(declared)} 项）"
        + (f"；命令行工具: {len(cli_used)} 项" if cli_used else ""),
    )


def check_external_commands(idx: _ModuleIndex) -> CheckResult:
    """外部命令依赖：subprocess/os.system 调用外部工具。

    - 未用 try/except 保护 → 外部工具缺失时直接崩溃（failed）
    - 已保护 → 仍有环境依赖（warning，要求在 limitations 中说明）
    """
    parent: dict[int, ast.AST] = {}
    for node in ast.walk(idx.tree):
        for child in ast.iter_child_nodes(node):
            parent[id(child)] = node

    calls: list[tuple[int, str, bool]] = []  # (lineno, 名称, 是否受 try 保护)
    for node in ast.walk(idx.tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        if not isinstance(f, ast.Attribute) or f.attr not in _EXTERNAL_CMD_ATTRS:
            continue
        base = f.value
        base_name = base.id if isinstance(base, ast.Name) else (
            base.attr if isinstance(base, ast.Attribute) else "")
        if base_name not in _EXTERNAL_CMD_BASES:
            continue
        guarded = False
        cur = parent.get(id(node))
        while cur is not None:
            if isinstance(cur, ast.Try):
                guarded = True
                break
            cur = parent.get(id(cur))
        calls.append((node.lineno, f"{base_name}.{f.attr}()", guarded))

    if not calls:
        return CheckResult(
            id="external_command", name="外部命令依赖检测", category=CATEGORY_QUALITY,
            status=STATUS_PASSED, details="未使用 subprocess/外部命令行工具",
        )
    unguarded = [f"第 {ln} 行: {name} 未用 try/except 保护，外部工具缺失时将直接崩溃"
                 for ln, name, g in calls if not g]
    if unguarded:
        return CheckResult(
            id="external_command", name="外部命令依赖检测", category=CATEGORY_QUALITY,
            status=STATUS_FAILED,
            details="存在未保护的外部命令调用（算法隐含要求用户预装系统级工具）",
            evidence=unguarded[:6],
        )
    guarded_hits = [f"第 {ln} 行: {name}（已保护）" for ln, name, _ in calls]
    return CheckResult(
        id="external_command", name="外部命令依赖检测", category=CATEGORY_QUALITY,
        status=STATUS_WARNING,
        details="依赖外部命令行工具（已有 try/except 保护），须在 limitations 中说明环境要求",
        evidence=guarded_hits[:6],
    )


_LOGGING_FUNC_NAMES = {"print", "info", "warning", "error", "debug", "exception", "log", "warn"}


def check_swallowed_exceptions(idx: _ModuleIndex) -> CheckResult:
    """吞异常兜底：except Exception 后仅静默返回默认值，无任何记录/上抛。"""
    hits: list[str] = []

    def _has_trace(node: ast.AST) -> bool:
        for sub in ast.walk(node):
            if isinstance(sub, ast.Raise):
                return True
            if (isinstance(sub, ast.Expr) and isinstance(sub.value, ast.Call)):
                f = sub.value.func
                fname = f.attr if isinstance(f, ast.Attribute) else (
                    f.id if isinstance(f, ast.Name) else None)
                if fname in _LOGGING_FUNC_NAMES:
                    return True
        return False

    for node in ast.walk(idx.tree):
        if not isinstance(node, ast.Try):
            continue
        for h in node.handlers:
            broad = h.type is None or (
                isinstance(h.type, ast.Name) and h.type.id in ("Exception", "BaseException"))
            if not broad:
                continue
            if not _has_trace(h):
                hits.append(f"第 {h.lineno} 行: except Exception 静默吞掉错误（无日志/无上抛）")

    if hits:
        return CheckResult(
            id="swallowed_exception", name="吞异常兜底检测", category=CATEGORY_QUALITY,
            status=STATUS_WARNING,
            details="发现 except Exception 后静默返回默认值的模式，会掩盖真实错误，请记录错误信息",
            evidence=hits[:8],
        )
    return CheckResult(
        id="swallowed_exception", name="吞异常兜底检测", category=CATEGORY_QUALITY,
        status=STATUS_PASSED, details="未发现吞异常式静默失败",
    )


def _cyclomatic(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> int:
    """计算函数圈复杂度（分支/循环/异常处理/布尔短路各 +1）。"""
    cc = 1
    for node in ast.walk(fn):
        if isinstance(node, (ast.If, ast.For, ast.AsyncFor, ast.While,
                             ast.ExceptHandler, ast.Assert, ast.IfExp)):
            cc += 1
        elif isinstance(node, ast.BoolOp):
            cc += len(node.values) - 1
        elif isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            cc += sum(1 for g in node.generators if g.ifs)
    return cc


def _max_nesting(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> int:
    """计算函数内最大嵌套深度（if/for/while/try/with 计一层）。"""
    best = 0

    def rec(node: ast.AST, depth: int) -> None:
        nonlocal best
        if isinstance(node, (ast.If, ast.For, ast.AsyncFor, ast.While,
                             ast.Try, ast.With, ast.AsyncWith)):
            depth += 1
            best = max(best, depth)
        for child in ast.iter_child_nodes(node):
            rec(child, depth)

    rec(fn, 0)
    return best


def check_complexity(idx: _ModuleIndex) -> CheckResult:
    """复杂度检查：圈复杂度 / 函数长度 / 嵌套深度。

    阈值基于存量 14 模型证据标定（存量最大圈复杂度 3~24、最长函数 68 行、
    最深嵌套 5），确保正常模型不误报，超标者给出定位证据。
    """
    evidence: list[str] = []
    worst = ("", 0, 0, 0)  # (函数名, cc, 长度, 嵌套)
    for name, fn in idx.functions.items():
        cc = _cyclomatic(fn)
        length = (fn.end_lineno or fn.lineno) - fn.lineno + 1
        nesting = _max_nesting(fn)
        if (cc, length, nesting) > (worst[1], worst[2], worst[3]):
            worst = (name, cc, length, nesting)
        if cc > _COMPLEXITY_CC_FAIL:
            evidence.append(f"第 {fn.lineno} 行 {name}(): 圈复杂度 {cc}（超过 {_COMPLEXITY_CC_FAIL}，必须拆分）")
        elif cc > _COMPLEXITY_CC_WARN:
            evidence.append(f"第 {fn.lineno} 行 {name}(): 圈复杂度 {cc}（超过 {_COMPLEXITY_CC_WARN}，建议拆分）")
        if length > _COMPLEXITY_LEN_WARN:
            evidence.append(f"第 {fn.lineno} 行 {name}(): 长度 {length} 行（超过 {_COMPLEXITY_LEN_WARN}）")
        if nesting >= _COMPLEXITY_NEST_WARN:
            evidence.append(f"第 {fn.lineno} 行 {name}(): 嵌套深度 {nesting}（达 {_COMPLEXITY_NEST_WARN} 层）")

    name, cc, length, nesting = worst
    summary = f"最高圈复杂度 {cc}({name}) / 最长函数 {length} 行 / 最深嵌套 {nesting}"
    has_fail = any("必须拆分" in e for e in evidence)
    if has_fail:
        return CheckResult(
            id="complexity", name="复杂度检查", category=CATEGORY_QUALITY,
            status=STATUS_FAILED, details=f"{summary}，存在严重超标函数", evidence=evidence[:8],
        )
    if evidence:
        return CheckResult(
            id="complexity", name="复杂度检查", category=CATEGORY_QUALITY,
            status=STATUS_WARNING, details=f"{summary}，部分函数超标", evidence=evidence[:8],
        )
    return CheckResult(
        id="complexity", name="复杂度检查", category=CATEGORY_QUALITY,
        status=STATUS_PASSED, details=summary,
    )


def _has_cache_decorator(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """判断函数是否被结果缓存装饰器包装（lru_cache 等）。"""
    for dec in fn.decorator_list:
        d = dec.func if isinstance(dec, ast.Call) else dec
        name = d.attr if isinstance(d, ast.Attribute) else (d.id if isinstance(d, ast.Name) else None)
        if name in _CACHE_DECORATORS:
            return True
    return False


def check_heavy_init(idx: _ModuleIndex) -> CheckResult:
    """重型资源重复初始化检查（性能）。

    main_process 可达函数体内每次调用都构造重型资源（mediapipe Pose、
    级联分类器、from_pretrained 等）→ 批量调用时性能极差。
    模块级一次性初始化或 lru_cache 包装不视为缺陷。
    """
    if "main_process" not in idx.functions:
        return CheckResult(
            id="heavy_reinit", name="重型资源重复初始化检查", category=CATEGORY_QUALITY,
            status=STATUS_SKIPPED, details="无 main_process 入口，跳过",
        )
    hits: list[str] = []
    for fname in idx.reachable_from_entry():
        fn = idx.functions[fname]
        if _has_cache_decorator(fn):
            continue
        for sub in ast.walk(fn):
            if not isinstance(sub, ast.Call):
                continue
            f = sub.func
            name = f.attr if isinstance(f, ast.Attribute) else (
                f.id if isinstance(f, ast.Name) else None)
            if name in _HEAVY_INIT_ATTRS:
                hits.append(f"第 {sub.lineno} 行 {fname}(): {name}(...) 在函数体内，每次调用都会重新初始化")
    if hits:
        return CheckResult(
            id="heavy_reinit", name="重型资源重复初始化检查", category=CATEGORY_QUALITY,
            status=STATUS_WARNING,
            details=f"发现 {len(hits)} 处函数体内重型资源构造（应提升到模块级或加缓存），批量调用性能极差",
            evidence=hits[:8],
        )
    return CheckResult(
        id="heavy_reinit", name="重型资源重复初始化检查", category=CATEGORY_QUALITY,
        status=STATUS_PASSED, details="无可达路径上的重复重型初始化",
    )


# ── 入口 ────────────────────────────────────────────────────────────────

def run_static_checks(
    source: str,
    code_dir: Path,
    *,
    algorithm_category: str = "",
    category_params: dict | None = None,
    constraints: list[str] | None = None,
) -> list[CheckResult]:
    """执行全部静态检查。语法错误时仅返回编译失败项。"""
    results: list[CheckResult] = [check_syntax(source)]
    if results[0].status == STATUS_FAILED:
        return results

    idx = _ModuleIndex(source)
    params = category_params or {}
    cons = {str(c).strip() for c in (constraints or params.get("constraints") or [])}

    # 反作弊
    results.append(check_random_core(idx))
    results.append(check_placeholders(source))
    results.append(check_stubs(idx))
    results.append(check_torch_random_only(idx))

    # 约束（仅检查用户勾选的）
    if "no_gpu" in cons:
        results.append(check_constraint_no_gpu(source))
    if "no_llm" in cons:
        results.append(check_constraint_no_llm(idx))
    if "no_training" in cons:
        results.append(check_constraint_no_training(idx))
    if "pretrained_only" in cons:
        results.append(check_constraint_no_training(idx, pretrained_only=True))
    if "rule_based" in cons:
        results.append(check_constraint_rule_based(source))
    if "single_file" in cons:
        results.append(check_constraint_single_file(idx, code_dir))

    # 契约与规范
    results.append(check_entry_contract(idx))
    labels = params.get("labels") or []
    if algorithm_category == "classification" or labels:
        results.append(check_label_contract(idx, [str(l) for l in labels]))
    results.append(check_unused_imports(source, idx))

    # 工程质量
    results.append(check_dependency_declaration(source, idx, code_dir))
    results.append(check_external_commands(idx))
    results.append(check_swallowed_exceptions(idx))

    # 复杂度与性能
    results.append(check_complexity(idx))
    results.append(check_heavy_init(idx))

    return results
