"""Generate a source-linked callable reference for SaddleLLM.

The report is deliberately derived from the checked-in first-party source.  It
does not import SaddleLLM (and therefore does not need GPU/model dependencies),
so it is safe to run as part of documentation checks.
"""
from __future__ import annotations

import argparse
import ast
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


ROOT = Path(__file__).resolve().parents[1]
PYTHON_ROOTS = (ROOT / "saddlellm", ROOT / "saddle_llm")
FRONTEND_ROOT = ROOT / "spatial-studio" / "src"
DEFAULT_OUTPUT = ROOT / "docs" / "FUNCTION_REFERENCE.md"


@dataclass(frozen=True)
class CallableRecord:
    path: Path
    line: int
    qualified_name: str
    signature: str
    kind: str
    summary: str
    calls: tuple[str, ...]


def _one_line(value: str, maximum: int = 240) -> str:
    value = " ".join(value.strip().split())
    if len(value) <= maximum:
        return value
    return value[: maximum - 1].rstrip() + "…"


def _first_doc_line(node: ast.AST) -> str:
    doc = ast.get_docstring(node, clean=True) or ""
    if not doc:
        return ""
    paragraph = doc.split("\n\n", 1)[0]
    return _one_line(paragraph)


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _call_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def _direct_calls(node: ast.AST) -> tuple[str, ...]:
    class CallVisitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.names: list[str] = []

        def visit_Call(self, item: ast.Call) -> None:  # noqa: N802
            name = _call_name(item.func)
            if name and name not in self.names:
                self.names.append(name)
            self.generic_visit(item)

        # A nested definition has its own row and is not a call made by the
        # enclosing function. Defaults/decorators are evaluated by Python, but
        # reporting them as body calls is more confusing than useful here.
        def visit_FunctionDef(self, item: ast.FunctionDef) -> None:  # noqa: N802
            return

        def visit_AsyncFunctionDef(self, item: ast.AsyncFunctionDef) -> None:  # noqa: N802
            return

        def visit_ClassDef(self, item: ast.ClassDef) -> None:  # noqa: N802
            return

    visitor = CallVisitor()
    body = getattr(node, "body", ())
    for statement in body:
        visitor.visit(statement)
    return tuple(visitor.names[:10])


def _all_visible_calls(node: ast.AST) -> tuple[str, ...]:
    """Legacy broad scanner retained for debugging the generator itself."""
    calls: list[str] = []
    for item in ast.walk(node):
        if isinstance(item, ast.Call):
            name = _call_name(item.func)
            if name and name not in calls:
                calls.append(name)
    return tuple(calls[:10])


def _arguments(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    rendered = ast.unparse(node.args)
    prefix = "async " if isinstance(node, ast.AsyncFunctionDef) else ""
    returns = f" -> {ast.unparse(node.returns)}" if node.returns is not None else ""
    return f"{prefix}{node.name}({rendered}){returns}"


_ACTION_LABELS = {
    "add": "添加",
    "analyze": "分析",
    "append": "追加",
    "apply": "应用",
    "authorize": "鉴权",
    "balance": "平衡",
    "build": "构建",
    "check": "检查",
    "clean": "清洗",
    "close": "关闭",
    "collect": "收集",
    "compare": "比较",
    "compile": "编译",
    "compute": "计算",
    "configure": "配置",
    "convert": "转换",
    "create": "创建",
    "decode": "解码",
    "deduplicate": "去重",
    "delete": "删除",
    "describe": "描述",
    "detect": "检测",
    "download": "下载",
    "encode": "编码",
    "ensure": "确保",
    "estimate": "估算",
    "evaluate": "评估",
    "export": "导出",
    "extract": "提取",
    "filter": "过滤",
    "find": "查找",
    "fit": "拟合",
    "format": "格式化",
    "generate": "生成",
    "get": "读取",
    "handle": "处理",
    "infer": "推断",
    "inspect": "检查",
    "install": "安装",
    "list": "列出",
    "load": "加载",
    "log": "记录",
    "merge": "合并",
    "normalize": "规范化",
    "parse": "解析",
    "plan": "规划",
    "predict": "预测",
    "prepare": "准备",
    "process": "处理",
    "read": "读取",
    "record": "记录",
    "register": "注册",
    "render": "渲染",
    "report": "报告",
    "resolve": "解析",
    "restore": "恢复",
    "run": "执行",
    "sample": "采样",
    "save": "保存",
    "score": "评分",
    "serve": "启动服务",
    "set": "设置",
    "simulate": "模拟",
    "split": "切分",
    "summarize": "汇总",
    "tokenize": "分词",
    "train": "训练",
    "update": "更新",
    "validate": "校验",
    "verify": "验证",
    "write": "写入",
}

_OBJECT_LABELS = {
    "action": "动作",
    "adapter": "适配器",
    "artifact": "产物",
    "attention": "注意力",
    "backend": "后端",
    "batch": "批次",
    "blueprint": "模型蓝图",
    "cache": "缓存",
    "checkpoint": "检查点",
    "config": "配置",
    "data": "数据",
    "dataset": "数据集",
    "device": "设备",
    "distributed": "分布式运行时",
    "environment": "环境",
    "feedback": "反馈",
    "flow": "流程",
    "grid": "占用栅格",
    "image": "图像",
    "input": "输入",
    "manifest": "数据清单",
    "metrics": "指标",
    "model": "模型",
    "observation": "观测",
    "operator": "算子",
    "output": "输出",
    "path": "路径",
    "plan": "计划",
    "prompt": "提示词",
    "recipe": "训练配方",
    "release": "发布包",
    "report": "报告",
    "request": "请求",
    "response": "响应",
    "route": "路线",
    "runtime": "运行时",
    "source": "数据源",
    "stage": "训练阶段",
    "state": "状态",
    "tokenizer": "分词器",
    "training": "训练",
    "world": "世界",
}


def _name_words(name: str) -> list[str]:
    name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name).strip("_").lower()
    return [part for part in name.split("_") if part]


def _inferred_summary(name: str, owner: str | None, kind: str) -> str:
    if name == "__init__":
        return f"初始化 `{owner}` 实例及其运行依赖。" if owner else "初始化对象。"
    if name == "__post_init__":
        return f"在 `{owner}` 创建后校验并规范化字段。" if owner else "在数据对象创建后校验并规范化字段。"
    if name == "__getattr__":
        return "按需解析未直接绑定的属性，主要用于延迟导入或兼容转发。"
    if name == "__dir__":
        return "返回该模块或兼容命名空间可发现的公开名称。"
    if name == "forward":
        return f"执行 `{owner}` 的前向计算。" if owner else "执行前向计算。"
    if name in {"to_dict", "as_dict"}:
        return f"把 `{owner}` 转为可序列化字典。" if owner else "把当前对象转为可序列化字典。"
    if name == "from_dict":
        return f"从字典解析并创建 `{owner}`。" if owner else "从字典解析并创建对象。"
    if name in {"save_pretrained", "from_pretrained", "from_pretrained_saddle"}:
        verb = "保存" if name == "save_pretrained" else "从检查点加载"
        return f"{verb} `{owner or '模型'}`，遵循预训练模型的目录契约。"
    if name.startswith("_cmd_"):
        return f"实现 CLI `{name[5:].replace('_', '-')}` 子命令，并返回进程退出码。"

    words = _name_words(name)
    action = next((_ACTION_LABELS[word] for word in words if word in _ACTION_LABELS), "实现")
    objects = [_OBJECT_LABELS[word] for word in words if word in _OBJECT_LABELS]
    subject = "、".join(dict.fromkeys(objects)) or f"`{name.lstrip('_')}`"
    scope = f"`{owner}` 中" if owner else "模块级"
    private = "内部辅助逻辑" if name.startswith("_") else "公开操作"
    if kind == "nested function":
        private = "局部回调/辅助逻辑"
    return f"{scope}{action}{subject}的{private}。"


class _PythonVisitor(ast.NodeVisitor):
    def __init__(self, path: Path) -> None:
        self.path = path
        self.scope: list[tuple[str, str]] = []
        self.records: list[CallableRecord] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:  # noqa: N802
        self.scope.append((node.name, "class"))
        self.generic_visit(node)
        self.scope.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:  # noqa: N802
        self._visit_function(node)

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        owner = next((name for name, kind in reversed(self.scope) if kind == "class"), None)
        parent_function = any(kind == "function" for _, kind in self.scope)
        kind = "nested function" if parent_function else "method" if owner else "function"
        qualified = ".".join([name for name, _ in self.scope] + [node.name])
        summary = _first_doc_line(node) or _inferred_summary(node.name, owner, kind)
        self.records.append(
            CallableRecord(
                path=self.path,
                line=node.lineno,
                qualified_name=qualified,
                signature=_arguments(node),
                kind=kind,
                summary=summary,
                calls=_direct_calls(node),
            )
        )
        self.scope.append((node.name, "function"))
        self.generic_visit(node)
        self.scope.pop()


def _python_records() -> list[CallableRecord]:
    records: list[CallableRecord] = []
    for source_root in PYTHON_ROOTS:
        for path in sorted(source_root.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
            visitor = _PythonVisitor(path)
            visitor.visit(tree)
            records.extend(visitor.records)
    return records


_TS_FUNCTION = re.compile(
    r"^(?P<indent>[ \t]*)(?:export\s+)?(?:async\s+)?function\s+"
    r"(?P<name>[A-Za-z_$][\w$]*)\s*",
    re.MULTILINE,
)
_TS_ARROW = re.compile(
    r"^(?P<indent>[ \t]*)(?:export\s+)?const\s+(?P<name>[A-Za-z_$][\w$]*)\s*=\s*"
    r"(?:async\s*)?(?P<args>\([^\n=]*\)|[A-Za-z_$][\w$]*)\s*=>",
    re.MULTILINE,
)
_TS_CALLBACK = re.compile(
    r"^(?P<indent>[ \t]*)(?:export\s+)?const\s+(?P<name>[A-Za-z_$][\w$]*)\s*=\s*"
    r"(?P<wrapper>useCallback|useMemo)\s*\(\s*(?:async\s*)?",
    re.MULTILINE,
)
_TS_CONSTRUCTOR = re.compile(
    r"^(?P<indent>[ \t]+)constructor\s*",
    re.MULTILINE,
)


def _balanced_parentheses(text: str, start: int) -> tuple[str, int]:
    """Return the parenthesized parameter text beginning at/after ``start``."""
    opening = text.find("(", start)
    if opening < 0:
        return "()", start
    depth = 0
    quote = ""
    escaped = False
    for index in range(opening, len(text)):
        char = text[index]
        if quote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = ""
            continue
        if char in {'"', "'", "`"}:
            quote = char
        elif char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return text[opening : index + 1], index + 1
    return "()", start


def _frontend_records() -> list[CallableRecord]:
    records: list[CallableRecord] = []
    if not FRONTEND_ROOT.is_dir():
        return records
    for path in sorted(FRONTEND_ROOT.rglob("*")):
        if path.suffix not in {".ts", ".tsx"} or ".test." in path.name or path.parent.name == "test":
            continue
        text = path.read_text(encoding="utf-8")
        matches: list[tuple[int, re.Match[str], str]] = []
        matches.extend((item.start(), item, "function") for item in _TS_FUNCTION.finditer(text))
        matches.extend((item.start(), item, "arrow") for item in _TS_ARROW.finditer(text))
        matches.extend((item.start(), item, "callback") for item in _TS_CALLBACK.finditer(text))
        matches.extend((item.start(), item, "constructor") for item in _TS_CONSTRUCTOR.finditer(text))
        seen: set[tuple[int, str]] = set()
        for _, match, pattern_kind in sorted(matches, key=lambda item: item[0]):
            name = "constructor" if pattern_kind == "constructor" else match.group("name")
            line = text.count("\n", 0, match.start()) + 1
            identity = (line, name)
            if identity in seen:
                continue
            seen.add(identity)
            indent = len(match.group("indent").replace("\t", "  "))
            kind = "constructor" if pattern_kind == "constructor" else "local function" if indent else "function/component"
            if pattern_kind == "arrow":
                args = match.group("args")
            else:
                args, _ = _balanced_parentheses(text, match.end())
            signature = f"{name}{_one_line(args, 160)}"
            records.append(
                CallableRecord(
                    path=path,
                    line=line,
                    qualified_name=name,
                    signature=signature,
                    kind=kind,
                    summary=(
                        f"初始化 `{path.stem}` 中定义的 TypeScript 类实例。"
                        if pattern_kind == "constructor"
                        else _inferred_summary(name, path.stem if indent else None, "nested function" if indent else kind)
                    ),
                    calls=(),
                )
            )
    return records


def _relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def _escape(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def _module_heading(path: Path) -> str:
    return _relative(path)


def render(records: Sequence[CallableRecord]) -> str:
    python_count = sum(record.path.suffix == ".py" for record in records)
    frontend_count = len(records) - python_count
    modules = sorted({record.path for record in records}, key=_relative)
    lines = [
        "# SaddleLLM 函数与方法参考",
        "",
        "> 本文档由 `python tools/generate_function_reference.py` 从一方源码静态生成。",
        "> “直接调用”来自 AST 静态扫描，只表示源码中可见的调用，不等同于完整运行时调用图；反射、回调和第三方框架注入不会被完全捕获。",
        "",
        "## 覆盖范围",
        "",
        f"- Python：`saddlellm/` 与兼容包 `saddle_llm/`，共 **{python_count}** 个模块函数、方法和命名的嵌套函数。",
        f"- 前端：`spatial-studio/src/`（排除测试），共 **{frontend_count}** 个命名函数、组件、Hook 回调或构造器。",
        f"- 涉及 **{len(modules)}** 个含可调用项的源码文件；排除测试、第三方研究仓库、构建产物、依赖目录和匿名内联回调。",
        "- 职责说明优先采用源码 docstring；没有 docstring 时，根据函数名、所属类和函数类型生成明确的用途说明。",
        "",
        "## 阅读方式",
        "",
        "- 链接会跳到对应源码行。",
        "- `function` 是模块级函数，`method` 是类方法，`nested function` 是函数内部具名回调/辅助函数。",
        "- 调用链和模块边界请先阅读 [ARCHITECTURE.md](ARCHITECTURE.md)。",
        "",
    ]
    by_path: dict[Path, list[CallableRecord]] = {}
    for record in records:
        by_path.setdefault(record.path, []).append(record)
    for path in modules:
        path_records = sorted(by_path[path], key=lambda item: (item.line, item.qualified_name))
        lines.extend(
            [
                f"## `{_module_heading(path)}`",
                "",
                f"共 {len(path_records)} 个具名可调用项。",
                "",
                "| 函数 / 方法 | 类型 | 意义 | 静态可见的直接调用（最多 10 个） |",
                "|---|---|---|---|",
            ]
        )
        relative_link = "../" + _relative(path)
        for record in path_records:
            label = f"[`{record.qualified_name}`]({relative_link}#L{record.line})"
            signature = _escape(record.signature)
            summary = _escape(record.summary)
            calls = ", ".join(f"`{_escape(call)}`" for call in record.calls) or "—"
            lines.append(
                f"| {label}<br><sub>`{signature}`</sub> | {record.kind} | {summary} | {calls} |"
            )
        lines.append("")
    lines.extend(
        [
            "## 生成与校验",
            "",
            "```powershell",
            "python tools/generate_function_reference.py",
            "python tools/generate_function_reference.py --check",
            "```",
            "",
            "`--check` 不改文件；当生成结果与已提交文档不一致时返回非零退出码。",
            "",
        ]
    )
    return "\n".join(lines)


def collect() -> list[CallableRecord]:
    return _python_records() + _frontend_records()


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    output = args.output.resolve()
    rendered = render(collect())
    if args.check:
        if not output.is_file() or output.read_text(encoding="utf-8") != rendered:
            print(f"Function reference is stale: {output}")
            return 1
        print(f"Function reference is current: {output}")
        return 0
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8", newline="\n")
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
