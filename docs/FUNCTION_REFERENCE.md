# SaddleLLM 函数与方法参考

完整索引从源码按需生成，包含函数签名、职责、源码行号与静态可见的直接调用。
仓库仅保留生成脚本，避免重复提交约 1 MB 的派生内容。

在仓库根目录运行：

```powershell
python tools/generate_function_reference.py
```

打开本地 `outputs/FUNCTION_REFERENCE.md` 阅读完整报告。`outputs/` 已被 Git 忽略。
源码变更后重新生成；如需检查已有本地报告是否同步：

```powershell
python tools/generate_function_reference.py --check
```

生成过程仅使用 Python 标准库，不加载模型或训练依赖。
模块关系和调用流程见 [架构说明](ARCHITECTURE.md)。
