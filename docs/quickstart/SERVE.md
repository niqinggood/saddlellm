# 路线一：部署聊天服务

前提：已按 [安装指南](../GETTING_STARTED.md) 安装 `[serve]`，并准备完整的本地聊天模型。
下面把模型目录写作 `./models/Qwen3-0.6B`，请替换成实际位置；需要 config、tokenizer 和全部权重分片，不是只有一个权重文件。

## 先检查，再启动

```text
saddlellm doctor --profile serve
saddlellm serve ./models/Qwen3-0.6B --backend local --local-files-only --dry-run
saddlellm serve ./models/Qwen3-0.6B --backend local --local-files-only --port 8000 --no-chat-history --open
```

`dry-run` 不加载模型，不保证容量足够。最后一条会占用终端，初始化后提供网页和 API；按 Ctrl+C 停止。
打开 `http://127.0.0.1:8000/` 聊天；`/readyz` 返回就绪状态，`/health` 显示实际模型和后端。示例不自动关闭模型思考、不自动量化。
CPU 可显式增加 `--device cpu --dtype float32`；大模型在 CPU 上可能很慢。

`--no-chat-history` 关闭内置历史的所有读取和写入，不删除旧数据库。当前对话只留在页面内存，刷新即清空。

省略该参数则历史保存在启动目录的 `.saddlellm/chat_history.sqlite3`，这是同一模型的**共享工作区**，不是多人私密聊天系统。

## 选择性能预设

省略 `--performance` 保留现有自动部署策略。对原生后端已经支持的模型，可以用一个预设配置完整重复提示词缓存或流式合批：

```text
saddlellm serve ./models/MODEL --backend native --dtype bfloat16 --performance latency --dry-run
saddlellm serve ./models/MODEL --backend native --dtype bfloat16 --performance throughput --open
```

原生路线的 `latency` 配置 64 MiB 前缀快照，只匹配至少 256 token 的相同完整提示词，适用于重试、重新生成等重复请求，主要减少重复 prefill。普通续聊会改变输入，预设不启用部分前缀匹配，不能据此承诺续聊提速；首次请求、独特或短提示、后续 decode 也不保证更快。缓存命中仍复制请求私有的 KV，回答会重新生成。

`throughput` 为未指定的参数补齐四请求原生流式合批、四并发和对应 token 预算，保持 `compatible` 数值模式。上例显式使用 BF16；实际合批支持 FP32 或 CUDA BF16。框架按加载后的模型检查资格，不支持时整组撤销预设添加项，并在 `/health` 的 `backend_selection.performance` 中说明原因。预设不自动改变精度、量化、思考或输入长度，配置预算也不等于 GPU 显存准入保证。

显式参数优先，例如 `--prefix-cache-mb 0` 禁用预设缓存，`--max-concurrency 1` 保留单请求策略。`--dry-run` 显示计划，加载后状态以 `/health` 为准。

Linux CUDA 的 vLLM 路线也接受这两个预设，启用未被显式配置覆盖的前缀缓存，并保留已有调度参数；使用 `--no-vllm-prefix-caching` 可明确关闭。普通 HF/adapter 模型不会因预设被自动转换为原生模型。

Python 使用同一入口：`serve(path, backend="native", dtype="bfloat16", performance="throughput")`。Web 和 OpenAI 客户端调用不变。性能实测和适用范围见 [本轮推理优化记录](../INFERENCE_SPEED_20261002.md)。

## 用 Python 调用

在客户端环境安装 `python -m pip install "openai>=1.109,<3"`，保存为 `client.py`，然后 `python client.py`：

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="local-unused")
model = client.models.list().data[0].id
reply = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": "用三句话解释 KV cache。"}],
    max_tokens=128,
)
print(reply.choices[0].message.content)
print(reply.usage)
```

上面的占位 key 只适用于未启用鉴权的 loopback 示例。真实服务启用 `--api-key-env` 后，客户端必须使用对应密钥。
OpenAI 兼容范围为文本 chat/completion 和 SSE；不要期待工具调用、图片或 Responses API。

## 用 Python 启动同一个服务

```python
from saddlellm import serve

if __name__ == "__main__":
    serve("./models/Qwen3-0.6B", backend="local", local_files_only=True,
          port=8000, chat_history=False)
```

这是启动服务，不是客户端调用；不要同时在同一个端口启动两次。
原生后端可以显式选择 `--backend native`，但只限 [已经适配的配置](../NATIVE_MULTI_MODEL.md)。Linux CUDA 上的 vLLM 安装和多卡边界见 [服务手册](../OPENAI_SERVING.md)。
向其他机器开放前，应配置鉴权与 HTTPS 反向代理；多人场景先关闭内置共享历史。
