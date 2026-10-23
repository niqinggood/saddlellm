# 第一次使用 SaddleLLM

SaddleLLM 用统一命令和 Python 接口组织大模型训练、评测与推理部署。先选一个任务，不需要安装所有后端。
当前为 2.40 本地候选包，本文不表示该版本已发布到 PyPI。模型权重不包含在 wheel 中。

## 1. 安装：选一个独立环境

建议使用 Python 3.11。Windows PowerShell：

```powershell
python -m venv .venv-saddle
.venv-saddle\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

若系统不允许激活脚本，可把下文的 `python` 替换为 `.venv-saddle\Scripts\python.exe`，把 `saddlellm` 替换为 `.venv-saddle\Scripts\saddlellm.exe`。

Linux/macOS：

```bash
python3 -m venv .venv-saddle
source .venv-saddle/bin/activate
python -m pip install --upgrade pip
```

把收到的 wheel 放在当前目录，然后选择安装用途：

```text
# 普通推理或 Saddle 原生推理
python -m pip install "./saddlellm-2.40-py3-none-any.whl[serve]"

# 训练：另建训练环境，不要把全部后端叠加到同一环境
python -m pip install "./saddlellm-2.40-py3-none-any.whl[posttrain]"

# 高吞吐服务：另建 Linux CUDA 环境
python -m pip install "./saddlellm-2.40-py3-none-any.whl[serve-vllm]"
```

CPU 验收环境可在安装 `[serve]` 前运行 `python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu`，并使用随源码提供的 `deploy/constraints-serve.txt` 约束安装。
CUDA 用户应先安装与驱动兼容的 CUDA PyTorch；不要在 GPU 环境照抄 CPU 安装命令。需要离线安装时，必须同时准备匹配平台的依赖 wheel 和完整模型目录；只有 Saddle wheel 并不足够。

## 2. 按用途自检

```text
saddlellm doctor --profile serve
saddlellm doctor --profile native --device cpu
saddlellm doctor --profile train
saddlellm doctor --profile vllm --device cuda
python -m pip check
```

只执行自己需要的一项 profile。默认 `doctor` 等同于 `--profile train`。
`issues` 是依赖/设备问题，导致退出码 1；`warnings` 是资源风险。CPU 不会仅因为没有 CUDA 被判不可用。
`gpus` 列出各 CUDA 设备的驱动可用显存，查询失败显示 `null`，不是按当前 Python 进程占用推算。
`ready=true` 不保证任意模型都能放下，也不代表完整 vLLM 引擎已成功启动。

## 3. 选择一条上手路线

1. [部署聊天与 OpenAI 接口](quickstart/SERVE.md)：已有模型 → 命令启动 → 网页聊天 → Python 调用。
2. [第一次 LoRA 微调](quickstart/TRAIN.md)：样例数据 → 检查 → 5 步训练 → 找到 adapter。
3. [评测、导出与重新部署](quickstart/EVALUATE_DEPLOY.md)：对照验证 → 了解发布门禁 → 导出 → 部署。

先阅读 [模型与平台支持范围](SUPPORTED_ENVIRONMENTS.md)。完整命令见 [命令手册](CLI_COMMAND_GUIDE.md)，故障处理见 [排错指南](TROUBLESHOOTING.md)。
