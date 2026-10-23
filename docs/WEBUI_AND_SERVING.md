# SaddleLLM 训练看板与推理 WebUI

## 1. 最简使用方式

SaddleLLM 保留训练、查看、服务三个明确动作，不要求用户理解或维护独立前端工程：

```powershell
# 1. 训练
saddlellm train config.yaml

# 2. 查看训练过程和结果；webui 是 dashboard 的等价别名
saddlellm dashboard outputs/my-run --open

# 3. 启动网页对话和 OpenAI 兼容 API
saddlellm serve outputs/release --open
```

`dashboard` 的参数既可以是某一次运行目录，也可以是包含多次运行的父目录。传入父目录时自动选择最近更新的运行。服务都默认只监听 `127.0.0.1`，按 `Ctrl+C` 正常停止。

修改源码后若在源码目录外仍看不到新命令，说明 `saddlellm.exe` 加载的是旧安装包。重新安装当前源码并再次检查：

```powershell
python -m pip install --no-deps --force-reinstall E:\DEV\saddle-ml-llm\build-saddlellm
saddlellm dashboard --help
```

## 2. 训练可视化

Trainer 路径统一写出两个轻量文件：

- `training_progress.json`：状态、step、max steps、epoch、最近指标和更新时间；
- `training_metrics.jsonl`：每次日志事件一行，便于实时刷新和其他程序读取。

看板同时兼容历史训练结果。没有上述实时文件时，会自动读取递归目录中的 `trainer_state.json`、`pipeline_state.json`、`training_summary.json` 和 `training.log`。页面显示：

- 训练进度、训练 loss、验证 loss；
- 对数尺度 loss 曲线；
- 当前阶段和各阶段状态；
- 发布门禁；
- checkpoint、评估与报告产物；
- 最近训练日志。

看板每 2 秒读取一次本地文件，不要求训练进程向另一个监控服务推送数据，不上传模型、数据或指标。TensorBoard、WandB 等仍可通过 `logging.backend` 选择，但不再是基本使用的前置条件。

## 3. 推理网页与 API

`serve` 只加载一份模型。浏览器页面和 API 共用同一个所选引擎，所以不会因为打开 WebUI 再占一份显存。
2.40 默认 auto：兼容 Linux CUDA vLLM 环境走异步连续批处理，否则明确选择 local。
完整参数与官方 OpenAI SDK 调用见 [一条命令部署手册](OPENAI_SERVING.md)。

加载原生模型时默认合并 Q/K/V 和 SwiGLU gate/up 投影，旧模型不需要重新训练。
想禁用该布局优化做对照，添加 `--no-fuse-projections`；HF 模型不套用这条合并逻辑。
`/health.native_projection_packing` 显示模型加载时的合并报告，未执行时为 null。
Python 已加载模型的用户可在模型独占、eval 模式下调用 `optimize_native_inference(model)`；
通用训练加载器不自动执行该优化。详见 [原生推理优化](NATIVE_INFERENCE_ENGINE.md)。

部署和评估职责分离：`serve` 只负责加载用户指定的完整模型、checkpoint 或 adapter，不重新执行评估，也不阻止调试模型启动。质量评估由 `evaluate` 和 `compare` 完成；需要在 CI/发布流水线中阻止未通过模型时，使用可选的 `export --require-gate`。如果服务目录带有 release manifest，`/health` 和页面会展示其中的门禁状态，供调用方判断。旧名称 `serve-model`、`evaluate-model`、`compare-evaluations` 和 `export-model` 仍可使用。

默认入口：

- `/`：内置网页对话；
- `/saddle`：同一问答 Web 中的 SaddleLLM 功能介绍；
- `/api/saddle/catalog`：静态功能说明与命令示例，不执行训练或模型推理；
- `/health`：健康、模型、发布门禁和流式能力；
- `/livez`、`/readyz`：进程存活/服务就绪，不运行 GPU forward；
- `/v1/models`：OpenAI 风格模型发现；
- `/v1/chat/completions`：对话，支持 `stream=true`；
- `/v1/completions`：文本补全，也支持 `stream=true`。

两个流式入口均可设置 `stream_options: {"include_usage": true}`，结束前返回真实 token 用量。
CPU/CUDA/MPS 的平台边界、Docker、健康检查和真实 HTTP 压测见
[跨平台部署指南](CROSS_PLATFORM_SERVING.md)。MPS 路由/三平台 CI 已加入，macOS/Linux 真机验收尚未完成。

输出预算默认是 256 个新 token、最大 512 个。2.40 保留模型模板原本的思考默认值；
长推理可能用完预算而未完成答案，需按业务增加预算。`--disable-thinking` 是显式的延迟/行为权衡，
`--enable-thinking` 明确启用受支持的模板开关。`--no-web-ui` 可用于纯 API 部署。

浏览器默认使用 `temperature=0.2`、256 个新 token。回答按常用 Markdown 显示标题、列表、引用、链接、代码块和表格；代码块可单独复制，宽表格及长代码可横向滚动。数学公式支持 `$...$`、`$$...$$`、`\(...\)` 和 `\[...\]`，代码中的美元符号不会当作公式。Markdown 经允许标签过滤，公式以不信任输入模式渲染；图片不自动加载。医疗、护理等高风险结论必须继续经过专业人员复核，网页对话不能替代发布门禁。

网页对话完成或手动停止后，会把用户/助手消息写到**执行 `saddlellm serve` 命令时的工作目录**下的 `.saddlellm/chat_history.sqlite3`。例如在 `D:\models\demo` 中启动，历史文件就在 `D:\models\demo\.saddlellm\chat_history.sqlite3`。当前标签页会按模型记住成功保存或主动打开过的会话编号，刷新后自动恢复该会话。旧版未记住编号的记录可先在左侧“历史记录”点击一次；系统不会擅自选择最新的一条记录。“新建对话”会清除当前标签页的会话指针，保留服务器上的旧记录；列表中的删除按钮会永久删除对应会话。记录按模型名称分开显示；同一工作目录下运行的多个服务可共用文件。未启用 API Key 时，历史接口仅允许本机回环地址访问；远程使用应设置 `--api-key-env`。目前这是同一服务密钥下的共享历史，不是多用户隔离系统，请勿把未鉴权服务直接暴露到公网。

浏览器 `sessionStorage` 只保存当前标签页的会话 UUID，不保存消息内容或 API Key。历史关闭、存储受限、未完成鉴权时可继续使用普通问答；加载历史失败会保留原指针。保存失败会显示实际 HTTP 状态与简短原因。重新启动服务时应保持工作目录和模型名称一致，才能读到相同的历史文件。

流式片段按浏览器动画帧合并刷新，避免每收到一个片段就重复解析整段 Markdown；这降低的是前端重排开销，不改变模型生成的 token/s。

## 4. 设计边界

local/native 路线使用模型的生成实现，适合单机功能验证、演示和相应并发场景；
高并发优先兼容的 vLLM 路线，由异步引擎管理连续批处理与 KV 调度。
原生模型已提供 KV 缓存复用、最后位置词表投影和生成缓冲区复用；添加
`--batch-size 4` 还可合并兼容的非流式、temperature=0 请求。采样、流式与
Hugging Face/custom 模型继续逐请求执行。批处理队列有上限，满载返回可重试的 503。
具体使用、测量方法和限制见 [原生推理优化](NATIVE_INFERENCE_ENGINE.md)。
进一步可选长度分桶、padded token 批预算和队列背压，已有窗口层自动回收旧 KV；
用法与独立测量见 [窗口 KV 与调度](INFERENCE_WINDOW_AND_SCHEDULING.md)。

`--prefix-cache-mb 64` 可开启原生 dense 模型的单请求 prompt 快照缓存，默认只复用完全相同
的 token 提示词；`--prefix-cache-partial` 另外开启部分前缀复用。原生非流式和流式共用，
窗口模型、固定合批、HF/custom、MLA/MoE 不使用该路径。`/health.prefix_cache` 报告实际命中与保留字节。
缓存默认关闭，短请求不保证更快；MiB 预算不包括权重和活跃请求，不能当作总显存限制。

普通原生模型的流式请求已使用可暂停的逐 token 会话，不额外创建每请求生成线程；
HTTP 模型步骤仍在线程池中执行。Hugging Face/custom 和重写 generate 的模型保留原路径。
`/health.streaming_backend` 显示实际执行方式。客户端断连后会等待正在进行的计算结束、
关闭会话缓存，再释放模型调用许可；这仍不是流式连续批处理。

`--max-concurrency` 控制同时执行的模型调用数，`--batch-size` 控制每批请求数，
二者都不是吞吐能力或显存安全的证明。默认不开启合批，也不提高并行度。
本阶段是固定批次，不是连续批处理；100 并发生产验收仍需连续批处理后端、
限流和真实压测。SaddleLLM 保持同一套 OpenAI 兼容接口，便于后续替换执行后端。

不在核心包中硬编码护理字段、学院名称或比赛规则。领域差异由训练数据、系统提示词、评估插件和发布门禁表达；WebUI 只消费通用运行产物。

## 5. 调研依据

- [LLaMA-Factory WebUI](https://llamafactory.readthedocs.io/en/latest/getting_started/webui.html) 将训练、评估、对话和导出放在统一界面，说明端到端入口有实际使用价值。
- [LLaMA-Factory 监控文档](https://llamafactory.readthedocs.io/en/latest/advanced/monitor.html) 同时保留内置 loss 看板与 TensorBoard/WandB 等外部监控。SaddleLLM 采用相同分层，但基础看板只读本地产物。
- [Hugging Face Trainer callbacks](https://huggingface.co/docs/transformers/main_classes/callback) 提供只读训练事件回调；本地进度文件通过 callback 生成，不改训练计算。
- [Gradio ChatInterface](https://gradio.app/docs/gradio/chatinterface) 支持生成器流式更新。SaddleLLM 没有继续引入 Gradio 依赖，而是在现有 FastAPI 服务上提供同样的流式交互。
- [vLLM OpenAI-compatible server](https://docs.vllm.ai/en/latest/serving/openai_compatible_server/) 说明生产推理后端可保持 OpenAI 风格协议。SaddleLLM 内置页面因此只依赖协议，不绑定本地生成实现。

## 6. 验收检查

```powershell
python -m compileall -q saddlellm tests
python -m pytest -q
```

页面验收至少覆盖：桌面与手机宽度、训练完成/运行/失败状态、无指标空状态、流式对话、API 鉴权、关闭思考、Ctrl+C 正常停止，以及浏览器控制台无应用错误。

## 7. 公共政策实验工作台

第一阶段提供已有研究的只读浏览入口。命令从当前源码目录运行时，也可以使用 `python -m saddlellm.cli`，避免加载旧安装版本：

```powershell
python -m saddlellm.cli workbench --project "E:\DEV\saddle-ml-llm\public_policy_rlvr" --study qwen3_06b_direct_seed42 --open
```

默认地址为 `http://127.0.0.1:7861`；`--study` 可省略，在页面中选择研究。按 `Ctrl+C` 停止网页服务。

工作台使用已有的公共政策研究产物，提供六个阅读入口：

1. 数据：分别解释 A 的立场分类和 B 的多主题分类，查看原始记录、所选训练/评测案例及保存的实际模型输入。
2. 检查：查看历史 doctor 的实际结果、检查范围、问题记录和对应命令。
3. 实验：按明确研究与运行编号列出基础模型和各方法，显示配置、数据版本和不同评测目标。
4. 训练：读取所选运行的 SFT/DPO 日志和 GRPO 指标，缺少记录时显示缺失，不将不同方法的 loss 作为统一排名。
5. 效果：按相同案例比较 Base 和所选方法的第一个采样回答，筛选改对、改错、一直正确或一直错误；A 的 accuracy 和 B 的 exact-match@1 分开展示。
6. 报告：下载已存在的入门报告、比较结果、原始证据和复现工具，复制保存的命令。

页面筛选保存在 URL，刷新与复制链接可恢复视图；图表同时提供数值与表格，手机端以纵向内容和可横向滚动的宽表呈现。历史检查带有来源，不能当作当前机器的实时诊断。原始官方数据划分、联合派生划分和本轮抽取样本数分别显示。

本阶段不提供训练启动、停止或恢复的写接口，不加载模型，不执行研究目录中的代码，也不修改原始数据和历史输出。新训练仍从项目的 `saddlellm experiment --project ...` 入口执行。面向其他研究的数据适配和网页任务调度将沿用同一证据结构逐步接入。

## 8. 在原问答 Web 中查看 SaddleLLM 介绍

启动已有的 `saddlellm serve <模型目录> --open` 后，点击侧栏的“SaddleLLM 介绍”，或打开同一地址的 `/saddle`。此页面属于原推理问答服务，与当前模型共用入口；local/native 和 vLLM 路线均使用同一页面安装逻辑。

介绍以用途组织：先了解数据与环境检查，再查看训练微调、偏好优化、决策模型、评估、蒸馏、轻量化和服务部署。每项包含使用场景、数据示例、操作步骤和可复制的命令；命令中模型、数据与输出路径为需要自行替换的示例。

网页只展示这些功能的使用说明，不把复制命令当成已经执行。SFT/DPO 等直接命令与项目自定义 GRPO/RLVR 入口分别说明，LoRA/QLoRA 与训练目标分别说明。

“模型问答”与“介绍”在同一页面中切换，保留已显示消息、未发送草稿和当前设置；浏览器前进/后退也可切换。直接打开 `/saddle` 时默认展示介绍。`--no-chat-history` 仅关闭历史保存，介绍页仍可使用；`--no-web-ui` 同时关闭问答页和介绍页。

更新源码后需要重启原来的推理服务，使其加载新的页面与路由。

介绍页支持中文与 English 切换。英文可直接访问 `/saddle?lang=en`；`/api/saddle/catalog?lang=zh|en` 返回对应说明。两种语言使用相同的功能编号、命令和教学数据格式，切换语言不重建聊天区域。

预训练章节分别解释从零初始化、加载已有权重继续预训练、从训练 checkpoint 恢复。`configs/tinystories_tiny_scratch.yaml` 与 `configs/tinystories_zh_16k_continue.yaml` 使用不同的词表与模型，不能作为同一模型前后相接的两条命令；每个示例明确列出语料、分词器和基础权重的准备条件。

各方法附有可展开的疑难问题说明，按“症状 → 可能原因 → 检查 → 修复 → 验证”组织。SFT、DPO、GRPO/RLVR 的数据监督、截断、显存、梯度与效果退化分别说明；诊断建议是排查路径，不表示当前用户已发生该问题。

正文采用深色与较大字号，命令示例可换行；页面上下留白缩小，功能目录与主侧栏、正文之间单独保留间距。手机端目录改为上方筛选，避免挤压正文。

## 9. 训练评测与模型导出的一致性

`evaluate` 与训练流水线的任务评测现在共用 `run_evaluation`。JSONL 和平坦 JSON 数组仍可直接评测；如果一个 JSON 同时包含 train/validation/test 等多份数据，应明确选择：

```powershell
saddlellm evaluate outputs/checkpoint data/evaluation.json --task completion --split validation --output outputs/evaluation-validation.json
```

对应流水线配置示例：

```yaml
eval:
  enabled: true
  task: completion
  dataset: data/evaluation.json
  split: validation
  prompt_field: instruction
  completion_field: output
  device: auto
```

旧 `eval.tasks` 配置继续可用，默认集合仍是 validation。指定集合不存在会明确报告错误，不读取另一个集合代替。对于单文件，split 是调用者声明的用途；框架不会凭文件名证明它是独立留出数据。

普通导出无需发布报告或新审批：

```powershell
saddlellm export outputs/checkpoint outputs/release
```

只有主动使用已有的 `--require-gate` 时，才要求通过报告包含当前模型的内容身份：

```powershell
saddlellm export outputs/checkpoint outputs/release --gate-result outputs/release_gate.json --require-gate
```

`evaluate` 生成本地模型、adapter/基础模型及 tokenizer 的内容身份，`compare` 透传对应身份；同路径替换权重或拿其他模型的报告会被严格检查识别。旧报告缺指纹、远程模型内容未解析时，可以普通导出；严格导出需要对本地实际产物重新 evaluate/compare。此检查只核对报告和模型对应关系，不增加用户权限或审批流程。

原生 Saddle 检查点保持原生格式：

```powershell
saddlellm export outputs/native-checkpoint outputs/native-release --format saddle --no-merge-lora
```

`native` 是 `saddle` 的格式别名。导出清单分别记录被评测源产物和实际发布产物的身份；合并或序列化造成内容变化时保留转换说明，源模型报告不等于对所有转换后效果都做了重新评测。
