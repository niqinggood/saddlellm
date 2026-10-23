# 配置可选的 Megatron 训练与 SGLang 采样

## 先理解两个开关

Megatron 是训练执行后端，SGLang 是生成样本的执行后端，不是同一个开关的两个选项。
原来的 `model.backend: hf / saddle` 仍表示模型加载方式，不改成 megatron 或 sglang。

```yaml
backends:
  training: default  # default / megatron
  rollout: native    # native / sglang
```

不写这个配置块，旧行为保持不变。只有选中后端并实际执行时才加载外部运行时。
`training: default` 继续使用原有 HF / Saddle 训练；`rollout: native` 继续原有学生采样路径。
仅设置 SGLang 不会让离线预训练、SFT、DPO 自动启动采样服务。

当前接通的是 Megatron Bridge 的 pretrain / SFT、SGLang 采样及 MOPD 接入。
没有接通 Megatron DPO/KTO/PPO/GRPO，没有实现持续重叠的异步在线 RL。
默认 `serve` 仍是原生/HF 服务，此训练/rollout 配置不把它替换成 SGLang 网关。
2.37 另有独立的 [可选 vLLM 服务后端](VLLM_SERVING_BACKEND.md)：`serve --backend vllm`，
不混淆训练、rollout 和面向用户的推理服务三种执行后端。

## 最容易上手的命令

先复制示例，替换模型路径、数据路径和教师端点：

- [仅 SGLang 采样](../configs/sglang_mopd.yaml)。
- [仅 Megatron SFT](../configs/megatron_sft.yaml)。
- [SGLang 采样后接 Megatron SFT](../configs/megatron_sglang_mopd.yaml)。

```powershell
saddlellm validate-config configs/megatron_sglang_mopd.yaml --skip-data-inspection
saddlellm train configs/megatron_sglang_mopd.yaml --dry-run
```

`--dry-run` 不导入 Megatron/SGLang，不启动 GPU 进程、不连接教师、不更新服务器权重。
它能检查 SaddleLLM 配置、阶段组合、后端选择和启动参数，不能验证未安装的 Bridge
recipe 具体字段、模型格式、远端路径或 GPU 可用性。

环境与文件准备好后，实际运行还是同一个命令：

```powershell
saddlellm train configs/megatron_sglang_mopd.yaml
```

组合示例执行的是一轮：当前学生生成 → 教师改进 → 保存 MOPD JSONL → Megatron SFT
→ 可选导出 HF。它不是循环 RL。示例值是接入样板，不是已验证的收敛或性能配方。

## 环境要求：不影响普通用户

SGLang HTTP 客户端只需要小型可选依赖：

```powershell
pip install -e ".[rollout]"
```

HTTP 客户端可在 Windows 使用，GPU SGLang 服务运行在受支持的环境。
本地 SGLang Engine 与 Megatron GPU 训练要求受支持的 Linux CUDA 环境。
建议在官方 Megatron Bridge/NeMo GPU 环境安装 SaddleLLM 的基础包和 rollout extra；
不要直接把普通训练环境的固定 torch/transformers 版本与外部运行时依赖混装。
框架不会自动安装这些大型依赖、下载模型、创建容器或连接集群。

这里直接调用 NVIDIA 的 Megatron Bridge，而不是给普通 Trainer 加一个 Megatron 名称。
安装和 GPU/软件版本要求以[官方 Bridge 文档](https://docs.nvidia.com/nemo/megatron-bridge/latest/)为准。
接口依据为[官方训练入口](https://docs.nvidia.com/nemo/megatron-bridge/latest/training/entry-points.html)
及[recipe 用法](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/docs/recipe-usage.md)。

## SGLang：连接服务或者本地启动

### HTTP 模式

```yaml
backends:
  rollout: sglang
  sglang:
    mode: http
    base_url: http://127.0.0.1:30000
    policy_version: student-v1
    concurrency: 4
    max_pending_requests: 128
    timeout_seconds: 120
    refresh_weights: true
    return_logprob: false
    # 有鉴权时启用，不要直接把密钥放进 YAML：
    # api_key_env: SGLANG_API_KEY
```

连接的是 SGLang **native API 根地址**，不是 `/v1`。服务需要已经启动；框架不会
通过 SSH 帮你开启另一台机器。可在已安装 SGLang 的 Linux GPU 环境按其官方方式启动：

```bash
python -m sglang.launch_server --model-path /models/llama-3.2-1b --port 30000
```

采样使用 `/generate`，支持异步多请求、有界并发和按输入顺序返回。
超时/失败/取消后，对本批未完成的 request_id 最佳努力调用 `/abort_request`；
不发送 abort_all。网络断开不保证远端 GPU 立即停止，需要服务器监控确认。
超过单次 max_pending_requests 的 import 调用明确拒绝；MOPD 会分成有限批次。

MOPD 要求 policy_version。默认 refresh_weights=true：采样前通过
`/update_weights_from_disk` 更新完整模型，携带 weight_version，并等待成功。
model.name_or_path 或上游 HF 产物的路径必须在 **SGLang 服务器** 上可见，通常需要共享挂载。
不是把本机 Windows 路径转换成远端 Linux 路径，也不是上传 checkpoint 文件。

每个返回样本的 meta_info.weight_version 必须匹配预期，否则拒绝样本；不会靠客户端
自行填标签冒充服务器确认。上游 Megatron 产物带 run_id 时，版本自动采用
`配置版本:run_id`，防止不同训练产物复用同一标签。更新失败或取消后必须重新发布成功
才能采样。本适配器阻止自身批次与权重更新重叠，但不能管理同一服务器的其他客户端；
在线训练采样应使用自己管理的专用服务。

refresh_weights=false 表示你自行管理已部署权重；仍检查返回版本，不能仅凭版本名称
证明你把正确模型部署到了服务。HTTP 默认不重试采样，避免重复生成、重复奖励和混淆 request_id。

协议参照 [SGLang native API](https://github.com/sgl-project/sglang/blob/v0.5.5/docs/basic_usage/native_api.ipynb)
及[权重更新请求](https://github.com/sgl-project/sglang/blob/v0.5.5/python/sglang/srt/managers/io_struct.py)。
部署其他版本时应核对这些字段并进行真实联调，不能仅凭 OpenAI 兼容就认定可用。

### 本地 Engine 模式

```yaml
backends:
  rollout: sglang
  sglang:
    mode: engine
    model_path: /models/llama-3.2-1b
    policy_version: student-v1
    engine_kwargs:
      tp_size: 1
      mem_fraction_static: 0.6
```

首次采样才导入并创建 `sglang.Engine`。普通用户只运行训练配置，不必再管理 HTTP 服务。
engine_kwargs 原样传给本机安装版本的 Engine；版本不支持的参数会报错。
模型/版本由专用字段管理，不能在 engine_kwargs 中覆盖。

本地 Engine 把提示词批量交给 SGLang；异步 import 调用通过工作线程执行阻塞生成。
取消会等待运行中的本地批次结束再交还所有权，不承诺立即中断其 CUDA 执行。
更新策略时关闭旧 Engine，再以新模型/版本延迟创建，避免旧 KV 被复用。
MOPD 结束时会关闭自建 Engine，随后才能进入同机 Megatron 训练。
连接外部 HTTP 服务时不会关闭该服务器，显存分配和 GPU 隔离仍由部署者负责。

## Megatron：recipe 是实际训练配置

```yaml
backends:
  training: megatron
  megatron:
    recipe: megatron.bridge.recipes.llama:llama32_1b_sft_config
    nproc_per_node: 1
    hf_export_source: /models/llama-3.2-1b
    overrides:
      train:
        train_iters: 100
        global_batch_size: 2
        micro_batch_size: 1
      model:
        tensor_model_parallel_size: 1
        pipeline_model_parallel_size: 1
        context_parallel_size: 1
        seq_length: 1024
      dataset:
        seq_length: 1024
```

recipe 必须是安装版本中实际存在的 `megatron.bridge.recipes.模块:函数`。
预训练选择 pretrain recipe，SFT 选择 sft/peft recipe，不要混用。
多阶段可配置 recipes.pretrain / recipes.sft，公共 overrides 加阶段专用 stage_overrides：

```yaml
backends:
  training: megatron
  megatron:
    recipes:
      pretrain: megatron.bridge.recipes.llama:llama32_1b_pretrain_config
      sft: megatron.bridge.recipes.llama:llama32_1b_sft_config
    stage_overrides:
      pretrain:
        dataset:
          data_path: /data/preprocessed_text_document
```

公共 overrides 先应用，阶段 overrides 覆盖同名字段。未知字段在 worker 中报错，不会
创建一个没被使用的属性。显式 dtype 可写 torch.float32/float16/bfloat16，不执行 eval。
recipe 的默认并行数可能很大；必须与实际 nproc_per_node、TP/PP/CP、global/micro batch 匹配。
FP8、通信重叠、重计算、PEFT 等由所选 Bridge recipe 决定，不是 SaddleLLM 自动保证。

**原生 training.max_steps、training.learning_rate、sft.use_lora 等不会自动翻译为 Bridge 配置。**
在此后端使用 Bridge 的 train.train_iters、optimizer.lr、peft recipe 等字段。
SaddleLLM 仍消费 stages、日志输出根目录、dry_run/preflight_only、模型源、SFT 数据路径及运行恢复控制。
使用 training.dry_run=true 通过 import 运行时，会输出计划并明确标记 planned，不声称已经训练。

### 数据与 checkpoint

预训练不走原生 data.sources 清洗/packing；请按照 Bridge recipe 的 dataset 准备
indexed dataset（通常为 .bin/.idx 前缀），或显式选择其 mock 数据。训练的数据来源
必须由 recipe/overrides 指定，SaddleLLM 不会把原生语料配置悄悄当作 Bridge 数据。

SFT 可直接用 sft.data_path 的 JSONL：instruction/output（可带 input）或
prompt/completion。框架转成 Bridge 所需的 input/output 文件；messages 对话格式目前
没有自动转换，需自行配置合适的 Bridge dataset。不要同时提供 sft.data_path 和
Bridge 的 dataset_root/manifest 数据源覆盖。可单独提供 sft.validation_data_path。
未提供验证文件时是否评估由 recipe 控制，请检查其 validation 设置，不会用训练集伪造验证集。

从零预训练不允许初始化 pretrained_checkpoint；recipe 默认的 load 目录不会被自动恢复。
继续预训练/SFT 的 model.name_or_path 应是 Bridge 可读取的本地 HF/full-Megatron 模型源。
完整恢复使用 overrides.checkpoint.load 指向原生 checkpoint 根目录，不是 training.resume_from_checkpoint。
只有显式 checkpoint.load 才会从其训练状态恢复。

每阶段产物放在 logging.output_dir/megatron_pretrain 或 megatron_sft：

- job-*.json：所选 recipe、合并后的覆盖、模型源和产物路径。
- launch_plan.node-N.json：实际 torchrun 参数列表，不通过 shell 拼接执行。
- checkpoints/：Megatron 原生训练 checkpoint。
- hf_model/：配置 hf_export_source 时，通过 AutoBridge 显式导出的 HF 模型。
- job-*.result.json：worker 完成后写入的结果，失败不会伪造 completed。

Megatron checkpoint 不是 HF/SGLang 模型目录。后续采样、原生评估或部署需要
hf_export_source，它必须对应所训练架构，供 AutoBridge 导出配置/分词器/权重。
导出可能额外加载和转换模型，并消耗时间/显存；没有提供该参数时只返回 checkpoint_path，
不把它伪装成 model_path。产物和完成标识缺失时，即使 launcher 返回 0 也会判失败。
已有模型输出不自动覆盖，另选运行目录；原生 resume 也不能复用已完成的 run_id。

### 多 GPU / 多节点

使用 backends.megatron.nproc_per_node 控制每节点进程数，不再叠加 --parallel、
distributed.ddp/fsdp/zero。distributed 保持默认 single / 1 / 1，因为它描述的是外层控制器。
Megatron 自己拥有内部 torchrun，拒绝嵌套在另一个多进程 worker 中运行。

多节点需在各节点启动同一份配置，设置各自的 node_rank，并配置相同的 nnodes、master_addr、
master_port 和新的共享 run_id；模型、数据和输出路径须为相同绝对路径的共享文件系统。
控制器的 pipeline_state 按节点分开，worker 完成结果共享。当前多节点控制器仅支持
pretrain/SFT；采样/评估/部署另开单控制器执行。不自动申请机器、SSH 或调度 Slurm。
本版受控多 rank 数据/导出屏障不支持 Bridge inprocess_restart；不是集群容错完成版。

## Python import 与异步采样

```python
from saddlellm import create_rollout_backend

config = {"backends": {
    "rollout": "sglang",
    "sglang": {"mode": "http", "base_url": "http://127.0.0.1:30000",
               "policy_version": "student-v1"},
}}
with create_rollout_backend(config) as backend:
    samples = backend.generate(["解释注意力机制", "介绍 KV Cache"],
                               {"max_new_tokens": 64, "temperature": 0})
    print([sample.text for sample in samples])
```

上述 import 例子检查服务器当前版本，不自动发布权重。需要时先调用
backend.prepare_policy("/服务器可见的/HF目录", "新版本")。
在事件循环中分别使用 await backend.aprepare_policy(...) 和 await backend.agenerate(...)，
不要在 async 函数里直接调用同步 HTTP 方法。一个实例只拥有一批/一次更新，忙时明确拒绝
再入和 close；可按业务分开实例，但不要绕过同模型的发布屏障。

native import 可用 create_rollout_backend({}, model=model, tokenizer=tokenizer)，
支持 generate/agenerate，返回生成部分而不是拼接 prompt 的文本，且恢复原来的 train/eval 状态。
它不是新的连续批处理器，不保证跨后端、精度、硬件的输出 token 完全一致。

RolloutSample 提供 request_id、text、后端确认的 policy_version、可用的 output_ids、
finish_reason；return_logprob=true 时要求返回逐 token 分数且与 output_ids 对齐。
logprob_semantics 标记为 backend_reported，不擅自认为是原始策略或 top-p 后分布。
MOPD JSONL 的 student_rollout 保存这些信息；它仍不是完整 RL 轨迹，没有 prompt token IDs、
工具 mask、奖励协议或现成的重要性校正损失。

## 验证与当前边界

```powershell
python -m pytest -q tests/test_runtime_backends.py
```

测试覆盖配置拒绝、延迟导入、HTTP 协议/并发/顺序/错误/取消/版本、Engine 生命周期、
Megatron worker 入口/覆盖/启动计划/完成产物检查、超时清理及 MOPD 分流。
HTTP 使用模拟服务器响应，Engine/Bridge 使用测试替身；它们证明适配逻辑，**不是 GPU 后端验收**。
本轮未安装或启动真实 Megatron/SGLang GPU 运行时，没有测 GPU 速度、显存和训练收敛。
上线前应在目标环境做小模型真实 SFT/导出/采样联调，并检查 tokenizer、token 分数、
停止规则、权重版本和失败恢复。后续异步在线训练边界见 [后端分离设计](ROLLOUT_BACKEND_DESIGN.md)。
