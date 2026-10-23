# 混合注意力与 Attention Residuals 使用和源码说明

实现日期：2026-09-27。这里是 Saddle 原生模型的新架构研究路径：可以从零训练、反向传播、
保存重载和带缓存生成。KDA 是顺序 PyTorch 参考实现，不是优化后的 Triton/CUDA 内核；
不声称复现 Kimi 权重、K3 完整架构、百万 token 能力或任何生产吞吐数字。

## 1. 先记住两个独立问题

- **混合注意力**解决“当前 token 怎样获取历史 token 信息”：部分层用 KDA 递归矩阵，
  部分层用全局 GQA/MLA，二者实际执行不同计算。
- **Attention Residuals（AttnRes）**解决“当前子层怎样选择之前子层的信息”：
  用可学习的深度加权替代固定残差求和。softmax 沿深度源轴，不沿 token 轴。

可以只用混合注意力而保留普通残差，也可以给原有纯 GQA/MLA 模型启用 AttnRes。
它们都改变新模型的结构/计算，不是给现有训练好的模型无损打开的推理加速开关。

## 2. 最简单的 Python 用法

```python
import torch
from saddlellm import ModelBlueprintLab

blueprint = ModelBlueprintLab.hybrid_kda(
    name="saddle-my-hybrid",
    hidden_size=64,
    layers=4,
    heads=4,
    kv_heads=2,
    vocab_size=128,
    seq_length=128,
    full_attention_kind="gqa",    # 或 "mla"，MLA 默认使用 latent 缓存
    linear_layers_per_full=3,      # KDA, KDA, KDA, 全局注意力
    attention_residuals="block", # "none" / "full" / "block"
    residual_block_size=4,        # 注意力+FFN 共 4 个子层，即 2 个 decoder 层
)
model = blueprint.build_model()  # 创建随机权重，不下载任何模型
ids = torch.randint(4, 128, (2, 12))
model.train()
loss = model(input_ids=ids, labels=ids).loss
loss.backward()

model.eval()
with torch.inference_mode():
    output = model.generate(ids[:1, :4], max_new_tokens=4,
                            do_sample=False, eos_token_id=[])
print(output.shape)
```

这个例子只演示流程。随机权重和一次 backward 不能生成有意义的语言。
预设每完成一个周期放一层全局注意力，至少要求一个完整周期；末尾不完整周期仍为 KDA 层。
例如 5 层、3:1 得到 `KDA,KDA,KDA,GQA,KDA`；想要完整等比例请让层数为 4 的倍数。
更复杂的位置/比例通过 `layers` 蓝图显式指定。

## 3. 命令行从零训练

配置文件：[hybrid_kda_attnres_scratch.yaml](../configs/hybrid_kda_attnres_scratch.yaml)。
准备自己有权使用的 UTF-8 `data/train.jsonl`，每行一条文本，例如：

```json
{"text": "这是第一条训练文本，需要用真实且足够丰富的语料替换。"}
{"text": "这是第二条训练文本，示例两行不适合评估模型能力。"}
```

在项目根目录运行：

```powershell
saddlellm validate-config .\configs\hybrid_kda_attnres_scratch.yaml --skip-data-inspection
saddlellm train .\configs\hybrid_kda_attnres_scratch.yaml --dry-run
saddlellm train .\configs\hybrid_kda_attnres_scratch.yaml
```

如果 CLI 没加入 PATH，可把 `saddlellm` 换成 `python -m saddlellm.cli`。
`--dry-run` 会校验原生蓝图并展示阶段计划，不更新权重；它不能证明数据质量、实际训练收敛或 GPU 吞吐。
正式命令先训练本地 tokenizer，再从随机权重进行 20 步流程试跑，最后保存到
`outputs/hybrid_kda_attnres/run/final_model`。此配置没有验证集，不作为正式质量训练配方。
正式实验需要独立 validation/test、充分的数据预算和完整的评测；请保留 Dense 控制组。

改变模型结构只需修改 `model.blueprint`，不要改训练循环。可把同样的字典交给
`TrainingOrchestrator.from_dict(config).run()`。原生路径使用 `model.backend: saddle`；
不能把该蓝图当普通 HF Llama 配置或 Megatron recipe 自动转换。

## 4. 蓝图怎样配置

最关键的配置片段如下，其余完整字段见上面的示例文件：

```yaml
model:
  backend: saddle
  blueprint:
    # 还需要 name、hidden_size、vocab_size、max_position_embeddings、ffn 等
    attention: {kind: gqa, backend: sdpa, num_heads: 4, num_kv_heads: 2}
    residual: {topology: serial, attention_mode: block, block_size: 4}
    layers:
      - repeat: 3
        attention: {kind: kda, backend: eager, kda_conv_size: 4, kda_gate_rank: 0}
      - repeat: 1
        attention: {kind: gqa, backend: sdpa}
```

| 字段 | 含义/效果 |
| --- | --- |
| `attention.kind: kda` | 用短卷积、通道衰减和 delta 更新维护矩阵状态，不存全部历史 KV |
| KDA 的 `backend: eager/auto` | 都执行当前 PyTorch 参考路径；不会暗中调用 SDPA 当作 KDA |
| `kda_conv_size` | 每个 Q/K/V 分支的因果深度卷积窗口，正整数；默认 4 |
| `kda_gate_rank` | 衰减门和输出门的低秩瓶颈；0 使用 head_dim，正数显式指定 |
| 全局层 `kind: gqa/mla` | 保留精确的全局 token 注意力；通常使用 SDPA |
| `residual.attention_mode: none` | 原有普通残差行为，不创建 AttnRes 参数；默认值 |
| `...: full` | 每个注意力/FFN 子层选择全部先前子层输出和 embedding |
| `...: block` | 选择 embedding、完成的块以及当前块内部分和，减少深度源数量 |
| `residual.block_size` | 以注意力/FFN 子层计数；Block 模式为至少 2 的偶数，默认 4 |

AttnRes 模式和 block_size 是模型全局拓扑，所有层必须一致；可分别设置原有分支 scale。
目前只支持 serial，不支持 parallel + AttnRes。KDA 不支持 `fused_qkv`、可学习的 `qk_norm`
或 `sliding_window`；这些错误配置会拒绝，不会无声忽略。KDA 自己使用 Q/K 的 L2 归一化，
它与原有 `qk_norm` 选项不是一回事。KDA 层不使用 RoPE，其顺序来自递归和因果短卷积。

如果只给纯 GQA 模型加 AttnRes，可以把原蓝图的 `residual.attention_mode` 改为 `full/block`，
但必须构建和训练新模型。用 `SaddleModelConfig` 直接构建时，对应字段是
`residual_attention_mode`、`residual_block_size`；蓝图则使用上述嵌套字段。

## 5. Attention Residuals 的实现逻辑

入口：[AttentionResiduals.py](../saddlellm/models/AttentionResiduals.py) 的
`SaddleAttentionResidual`。设此前深度源为 `v_i`：

```text
key_i = RMSNorm(v_i)
score_i = dot(learned_query, key_i)
weight_i = softmax(score)_i           # 只沿源/深度轴
input = sum_i(weight_i * v_i)        # value 是原始源，不是归一化的 key
```

每个注意力子层、FFN 子层和最终读出各有独立 query 与 key 归一化参数。
query 初始为零，因此初始权重均匀，但输出是平均值，不是传统残差的求和。
FP16/BF16 的评分、softmax 和求和在 FP32 中计算，输出使用第一个源的残差流 dtype；double 保留 FP64。
支持 AMP 中 FP32 embedding 与 FP16/BF16 分支同时作为源，不会因浮点 dtype 不同而误拒绝。
每个 token 单独做深度选择，不会因深度混合而看到未来 token。

**Full：**历史包含 embedding，以及每个已完成注意力/FFN 分支的输出。
先混合并做输入 norm，再调用当前注意力；加入注意力输出后，为 FFN 再混合一次。
FFN 输出也加入历史。最后用独立的读出 mixer 聚合所有源，再走最终 norm 和 LM head。

**Block：**完成的块仅保留其各子层输出的和，embedding 独立保留。
块内仍累计原始分支输出；每次计算时混合完成的块与当前部分和。达到块边界后把部分和
作为完成块加入历史并清空部分和。最后不足一个块的部分和也参与最终读出。
这些深度源只存在于当前 forward，不保存在模型对象或跨 token 请求缓存中。

模型连接点：[SaddleModeling.py](../saddlellm/models/SaddleModeling.py) 的
`SaddleDecoderLayer.forward_attention_residuals` 和 `SaddleForCausalLM.forward`。
模型通过 decoder 层的标准模块调用进入 `forward`，再分派到深度混合，保留 forward hooks。
单独调用启用 AttnRes 的 decoder 层时必须提供 `depth_sources`；缺少历史会明确报错，
不会无声执行普通残差。这里验证了标准 hooks，不代表 FSDP/TP/PP 的多 GPU 实测已完成。
梯度检查点把当次所需深度源作为显式张量参数传入，避免 backward 重算时误读后续层的历史。
MoE 重算保留原来的 router 状态保护，分块 LM loss 和 MTP 仍从最终混合表示计算。

## 6. KDA 参考路径的实现逻辑

入口：[KimiDeltaAttention.py](../saddlellm/models/KimiDeltaAttention.py)。
Q/K/V 投影先经过各自的因果短卷积和 SiLU，Q/K 再做 L2 归一化。
低秩门生成每个 key 通道的衰减 `alpha`，beta 门控制 delta 更新强度。
设状态 `S` 的最后两维是 `[D_key,D_value]`：

```text
decayed = diag(alpha_t) * S_previous
prediction = k_t^T * decayed
error = beta_t * (v_t - prediction)
S_t = decayed + k_t * error^T
o_t = q_t^T * S_t
```

`kda_recurrence` 逐 token 执行该更新，FP16/BF16 状态累积在 FP32 中。
输出经过每个 head 的 RMSNorm、低秩 sigmoid 输出门和输出投影。
这里先保证公式、梯度和分段状态传递正确，没有实现生产级 chunkwise/DPLR 融合路径。
顺序 Python 循环在长序列训练/prefill 上可能明显慢于成熟 SDPA，不能只因复杂度线性就声称更快。

## 7. 混合缓存与生成

`NativeGenerationSession` 按层分配缓存：

- KDA：请求私有的 `RecurrentAttentionCache`，保存矩阵、三份卷积历史及可选文档边界状态。
- GQA：原有 `InferenceLayerCache`，保存历史 KV。
- MLA latent：原有压缩潜变量缓存，生成时重建 K/V。

`cache_implementation="geometric"` 使用这些请求私有容器；`"dynamic"` 使用
KDA 字典状态和原有 KV/latent 动态结构；`use_cache=False` 重新计算完整历史。
两条带缓存路径都覆盖整段/分块/token 计算一致性。缓存执行要求关闭梯度；普通训练不启用缓存。
缓存包含 input-slot 长度，padding 也计入长度以与其他层对齐，但 padding 不更新 KDA 状态。
不同请求互不共享递归状态，session 关闭后释放引用；复用缓存时 batch、dtype、device 必须不变。

KDA 支持二维 0/1 padding mask。`document_ids` 表示各样本中连续的独立文档段，每段使用唯一 ID；
有效 token 跨文档时，同时清空矩阵和短卷积历史。缓存调用须持续提供覆盖过去与当前的
完整 `document_ids`/padding mask，不能中途切换分段模式。任意四维 attention mask 不支持。
全局 MLA latent 的打包缓存会从完整文档 ID 重建重置后的 RoPE 位置；任意自定义的缓存位置仍不支持。
这个能力不意味着随意改写过去文档 ID/遮罩能修复已计算的状态；过去输入和分段必须保持不变。

当前原生前缀快照的安全白名单不包含 KDA/AttnRes，因此这些模型会跳过前缀复用。
没有把递归状态误当历史 KV，也没有偷偷共享可修改的请求状态。

## 8. 显存收益如何理解

单层 KDA 的状态估算，batch=1：

```text
matrix_bytes = heads * head_dim * head_dim * 4
conv_bytes = 3 * heads * head_dim * (conv_size - 1) * activation_dtype_bytes
optional_document_bytes = 9
```

double 调试的矩阵使用 8 字节。KDA 部分不随上下文增长；全局层的 KV/latent 部分仍随上下文增长。
`blueprint.estimate_kv_cache_bytes_per_token()` 统计增长部分，
`blueprint.estimate_recurrent_cache_bytes()` 统计固定部分；`cost_profile()` 和
`ModelMetricsExtractor.cache_profile()` 同时报告两类。
这些是逻辑张量估计，包括可选文档状态的保守预算，不含 allocator、容量预留、mask、
临时 K/V 重建或训练 activations；不能当成峰值显存承诺。

Full AttnRes 要保留更多深度源；Block 减少源数量，但当前实现仍是逐层 stack + softmax，
没有论文中的两阶段推理融合、online-softmax 合并或跨流水线通信缓存。

## 9. 保存、兼容性与上线边界

`save_pretrained()` 保存完整分层配置和新参数；`load_causal_lm(path)` 自动走原生重载。
保存的是模型权重，不是请求缓存。Trainer 的恢复检查点另外保存 optimizer 等训练状态。
默认 `attention_mode: none` 不新增 AttnRes state_dict 键；旧配置缺少新字段仍使用旧行为。
新架构不能严格加载一个旧纯 GQA 检查点后就获得训练好的 KDA/AttnRes 能力。

蓝图到 legacy `ModelSpec` 或普通 HF dense config 的转换会显式拒绝 KDA/AttnRes，避免丢掉结构。
本次没有开发 Megatron 原生混合层、SGLang/vLLM 权重适配、TP/PP 专用通信或量化融合算子。
支持原生小模型研究不等于这些生产后端已支持该新架构，也不等于现有量化方案可直接套用。

## 10. 验收与推荐实验顺序

```powershell
python -m pytest -q tests/test_hybrid_attention_attnres.py tests/test_document_aware_packing.py tests/test_scratch_pretrain_e2e.py
```

测试包括独立矩阵公式对照、double gradcheck、Full/Block 手工深度流程对照、块边界与末尾部分块、
padding/文档隔离、动态/可变缓存、请求隔离、缓存/无缓存生成、梯度检查点和 MoE/MTP、
分块词表 loss、FP16/BF16 CUDA 检查、AdamW 参数更新及保存重载。
scratch E2E 还实际启动 Trainer，保存后恢复到第二步，并对最终模型执行生成。
这些只证明微型流程和数值正确性，不证明长期收敛、质量提升或集群可扩展性。

建议按相同语料/预算保留四个独立实验：Dense+普通残差、Hybrid+普通残差、Dense+Block AttnRes、
Hybrid+Block AttnRes。固定 tokenizer 和评测集，记录 PPL/任务分数、实际训练 token、
峰值显存、prefill/decode 延迟和总 GPU 时间；根据实测再决定优化融合内核或扩大规模。

## 11. 原始资料

- [Kimi Linear 原始论文](https://arxiv.org/abs/2510.26692)：KDA 更新、门控、短卷积和混合架构。
- [Attention Residuals 原始论文](https://arxiv.org/abs/2603.15031)：深度 softmax、Full/Block 和最终聚合。
- [Moonshot 官方 AttnRes 仓库](https://github.com/MoonshotAI/Attention-Residuals)：数学定义和参考伪代码。

本实现按公开数学定义独立编写，不复制发布权重。原论文的实验结果不是 Saddle 的结果。
尤其原 Kimi Linear 的全局 MLA 使用 NoPE，而本预设全局 GQA/MLA 保留现有 RoPE。
