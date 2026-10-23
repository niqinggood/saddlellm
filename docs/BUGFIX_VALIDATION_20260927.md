# 使用流程排错与回归验收（2026-09-27）

本轮在补混合注意力/AttnRes 的同时，按 CLI、Python import、训练恢复、SFT→DPO、
保存重载、服务加载和生成流程检查使用问题。修复包括新增路径的缺陷和原有配置/分析路径的问题。
没有安装新依赖、下载预训练权重、重置用户修改或启动长时间训练。

## 已修复的问题

| 场景 | 修复前的可复现问题 | 修复后的行为 |
| --- | --- | --- |
| AMP + AttnRes | FP32 embedding 与 FP16/BF16 分支混用时，报 `depth sources must share ... dtype`，实际 Trainer 组合回归失败 | 允许浮点 dtype 混合，FP32 做评分/聚合，保留 embedding 残差流 dtype；增加 CPU autocast 与 GPU 低精度检查 |
| 打包文档 + MLA latent 缓存 | 第二段调用产生重置的 position_ids，被当任意显式位置拒绝；可变缓存可能已 append | 用完整文档 ID 重建全部 key 的 RoPE 位置；不支持的自定义缓存位置在混合模型开始更新任何层缓存前拒绝 |
| 纯 KDA + 通用预热 | 没有 RoPE 的模型调用 `prewarm_rope_cache(..., freeze=True)` 被误判为未预热 | 没有 rotary 的模型允许通用预热/冻结调用，保持无 RoPE 状态；普通 GQA 冻结边界检查不变 |
| CLI `train --dry-run` | 非法原生蓝图只展示阶段计划并返回成功，例如 Block 的奇数 block_size、KDA 配置 SDPA | 无需构建模型就校验蓝图；失败返回非零和 JSON 错误，不创建训练输出目录；正常蓝图仍只做规划 |
| Python/config 参数 | `head_dim=0` 被 `or` 表达式当作默认值；直接配置未知 attention_kind 会走普通注意力 | 显式非法 head_dim 和未知种类提前拒绝；KDA 的合法奇数维度不再被 RoPE 的偶数约束误拦 |
| 误用生成缓存做训练 | 把非空 KDA 推理缓存传给带梯度 forward、同时 `use_cache=False`，可能触发难理解的 autograd 错误 | 所有非空 KDA 缓存计算要求关闭梯度；训练用无缓存路径，独立递归函数仍支持梯度 |
| 模块调用与 hooks | AttnRes 模型直接调用 decoder 的子方法，forward hooks 完全不执行；单独调用该层又会无声执行普通残差 | 模型回到标准模块 `forward` 分派，hooks 按序执行；独立层缺少深度历史明确拒绝，不冒充已完成多 GPU 包装器验收 |
| 参数/缓存估算 | MLA 参数公式误用 query 头数估算 K/V 投影；KV 模式也按 latent 大小估算；KDA 被当每 token KV | MLA 按实际两个 KV 投影和所选缓存模式估算；KDA 单列固定递归状态；GQA/MLA 混合微型模型参数估计与真实参数数目对照 |
| 架构分析提示 | KDA/AttnRes 研究模型被报告成“低风险 Dense GQA baseline” | 明确提示研究参考路径、深度 activation 和独立消融的风险，不把渐近缓存收益描述成实际加速 |

另外补齐 code-first builder 的 `kda`、`attnres_full`、`attnres_block` 命名预设。
默认旧模型不开启 AttnRes，不新增其 state_dict 参数键；旧配置仍可重载。

主要代码：

- [AttentionResiduals.py](../saddlellm/models/AttentionResiduals.py)：混合 dtype 和稳定深度聚合。
- [KimiDeltaAttention.py](../saddlellm/models/KimiDeltaAttention.py)：状态、padding/文档边界和缓存校验。
- [SaddleModeling.py](../saddlellm/models/SaddleModeling.py)：深度流程、检查点、MLA 位置与 RoPE 生命周期。
- [ModelBlueprint.py](../saddlellm/models/ModelBlueprint.py)、[ModelMetrics.py](../saddlellm/models/ModelMetrics.py)：参数/缓存统计和风险提示。
- [cli.py](../saddlellm/cli.py)：无模型分配的原生蓝图预检查。

## 验收结果

最终同一次组合运行：**673 passed，1 deselected，14 warnings，64.92 秒**。
这是选定范围的回归，不是全仓库通过或模型能力评测。

包含 81 项新架构/使用专项测试，以及原有生成、流式服务、缓存、模型组合、quickstart、
配置、预训练、分布式运行时、数据检查和 tokenizer 等回归。
本机 CUDA 检查覆盖 FP16/BF16 KDA 和已有推理路径；不是跨所有设备/软件版本的兼容保证。

真实流程测试还包括：

- Dense、MLA/MoE/MTP、Hybrid+Block AttnRes 的 scratch Trainer，保存第一步、恢复到第二步并生成。
- Dense、Hybrid+Block、Hybrid+Full 的原生 SFT→DPO，保存最终模型并重载。
- 三种模型的实际推理加载入口及 LocalGenerationEngine completion；原有 SSE/取消/停止回归。
- 独立矩阵公式、double gradcheck、Full/Block 手工对照、检查点梯度一致性和请求状态隔离。

唯一主动排除的是 `test_pinned_runtime_completes_real_post_training_smoke`，
即独立的 HF/TRL 长链路真实后训练 smoke。本轮没有重跑它，因此不声称该路径已验收。
原生 SFT/DPO 已实际测试，不是把它们一并跳过。
Megatron/SGLang 的相关检查仍是配置/协议/替身检查，不是 Linux 多 GPU 真机验收。

警告来自已有依赖弃用提示、PEFT Conv1D 兼容提示和 Windows 文件时间戳回退，
本轮未通过升级依赖来隐藏警告。严格新文件 E/F/I lint、既有文件语法/致命检查、
compileall 和 `git diff --check` 通过。

## 复现本轮完整选定回归

```powershell
python -m pytest -q `
  tests/test_generation_sampling.py tests/test_generation_stop.py `
  tests/test_native_generation_session.py tests/test_native_streaming.py `
  tests/test_streaming_text.py tests/test_inference_batching.py `
  tests/test_generation_fastpaths.py tests/test_inference_cache.py `
  tests/test_native_model_lifecycle.py tests/test_release_pipeline.py `
  tests/test_model_evaluation.py tests/test_completion_evaluation.py `
  tests/test_model_network_optimizations.py tests/test_model_memory_fastpaths.py `
  tests/test_chunked_lm_loss.py tests/test_rope_cache_growth.py `
  tests/test_operator_backends.py tests/test_cli_output.py tests/test_webui.py `
  tests/test_public_api.py tests/test_packed_projection_checkpoint_edges.py `
  tests/test_runtime_backends.py tests/test_inference_optimizer.py `
  tests/test_serving_benchmark.py tests/test_gqa_decode_folding.py `
  tests/test_native_prefix_cache.py tests/test_deepseek_style_training.py `
  tests/test_model_composition.py tests/test_model_distributed_runtime.py `
  tests/test_scratch_pretrain_e2e.py tests/test_hybrid_attention_attnres.py `
  tests/test_hybrid_user_workflows.py tests/test_document_aware_packing.py `
  tests/test_quickstart.py tests/test_pretrain_quality_controls.py `
  tests/test_post_training.py tests/test_distributed_pretrain.py `
  tests/test_efficient_v2_preset.py tests/test_auto_parallel.py `
  tests/test_training_data_inspector.py tests/test_environment_diagnostics.py `
  tests/test_tokenizer_roundtrip.py `
  -k "not pinned_runtime_completes_real_post_training_smoke" --tb=short
```

也可只检查新增使用路径：

```powershell
python -m pytest -q tests/test_hybrid_attention_attnres.py tests/test_hybrid_user_workflows.py
saddlellm validate-config .\configs\hybrid_kda_attnres_scratch.yaml --skip-data-inspection
saddlellm train .\configs\hybrid_kda_attnres_scratch.yaml --dry-run
```

配置检查不会证明实际语料可用或质量合格；正式训练前需要再检查自己的数据。
完整使用及源码流程见 [混合注意力与 AttnRes 指南](HYBRID_ATTENTION_AND_ATTNRES.md)。
KDA 仍是顺序 PyTorch 研究参考，没有生产融合内核、百万 token 训练或吞吐优势的承诺。
