# 多模型原生推理：范围、用法与实测边界

`saddlellm serve MODEL --backend native` 现在通过架构注册表读取兼容的 Qwen3、Qwen2 和 Llama 稠密权重。
仍是 Saddle 自己的模型前向、KV、采样与生成循环；tokenizer 和底层 PyTorch 算子不等于 HF `generate()`。
不支持的配置提前报错，绝不悄悄退回 HF。默认 auto 后端选择策略没有改变。

## 可以直接怎样使用

```powershell
saddlellm serve /models/DeepSeek-R1-Distill-Qwen-1.5B --backend native --device cuda --dtype bfloat16 --local-files-only --model-name deepseek-distill
```

默认地址 `http://127.0.0.1:8000/` 提供对话页面；文本接口为 `/v1/chat/completions`、`/v1/completions`，支持 SSE。
也可以把本地目录换成模型仓库 ID，允许下载时去掉 `--local-files-only`。先核对内存/显存，勿在共享机器上直接下载或加载巨大模型。
Python 入口仍是 `load_native_causal_lm(path, device="cuda", dtype="bfloat16")`，或 `serve(path, backend="native")`。
源码里的功能尚未因此自动发布到 PyPI；部署需使用本轮源码或重新构建的包。

## 国内模型不能只按品牌判断

| 模型/架构 | 本轮原生支持与验证 |
| --- | --- |
| Qwen3 稠密、全注意力、默认 RoPE | 已有原生后端；真实 0.6B CUDA BF16 A/B |
| Qwen2/Qwen2.5 类标准稠密结构 | 新适配 Q/K/V bias 与无 QK Norm；同权重 fixture 检查 |
| DeepSeek-R1-Distill-Qwen-1.5B | 实验性：真实权重原生加载、CUDA BF16 生成/API/A/B；双方 eager 的 16 步 logits 相同，但 SDPA HTTP 与 HF 回答分叉。不是完整版 R1 |
| Llama 标准稠密、默认未缩放 RoPE | 新适配；同权重小 fixture、CPU 与 CUDA 低精度检查；没有完整大权重测速 |
| DeepSeek-V3 / 完整 R1 | 不是 Qwen2 适配器；MLA、MoE、FP8 等完整加载语义尚未实现 |
| Kimi-K2 | 官方配置是 `kimi_k2`/DeepseekV3ForCausalLM，另有 MoE、MLA、YaRN、FP8；本轮没有声称原生支持 |
| MiniMax-M2 | 官方配置是 `minimax_m2`，包含 MoE、部分旋转维度、QK Norm 等专有配置；本轮没有原生支持 |

完整版模型可评估对应兼容版本的 HF/vLLM 等成熟后端，但“框架提供这个后端入口”不等于当前机器/依赖版本已经能加载这些权重。
本机 Transformers 4.57.6 的内置配置映射没有 `kimi_k2` 和 `minimax_m2`；不自动执行远程模型代码，不自动升级依赖。
依据：[DeepSeek-R1 配置](https://huggingface.co/deepseek-ai/DeepSeek-R1/blob/main/config.json)、[Kimi-K2 配置](https://huggingface.co/moonshotai/Kimi-K2-Instruct/blob/main/config.json)、[MiniMax-M2 配置](https://huggingface.co/MiniMaxAI/MiniMax-M2/blob/main/config.json)。

所有新稠密适配器目前都拒绝 RoPE scaling、滑窗/混合层、量化权重、多模态和不兼容投影。
因此不能笼统声称支持所有 Qwen2.5、Llama 或 DeepSeek 蒸馏版本。原生 INT8 仍只支持既有 Qwen3 路线。
Qwen2 的带 bias QKV 不会被错误地合并成无 bias 投影；仍可使用 gate/up 合并。
低精度跨引擎比较不保证逐 token 相同，需要另做质量检查。

后续修正：CUDA BF16 的 Qwen2/Llama 在本机这类没有 FlashAttention 的构建中，缺省选择 `sdpa_reference`，
保留 compact GQA 的参考调度而非强制高效内核/解码折叠。已消除所测 DeepSeek 1.5B 的分叉；该路径可能更慢，不是新加速内核。
显式 `--native-attention-backend sdpa` 仍可选择速度优先路径，但需自行验证质量。
`/health.native_attention_backend` 显示真实选择。新验证与流式并发改进见 [后续迭代记录](NATIVE_SERVING_ITERATION_20260928.md)。

## 源码如何串起来

`serve --backend native` → `InferenceServer.load_inference_model`
→ `ModelLoader.load_native_causal_lm` → `NativeArchitectures.native_architecture_config`
→ `NativeQwen3._load_native_checkpoint` → `SaddleForCausalLM`
→ `NativeGenerationSession` → OpenAI 文本/SSE 响应。

- 注册表按 `config.json.model_type` 和 architectures 验证计算语义，不按模型名字猜测。
- Qwen3 启用 QK Norm；Qwen2 启用 QKV bias；Llama 两者都不启用。
- 共用严格 safetensors key/shape/shard 检查、逐 tensor 加载、冻结权重、revision pin 与显存预检。
- `qkv_bias` 写入 Saddle 配置，保存后重载保持相同结构；旧配置缺省为 false。
- 不复制三套生成循环，因此已验证的 KV、stop、SSE、usage 与解码优化可以复用。

## 速度优化必须通过验证

本轮增加 CUDA 单 token 解码 Q/K RoPE 合并候选，不合并 GEMM、不降低精度、不改变思考设置。
长 prefill、CPU 和梯度路径仍执行参考实现。初测有收益，但独立复测的配对稳定性没有通过门槛，因此默认关闭，服务不会自动启用。
开发者可在独占模型初始化阶段调用 `optimize_native_inference(..., pack_decode_rotary=True)` 做显式实验；报告的 `decode_rotary_packed` 显示配置层数。
它不是 CUDA Graph、分页 KV 或连续批处理，也不是新的 C++/CUDA 融合算子。

另一项“单请求结束状态复用”候选已经实测，但没有稳定收益，因此默认禁用；保留私有 A/B 路径供实验，不提供用户部署开关。
性能数字、验收与后续迭代见 [验证方案](INFERENCE_VALIDATION_PLAN.md) 和 [本轮实验记录](NATIVE_MULTI_MODEL_RESULTS_20260928.md)。
