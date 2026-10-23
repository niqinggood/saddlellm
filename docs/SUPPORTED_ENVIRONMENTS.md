# 公开试用支持范围

本文区分“代码路径存在”“本地功能已测”和“目标环境实际验收”。CI 配置存在不表示对应远端作业已经运行通过。
测试状态以具体报告为准，不能从公司名称推断所有模型兼容。

| 路线 | 试用范围 | 当前边界 |
| --- | --- | --- |
| local / HF | Transformers 能加载且具有匹配 tokenizer 的文本因果模型 | 逐模型验证；量化依赖、远程代码和模板另有要求 |
| native 浮点 | 标准 Qwen3、受限 Qwen2/Llama 稠密配置 | 单设备；不支持任意 RoPE/MoE/混合架构；不兼容时明确失败 |
| native W8A16 | 已记录的 Qwen3 CUDA/Triton 路线 | 实验性；不把量化结果当成无损 BF16 性能 |
| vLLM | 独立 Linux CUDA 环境、当前声明的 vLLM 版本范围 | 已有适配与单机记录；多 GPU 本轮未实机验收 |
| Windows | 本地 CPU/CUDA 服务 | 不宣称 Windows 原生 vLLM/DeepSpeed 生产支持 |
| macOS | 已有设备路由与 CPU CI 配置 | MPS 多数是替身/路由测试，不冒充 Apple GPU 真机基准 |

具体已有证据：

- Qwen3-0.6B：Windows BF16 local/native 接口与输出检查；原生 BF16 未承诺任意提示逐 token 等于 HF。
- DeepSeek-R1-Distill-Qwen-1.5B：Windows BF16 原生 reference 路径的 32 步 logits 与 4 提示 HTTP 验证。不是完整版 DeepSeek 支持证明。
- Qwen3-8B：历史 L20 实验有独立版本/精度记录，不能挪用为新版本新配置吞吐。
- 完整 DeepSeek、Kimi、MiniMax 的 MoE/特殊注意力配置：当前不能标为 Saddle 自有引擎通用支持。可选后端的支持也应按具体 checkpoint 验证。

详见 [多模型边界](NATIVE_MULTI_MODEL.md)、[原生服务实测](NATIVE_SERVING_ITERATION_20260928.md)、[8B 实验](QWEN3_8B_SERVING_EXPERIMENT.md)。

对外支持一个新组合，至少记录：checkpoint/revision、硬件与驱动、依赖版本、后端/精度、上下文、聊天模板、真实请求结果。
吞吐宣传还需相同工作负载、预热、重复测量、质量门槛、P95/错误率和显存证据。
