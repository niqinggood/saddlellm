# SaddleLLM 推理优化与下一步优先级（2026-09-28）

## 医疗 Web 服务当前状态

09-28 约 07:24（北京时间）做了只读检查：公网首页及前端资源返回 HTTP 200；
`/api/auth/status` 返回 200，显示未登录访客需要鉴权。服务器上
`medmodel-web.service`、4B/8B 模型、embedding、reranker 和 RAG 单元均为
`active/running`，回环首页与两个模型发现接口返回 200。GPU 观察为空闲
12,880 MiB。服务原本就在运行，因此没有重启、替换模型、改动业务环境或开放新端口。

这些检查说明页面、认证状态接口、服务进程和模型发现链路在线；未凭管理员身份执行
真实医疗问答、RAG 内容质量或长时间压力验收，不能据此宣称整个业务无故障。
另一个明确的上线风险是：公网 IP 的 HTTP 首页直接返回登录页，没有重定向 HTTPS；
前端同源调用 `/api/auth/login`，该 IP 的 443 端口在本轮探测中不可连接。
如果用户直接从此 HTTP 地址登录，账号密码不会有传输层加密。可能另有 HTTPS 域名，
但本轮没有证据证明它覆盖这个 IP 入口。上线前应配置 HTTPS 域名并拒绝或重定向
HTTP 登录；没有域名/证书信息前，不擅自改动生产网关。

## 本轮已做：W8A16 短批次宽投影分块

在 [Int8WeightKernels.py](../saddlellm/models/Int8WeightKernels.py) 中，仅对
`5–16` 行、输入维为 `4096`、输出维至少 `6144` 的 W8A16 GEMM，
把分块从 `32×64` 改为 `16×128`。K 方向仍为 64，其他形状、单行 GEMV、
原始浮点和 vLLM 路线都不变。这针对原生 Qwen3-8B 固定微批中的 QKV 与
gate/up 宽投影；没有增加用户参数或启动时自动搜索开销。

本机 RTX 4090 Laptop、Torch 2.6.0+cu124、Triton 3.3.0，合成 INT8 权重和
BF16 激活，交替顺序各 8 次、报告中位数；数字只包括一次线性层调用，
不是完整模型或 HTTP 吞吐：

| 矩阵形状（行×输入×输出） | 原分块 | 新分块 | 观察加速 |
| --- | ---: | ---: | ---: |
| 8×4096×6144 | 0.1022 ms | 0.0871 ms | 1.17× |
| 8×4096×24576 | 0.4905 ms | 0.3445 ms | 1.42× |
| 8×12288×4096，对照不改分块 | 0.1762 ms | 0.1765 ms | 约 1.00× |

该组输入下新旧算子输出逐元素完全一致；专项测试还覆盖 FP16/BF16、
不同形状与 fallback。`tests/test_cuda_int8_linear.py` 63 项通过，
另有 148 项推理/服务相关回归通过。重跑命令：

```powershell
python tools/benchmark_w8a16_tiles.py --repeats 8
python -m pytest -q tests/test_cuda_int8_linear.py tests/test_inference_batching.py tests/test_native_generation_session.py tests/test_serving_plan.py tests/test_inference_platforms.py tests/test_vllm_serving.py tests/test_inference_optimizer.py
```

这次**尚未**将新代码部署到医疗服务器，也未重测完整 8B 的 token/s；
既有的 252.8 聚合输出 token/s 是优化前、另一台 L20 上的固定微批结果，
不能叠加 1.42× 宣称新吞吐。128 行 prefill 分块试验收益波动，最终没有启用。

## 接下来优先加强什么

1. **P0：合批回答一致性。** 既有完整 8B 测试中，逐请求与 8 路微批的返回文本哈希不同。
   先定位逐 token 首次分歧、logits 边际、padding/位置和量化矩阵形状；用护理业务题集
   检验影响。未完成前保持微批显式开启，不能宣传“无损提速”。
2. **P0：流式高并发。** 原生 8B 的 C8 流式仍约 31.8 聚合 token/s，P95 首段文字
   约 14.1 秒。优先在资源允许时做完整 8B 的新版 vLLM 路线同精度、同提示分布实测；
   Saddle 自有连续批处理和 KV 分页属于更长线工程，不能用调大 `max_concurrency` 代替。
3. **P0：医疗质量与发布门禁。** 框架服务速度不等于诊疗安全。训练侧应按题源分组切分，
   阻断近重复跨 split；`TrainingOrchestrator._run_eval_stage` 仍调用旧 `Evaluator`，
   应接入新选择题、开放回答、提示鲁棒性和严重错误审核，并保留逐题证据后再导出。
4. **P0：公网登录传输安全。** 给实际访问入口启用可信 HTTPS 证书，阻止 HTTP 提交登录；
   再检查鉴权、速率限制和审计。若目前已有独立 HTTPS 域名，仍要处理可访问的 HTTP IP 登录页。
5. **P1：可复现的容量验收。** 固定模型、量化、thinking、输入/输出长度、并发和缓存；
   同时报告 TTFT、P95/P99、总输出 token/s、错误/503、显存与取消/重启。
   不能只用健康接口 200 代替业务验收。
6. **P1：交付与启动效率。** W8A16 目前每次从原始浮点权重重新量化；若要缩短重启时间，
   可设计带哈希和版本校验的可重载量化包。新 wheel、Linux GPU、macOS/MPS 和容器实测
   都应与源码测试分别记录，不能把本地源码修改算作线上已升级。

总体顺序是先保证输出和医疗验收，再兑现并发性能，最后扩大平台与量化覆盖；
使用入口继续保持 `saddlellm serve` 和简单 Python import。
