# 安装与部署排错

先运行 `saddlellm doctor --profile serve`（训练改为 `train`；原生为 `native`；vLLM 为 `vllm`），再运行 `python -m pip check`。
用 `--output doctor.json` 保存诊断；公开问题前仍须人工移除路径中的个人信息，不上传 key、聊天记录、私有数据或权重。

| 症状 | 先检查 | 处理方式 |
| --- | --- | --- |
| 找不到 saddlellm 命令 | 当前激活环境与安装位置 | 使用该环境的 Scripts/bin 中的命令；确认 `python -m pip show saddlellm` |
| Missing dependency / outside supported range | 是否装了正确 extra、是否混用后端环境 | 新建独立环境，安装对应 wheel extra；不要直接覆盖正在运行服务的环境 |
| CUDA unavailable | Torch 是否为 CPU 版、驱动是否可用 | 安装匹配的 CUDA Torch；无 GPU 时显式 CPU 验证小模型 |
| 显存不足 | doctor 的每卡 `free_gb`，以及其他 GPU 进程 | 缩短上下文、减小并发、选更小模型；量化会改变质量，不能自动开启 |
| `free_gb=null` | 驱动查询失败 | 视为未知，不用总显存替代；检查驱动或容器设备权限 |
| 本地模型文件缺失 | 是否有 config、tokenizer、全部权重分片 | 准备完整目录；`--local-files-only` 不负责下载 |
| native 不支持配置 | 模型是否为 MoE、特殊 RoPE 等 | 查支持矩阵；显式改用兼容后端，原生不会偷偷切换 HF |
| 端口占用 | 是否已经启动过服务 | 选择其他 `--port`；不要直接终止未知进程 |
| 输入太长 | 输入与输出总预算 | 缩短输入或调整已验证的服务上限；默认不静默截断 |
| 队列满 / 503 | 并发、排队、token 预算 | 客户端带退避重试，减少请求；不要无限增大队列 |
| 历史接口 404 | 是否使用 `--no-chat-history` / `--no-web-ui` | 属于预期；关闭历史时仍可正常聊天，刷新不保留当前对话 |
| 历史接口 401/403 | key、访问来源 | 使用正确 key；无鉴权历史只允许 loopback，不能据此提供多人私密空间 |
| 思考很多但还没给答案 | 模型模板是否默认思考、输出预算 | 明确选择是否使用 `--disable-thinking`，并记录其质量/延迟取舍 |

多人部署时，内置历史是共享工作区，建议先使用 `--no-chat-history`，用户隔离交给业务层。
此开关不是全系统“零留存”：反向代理、外部观测服务和用户截图等仍需独立管理。旧数据库不会被删除，需要保留时先备份，由管理员明确决定清理。

反馈问题请包含：版本、系统/Python、硬件、具体模型和后端、脱敏后的命令、退出码、最小复现步骤、doctor 报告和错误片段。不要提交 `.env` 或整份聊天数据库。
