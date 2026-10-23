# 候选版本发布验收

以下用于对外试用包，不授权脚本自动上传 PyPI、推送仓库或修改线上服务。

## 构建前

- [ ] 明确候选版本、变更说明、已知限制和上一版回退方法。
- [ ] 审查 `git status --short`，纳入应交付的源码/测试/文档，排除本地实验、密钥、历史数据库和私有数据；不要直接 `git add .`。
- [ ] `artifacts/`、`outputs/`、模型权重、wheel、缓存和本地会话数据库仅保存在本机，由 `.gitignore` 排除。实验文档中的 `../artifacts/...` 是本地证据路径，仓库克隆不包含这些结果；需要公开的报告应脱敏后整理到 `docs/`。
- [ ] 用 `git ls-files -ci --exclude-standard` 检查已经跟踪但命中忽略规则的文件；`.gitignore` 不会自动移除已经跟踪的文件。
- [ ] 暂存后检查 `git diff --cached --stat` 和 `git diff --cached --name-only`，确认没有实验数据、重复安装包、凭据或本机服务器地址。保留 `saddlellm/runtime/assets/` 和 `saddlellm/spatial_studio_web/`，它们是安装包运行所需的静态资源。
- [ ] 运行 `python tools/check_release_consistency.py`。
- [ ] 运行 `python tools/generate_function_reference.py`，检查 `outputs/FUNCTION_REFERENCE.md`，再运行 `python tools/generate_function_reference.py --check`。报告按需生成，不提交；设计参考 PNG 和导出的 Word 文档也仅保留在本地。
- [ ] 按 [支持范围](SUPPORTED_ENVIRONMENTS.md) 分开标注稳定、实验性和未验收能力。

## 包验收

- [ ] `python -m build` 和 `python tools/check_wheel_contents.py <wheel路径>` 通过。
- [ ] 新建独立环境，安装 wheel 及所需 extra，执行 `python -m pip check`。
- [ ] 离开源码目录，使用 `python -I /源码/tools/smoke_installed_wheel.py --source-root /源码 --output /验收目录/report.json`。
- [ ] 上述报告 accepted 为 true：真实 CLI 启动、tiny CPU 模型生成、OpenAI SDK chat/completion/SSE、静态资源、关闭历史和请求释放全部通过。
- [ ] Windows/Linux 分别验收；`.github/workflows/inference-platforms.yml` 已加入该流程，但配置存在不等于作业通过。
- [ ] 安装前后版本、依赖清单、wheel SHA256 和原始报告归档，不覆盖失败记录。

## 模型与服务

- [ ] 运行三份上手教程；微型/5 步 smoke 只证明工程链路，不证明训练效果。
- [ ] 每个声称支持的真实模型至少完成启动、单轮、多轮、流式及错误请求检查。
- [ ] 目标 GPU 做固定工作负载基准和至少 30 分钟稳态/混合负载运行；另测慢客户端、取消与排队。
- [ ] 宣称双卡/多卡前完成对应硬件实测；单 GPU 或替身测试不能代替。
- [ ] 运行中记录显存、错误率、首字、P95 和吞吐；不得以 max_concurrency 配置值代替实际容量。

## 交付

- [ ] 用目标网关/鉴权配置验证；多人场景关闭共享历史或由业务层实施用户隔离。
- [ ] 请未参与开发的试用者独立完成安装和第一条请求，记录卡点。
- [ ] 提供问题反馈入口、更新说明、安全问题非公开联系方式。
- [ ] 经维护者确认后再提交/上传，保留上一版安装包与回退说明。

本轮结果见 [发布准备记录](RELEASE_READINESS_20260929.md)。完整多卡、MPS 真机、长期压力测试和独立用户试用仍属于后续验收，不在本轮自动打勾。
