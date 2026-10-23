# 2026-09-29 公开试用准备记录

本轮处理面向使用者的 P0 问题，没有上传 PyPI、提交/推送 Git，也没有替换现有 8127–8130 服务。
它不是“全部发布门槛已满足”的声明，后续条目见 [发布清单](RELEASE_CHECKLIST.md)。

## 已交付

1. `doctor --profile train|serve|native|vllm --device auto|cpu|cuda|mps`：按用途选择依赖，保留默认 train 和退出码语义。报告分开列出阻断问题与资源警告。
2. 显存按每张 CUDA 卡的 `mem_get_info` 驱动快照读取；失败显示未知，不用总显存减当前 Python 占用替代。旧训练工具的显存计算也修正，不再仅凭卡数推荐模型规模。
3. `serve --no-chat-history` 与 `serve(..., chat_history=False)`：关闭内置历史所有读取/写入，不初始化数据库、不删除旧文件。local/native 与 vLLM 共用策略，health 与页面显示实际状态。
4. 补充安装入口、三条教程、合成样例、支持矩阵、排错、发布清单、安全说明和 issue 模板；README 增加面向新用户的入口。
5. 新增隔离安装包验收脚本，以及 Windows/Linux CI 作业。脚本离开源码目录、用 `python -I` 运行并验证实际导入的是已安装包，然后启动真实 loopback HTTP 服务。
6. 重新生成函数手册，修复原先发布检查提示的过期内容。
7. 兼容已运行的旧 Web 渲染进程读取新版静态资源：模板采用可直接显示的历史默认值，不向旧服务暴露新占位符。8127/8129 实际页面已确认无未替换内容，无需重启。

## 已执行的验证

- 最终相关回归 **311 项全部通过**，含诊断、历史策略、原生架构、流式、OpenAI 协议、vLLM 替身、Web UI、wheel 契约、tiny 后训练及 LoRA 导出重载。[最终报告](../artifacts/release_readiness_20260929/regression_release.xml)。早期 305/310 项报告保留；补全包契约必需文件时发现旧测试夹具缺少文件，已修正并增加漏包反例测试。
- Windows 全新 venv，安装 CPU Torch 及 wheel 的 `[serve]` extra；`pip check` 通过。未借用开发环境的 site-packages，未用 editable 安装。
- 另建仅安装基础 wheel 的轻量 venv：无 Torch，CLI help 正常，doctor 返回结构化缺失依赖报告和退出码 1，`pip check` 通过。CI 基础安装已移除 `--no-deps`，避免依赖 runner 预装的 PyYAML。
- 安装包实际启动 tiny CPU 原生模型，OpenAI SDK 完成 chat/completion/SSE 与 usage 检查；静态资源正常，关闭历史时不创建 `.saddlellm`，请求结束后 active=0。[首次报告](../artifacts/release_readiness_20260929/installed_wheel_windows.json)、[最终包复验](../artifacts/release_readiness_20260929/installed_wheel_windows_release.json)。9 个核心变更源码/资源与最终 wheel 内容逐一一致。
- 新教程训练命令 dry-run 通过；8 条训练、4 条验证样例通过格式与跨集合文本检查，group id 缺失提示保留。[数据报告](../artifacts/release_readiness_20260929/tutorial_data_audit.json)
- 函数手册、发布元数据、wheel 内容契约、Python 错误级静态检查、JavaScript 语法和新增 YAML 语法检查通过。
- 本机开发环境被新 doctor 正确识别为 FastAPI/Uvicorn/Pydantic 低于声明范围；未升级原环境。独立安装环境满足声明范围并通过 SDK 验收。

浏览器验证遵循 frontend-testing-debugging 流程；该技能要求的 Browser 插件不可用，使用已有 Playwright 1.45 与 headless Edge。
测试入口为临时 `127.0.0.1:8151/8152`，完成后均已关闭；模型为随机 tiny CPU 功能夹具，非回答质量测试。

| 页面检查 | 结果 |
| --- | --- |
| URL/标题、首屏非空、无框架错误遮罩 | 通过 |
| 浏览器相关 warning/error、脚本异常 | 无 |
| 关闭历史：发送 → 显示回答 → 刷新 | 通过；历史请求数 0，刷新后消息清空 |
| 开启历史：发送 → 保存 → 刷新 → 打开历史 | 通过；记录与消息正常恢复 |
| 1365×900 桌面与 390×844 手机侧栏截图检查 | 通过；截图保存于临时验收目录 |

## 仍需验收 / 未声称完成

- Linux/macOS 的本轮 CI 未在此会话实际执行；本机 Python 是 3.11.0rc2，不能代替 Python 稳定版/其他版本矩阵。CI 使用 3.11 分支，仍须检查真实作业结果。
- 双 GPU、多机、MPS 真机，以及至少 30 分钟压力/故障测试仍待目标设备验收。
- 新三份教程中的完整 Qwen3 LoRA 实训、业务质量评测及独立用户试用仍待完成；本轮 dry-run 与 tiny 回归不能代替真实数据效果。
- 当前聊天历史仍非多租户系统；关闭历史是可选保护，不是新增用户隔离。
- 项目存在大量既有修改和未跟踪文件，需要维护者审查纳入提交；未执行 `git add .`，未把本地实验材料整体发布。
- 本轮没有吞吐提速结论。发行准备与推理性能改进分开记录。

最终候选 wheel 放在 `artifacts/release_readiness_20260929/wheel_release/`。`wheel/` 和 `wheel_final/` 是保留的早期候选，不作为最终交付。该包不是对当前线上/本机运行服务的自动升级。

SHA256：`afc3f88d44d43492adeca65a2edf4fbd20b41cf83f183bdbf9bf12fae6661822`。

后续同日增加了非流式断连取消修正，并构建了单独的新候选；本页以上是首批发布准备的历史验收记录。
最新推理改动、复验与候选位置见 [推理生命周期迭代](INFERENCE_LIFECYCLE_20260929.md)，不要把旧 wheel 当作已包含新改动。

临时验收目录位于系统 Temp 下的 `saddle-release-e9839ee0e13447368daac84e340e870b`；清理其中两套临时 venv 的命令被工具策略阻止，环境、脚本和截图均保留。没有清理或升级用户原有 Python 环境。
