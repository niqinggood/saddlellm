# SaddleLLM Stage 插件

SaddleLLM 的插件扩展面位于 `saddlellm.framework`。它允许在不修改
`TrainingOrchestrator` 的情况下增加训练 Stage。模型执行路径目前限定为项目
内置的 `hf` 和 `saddle` 两种。

## 查看当前插件

```powershell
saddle-llm plugins
```

命令只读取 Python 包的 entry-point 元数据。第三方插件实现不会因为执行
`saddle-llm plugins` 而被导入；只有配置实际引用插件时才会加载。

## 配置

插件和内置 Stage 使用同一种普通配置：

```yaml
metadata:
  name: domain-sft-v1
  labels:
    owner: llm-team
model:
  backend: hf
  name_or_path: Qwen/Qwen2.5-0.5B-Instruct
stages: [sft, eval]
sft:
  data_path: data/sft.jsonl
  use_lora: true
eval:
  enabled: true
logging:
  output_dir: outputs/domain-sft-v1
```

## 创建 Stage 插件

Stage 是无参数构造的类或一个已经创建的实例。运行状态由 `StageContext` 提供，
插件不应把一次运行的可变状态保存在全局变量中。

```python
from saddlellm.framework import BaseStage, StageCapabilities


class QualityAuditStage(BaseStage):
    name = "quality_audit"
    capabilities = StageCapabilities(
        distributed_strategies=frozenset({"single"}),
        tags=frozenset({"data", "quality"}),
    )

    def validate(self, context):
        audit = getattr(context.config, "quality_audit", None)
        if audit is None:
            raise ValueError("quality_audit configuration is required")

    def run(self, context):
        return {
            "status": "completed",
            "output_dir": str(context.output_dir),
            "upstream_stages": sorted(context.results),
        }
```

返回值必须是可写入 JSON 的 mapping，或者返回 `None` 并由插件自己写入
`context.results`。`validate()` 只能做静态检查，不应下载模型或分配 GPU。

在插件项目的 `pyproject.toml` 中声明：

```toml
[project.entry-points."saddlellm.stages"]
quality_audit = "my_saddle_plugin:QualityAuditStage"
```

安装插件后，配置可直接写入：

```yaml
stages: [quality_audit]
logging:
  output_dir: outputs/audit
```

开发和单元测试期间也可以显式注册：

```python
from saddlellm.framework import register_stage

register_stage(QualityAuditStage)
```

`model.backend` 不是插件入口。目前只能设置为 `hf` 或 `saddle`，两者在
`ModelAdapter.py` 中通过一个内部字典选择。需要第三种真实执行路径时，再扩展该接口。

## 当前边界

- Stage 目前仍按顶层 `stages` 顺序执行；显式依赖 DAG 是下一阶段能力。
- 只有 `pretrain` 内置 Stage 声明了 DDP/FSDP/DeepSpeed 策略；其他内置 Stage
  继续 fail closed。
- `uses_model_adapter=true` 的 Stage 会经过内置模型能力校验；自定义数据处理 Stage
  默认不依赖模型适配器。
- entry point 代表已安装的本地 Python 代码。部署环境仍应锁定依赖、审查插件包并
  使用隔离环境。
