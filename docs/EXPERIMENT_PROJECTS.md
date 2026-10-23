# 用 SaddleLLM 命令行运行研究项目

`saddlellm experiment --project <目录> <命令>` 将任务项目接入统一入口。任务保留自己的数据、提示、验证器与报告，SaddleLLM 提供训练算法。入口沿用当前 Python 环境，不自动安装依赖。

公共政策示例位于相邻的 `public_policy_rlvr`，使用 Qwen3-0.6B 比较 Base、SFT、DPO、GRPO、SFT+GRPO。先在该项目的虚拟环境中安装 SaddleLLM，然后执行：

```powershell
$python = '..\public_policy_rlvr\.venv\Scripts\python.exe'
$project = '..\public_policy_rlvr'
$config = '..\public_policy_rlvr\configs\studies\qwen3_06b_direct.yaml'

& $python -m saddlellm.cli experiment --project $project doctor --config $config
& $python -m saddlellm.cli experiment --project $project dry-run --config $config
& $python -m saddlellm.cli experiment --project $project test
& $python -m saddlellm.cli experiment --project $project all --config $config `
  --output-root '..\public_policy_rlvr\outputs\studies\my_study'
& $python -m saddlellm.cli experiment --project $project report --figures `
  --output-root '..\public_policy_rlvr\outputs\studies\my_study'
```

`saddlellm` console script 和 `python -m saddlellm.cli` 等价。公共政策项目支持 `--model <本地完整模型目录>`；先运行小样本预检，再运行固定配方的正式比较。算法、数据和预算改变时使用新的输出目录。项目的完整说明在 `public_policy_rlvr/docs/SADDLE_STUDY_GUIDE.md`。

## 接入另一个项目

在项目根目录添加 `saddlellm.experiment.json`：

```json
{
  "schema_version": 1,
  "entry_point": "my_research.cli:main",
  "pythonpath": ["src"]
}
```

声明的 `main(argv)` 接收参数列表，返回进程退出码。入口在子进程中直接调用该函数，透传退出码，参数包含空格或中文也作为独立参数保留；无需另写 `if __name__ == '__main__'`。项目内的 Python 路径按 manifest 加入导入路径，路径必须位于该项目中。

查看实际启动参数：

```powershell
& $python -m saddlellm.cli experiment --project $project --show-command all --config $config
# -- 后面的 help 属于任务项目。
& $python -m saddlellm.cli experiment --project $project -- --help
```

任务项目自行实现 `doctor/test/run/report` 等命令，SaddleLLM 不假设任务使用相同评估协议。

## 训练接口与记录

公共政策示例使用 `saddlellm.training.LabelledSFT.train_labelled_sft` 训练已标记 completion mask 的数据，使用 `saddlellm.training.Preference.train_preference` 做DPO，使用 `saddlellm.alignment.GRPOTrainer` 在线训练。SFT检查每行在因果移位后仍有监督token，避免截断后只训练提示或完全没有监督。

GRPO可传入 `metrics_callback` 与 `rollout_callback`，记录每个微步及被丢弃的采样组；非有限loss/gradient会中止并保留先前写出的记录。实际优化器更新数应与计划微步数分开报告。`beta=0`时跳过参考模型前向；采样显式启用KV cache，训练前向仍关闭cache。

研究报告应同时保留代码/数据/基础模型指纹、逐例预测、奖励分解、训练失败和指标退化。公共政策示例的 `source_snapshot` 是训练时源码，之后改进的分析代码单独记录，避免把当前工作区误认为当次训练源码。单次训练seed和小预算只支持范围有限的实验结论。

## 新公共政策实验项目

相邻 `public_policy_rl` 提供 Base、SFT、RAG、DPO、SFT+DPO 和在线 RL 组合，在线算法为 GRPO、DAPO、带学习价值头和 GAE 的 PPO。任务的运行说明见 `../public_policy_rl/README.md`。

```powershell
saddlellm experiment --project ../public_policy_rl dry-run --config configs/smoke.yaml --model <完整本地模型目录>
saddlellm experiment --project ../public_policy_rl all --config configs/acceptance.yaml --model <完整本地模型目录>
```

新增通用接口为 `saddlellm.alignment.DAPOTrainer`、`saddlellm.alignment.VerifiablePPO` 和 `saddlellm.experiments.TrainOnlyRetrieval`，均可通过顶层 lazy exports 导入。单设备实现范围及论文参考见任务项目的 `docs/IMPLEMENTATION.md`。
