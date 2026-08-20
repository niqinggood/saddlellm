"""
saddlellm 简易接口 — 一行代码完成最常见操作

设计原则: 隐藏所有复杂度, 每个函数 1-3 个参数, 有合理默认值

用法:
    import saddlellm.easy as llm

    # 创建模型
    model = llm.create("300m")

    # 训练
    llm.train(model, data="data/*.txt", steps=10000)

    # 蒸馏能力
    llm.distill(model, teacher="gpt-4o", prompts=["问题1", "问题2", ...])

    # 对话
    answer = llm.chat(model, "解释一下量子力学")

    # 一键提升
    llm.improve(model)
"""
import os
import logging

logger = logging.getLogger(__name__)

# ============================================================
# 1. 创建模型 — 一行代码
# ============================================================

def create(
    size: str = "300m",
    architecture: str = "llama",
    vocab_size: int = None,
) -> "PreTrainedModel":
    """
    创建模型。只需指定大小。

    size: "100m" | "150m" | "300m" | "350m" | "1b" | "1.5b" | "3b" | "7b" | "8b"
          也支持 "moe-1b" | "mixtral" | "deepseek"

    architecture: 仅用于未匹配到预设时, "llama" | "gpt2" | "gpt-neox" | "mixtral"

    示例:
        model = llm.create("300m")              # 标准 300M Llama
        model = llm.create("1b")                # 1B Llama
        model = llm.create("moe-1b")            # MoE 架构
    """
    from .ModelRegistry import ModelRegistry, MODEL_SPECS

    # 名称映射
    size_map = {
        "100m": "gpt2-small-124m", "124m": "gpt2-small-124m",
        "150m": "llama-tiny-150m",
        "300m": "llama-300m",
        "350m": "gpt-neox-350m",
        "1b": "llama-1b",
        "1.5b": "llama-1.5b",
        "3b": "llama-3b",
        "7b": "llama-7b",
        "8b": "llama-8b",
        "moe-1b": "moe-1b-8e",
        "mixtral": "mixtral-8x7b",
        "deepseek": "deepseek-v3-style",
    }

    spec_name = size_map.get(size.lower(), size)
    if spec_name in MODEL_SPECS:
        spec = MODEL_SPECS[spec_name]
        model = ModelRegistry.create_model(spec, vocab_size_override=vocab_size)
        print(f"创建模型: {spec.name} ({spec.human_params()})")
        return model
    else:
        # 尝试当作自定义大小: "1.2b" → 1.2e9
        try:
            if size.endswith("b"):
                params = float(size[:-1]) * 1e9
            elif size.endswith("m"):
                params = float(size[:-1]) * 1e6
            else:
                raise ValueError(size)

            # 根据参数量找最接近的预设
            closest = min(MODEL_SPECS.values(),
                          key=lambda s: abs(s.estimated_params - params))
            model = ModelRegistry.create_model(closest, vocab_size_override=vocab_size)
            print(f"创建模型: {closest.name} (最接近 {size})")
            return model
        except Exception:
            raise ValueError(f"未知模型大小: {size}. 可用: {list(size_map.keys())}")


# ============================================================
# 2. 训练模型 — 一行代码
# ============================================================

def train(
    model,
    tokenizer=None,
    data=None,
    steps: int = 10000,
    learning_rate: float = None,
    output_dir: str = "./my_model",
    use_ema: bool = True,
    data_lang: str = "zh",
    **kwargs,
):
    """
    训练模型。自动处理数据、优化器、EMA。

    data 可以是:
      - 文件路径: "data/*.jsonl"
      - HuggingFace 数据集名: "wikitext"
      - "auto" 或 None: 自动选择适合该模型的数据

    示例:
        # 最简单的用法
        llm.train(model, steps=10000)  # 自动选数据

        # 指定数据
        llm.train(model, data="data/*.jsonl", steps=50000)

        # 指定语言
        llm.train(model, data="auto", data_lang="zh", steps=20000)
    """
    import torch
    from transformers import AutoTokenizer

    # 1. Tokenizer
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        print("使用默认 GPT-2 tokenizer (建议训练自己的: llm.train_tokenizer())")

    # 2. 数据
    if data is None or data == "auto":
        print(f"自动选择数据 (语言: {data_lang})...")
        from .DataCatalog import DataCatalog
        dataset = DataCatalog.small_model_pack(lang=data_lang, streaming=True)
    elif isinstance(data, str):
        # 判断是文件路径还是数据集名
        if os.path.exists(data) or "*" in data or "." in os.path.splitext(data)[1]:
            # 文件路径
            dataset = _load_data(data, tokenizer)
        else:
            # 尝试作为数据集名
            try:
                from .DataCatalog import DataCatalog
                dataset = DataCatalog.fetch(data, streaming=True)
                print(f"加载数据集: {data} (streaming)")
            except Exception:
                dataset = _load_data(data, tokenizer)
    else:
        dataset = data

    # 3. 模型参数
    total_params = sum(p.numel() for p in model.parameters())

    # 4. 学习率 (自动选择)
    if learning_rate is None:
        if total_params < 5e8:
            learning_rate = 3e-4
        elif total_params < 2e9:
            learning_rate = 2e-4
        else:
            learning_rate = 1e-4

    # 5. 训练
    from .DensePretrainer import DensePretrainer, DensePretrainConfig, init_weights_llama_style

    config = DensePretrainConfig(
        max_steps=steps,
        learning_rate=learning_rate,
        output_dir=output_dir,
        checkpoint_dir=os.path.join(output_dir, "checkpoints"),
        **kwargs,
    )

    init_weights_llama_style(model)

    trainer = DensePretrainer(model, tokenizer, dataset, config)

    # EMA
    ema_callback = None
    if use_ema:
        from .AdvancedTechniques import EMACallback
        ema_callback = EMACallback(model, decay=0.999)

    trainer.train()

    # 保存
    if ema_callback:
        ema_callback.on_save_checkpoint(output_dir, tokenizer)
    else:
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

    print(f"模型已保存到 {output_dir}")
    return trainer


# ============================================================
# 3. 训练分词器
# ============================================================

def train_tokenizer(
    data_path: str,
    vocab_size: int = 32000,
    output_dir: str = "./my_tokenizer",
):
    """训练 BPE 分词器。"""
    from .TokenizerTrainer import TokenizerTrainer
    trainer = TokenizerTrainer(vocab_size=vocab_size)
    trainer.fit(data_path)
    trainer.save(output_dir)
    tokenizer = trainer.get_hf_tokenizer()
    print(f"分词器已保存到 {output_dir} (词表: {trainer.vocab_size})")
    return tokenizer


# ============================================================
# 4. 蒸馏能力 — 一行代码
# ============================================================

def distill(
    model,
    tokenizer=None,
    teacher: str = "gpt-4o",
    prompts=None,
    api_key: str = None,
    examples: int = 500,
    output_dir: str = "./my_distilled_model",
):
    """
    从强模型蒸馏能力。

    teacher:
      - "gpt-4o" / "gpt-4" / "gpt-3.5" → OpenAI API
      - "claude" / "claude-sonnet" → Anthropic API
      - "deepseek" / "deepseek-chat" → DeepSeek API
      - 本地模型路径 → 用本地模型做教师

    示例:
        llm.distill(model, teacher="gpt-4o", prompts=my_prompts)

        # 自动生成 prompts
        llm.distill(model, teacher="claude", examples=1000)
    """
    from transformers import AutoTokenizer
    from .UniversalDistiller import TeacherInterface, AutoDistiller

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token

    # 创建教师
    teacher_map = {
        "gpt-4o": lambda: TeacherInterface.from_openai("gpt-4o", api_key),
        "gpt-4": lambda: TeacherInterface.from_openai("gpt-4", api_key),
        "gpt-3.5": lambda: TeacherInterface.from_openai("gpt-3.5-turbo", api_key),
        "claude": lambda: TeacherInterface.from_anthropic("claude-sonnet-4-6", api_key),
        "claude-sonnet": lambda: TeacherInterface.from_anthropic("claude-sonnet-4-6", api_key),
        "deepseek": lambda: TeacherInterface.from_deepseek("deepseek-chat", api_key),
        "deepseek-chat": lambda: TeacherInterface.from_deepseek("deepseek-chat", api_key),
    }

    if teacher in teacher_map:
        teacher_obj = teacher_map[teacher]()
    elif os.path.exists(teacher):
        # 本地模型
        from transformers import AutoModelForCausalLM, AutoTokenizer as AT
        local_model = AutoModelForCausalLM.from_pretrained(teacher, torch_dtype="auto")
        local_tok = AT.from_pretrained(teacher)
        teacher_obj = TeacherInterface.from_local(local_model, local_tok)
    else:
        teacher_obj = TeacherInterface.from_openai(teacher, api_key)

    # Prompts
    if prompts is None:
        prompts = [f"详细解释 #{i}: {topic}" for i in range(examples)
                   for topic in ["机器学习", "Python编程", "世界历史", "物理学", "哲学思想"]]
        prompts = prompts[:examples]

    # 蒸馏
    auto = AutoDistiller(teacher_obj, model, tokenizer)
    result = auto.auto_distill(
        capabilities=["reasoning", "knowledge", "coding", "writing"],
        total_examples=examples,
    )

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"蒸馏完成! 模型保存到 {output_dir}")
    return result


# ============================================================
# 5. 对话
# ============================================================

def chat(
    model,
    prompt: str,
    tokenizer=None,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    system_prompt: str = None,
    use_cot: bool = False,
    use_self_consistency: bool = False,
) -> str:
    """
    与模型对话。

    示例:
        answer = llm.chat(model, "你好, 解释一下深度学习")
        answer = llm.chat(model, "1+2*3=?", use_cot=True)  # 带推理
    """
    import torch
    from transformers import AutoTokenizer

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token

    full_prompt = ""
    if system_prompt:
        full_prompt += f"系统: {system_prompt}\n\n"
    if use_cot and use_self_consistency:
        from .AdvancedTechniques import SelfConsistency
        sc = SelfConsistency(model, tokenizer)
        result = sc.solve(prompt)
        return result["answer"]
    elif use_cot:
        full_prompt += f"问题: {prompt}\n\n让我们一步步思考。\n\n"
    else:
        full_prompt += f"{prompt}\n\n"

    # 使用安全生成 (防骂人 + 防车轱辘话)
    from .SafeGenerate import SafeGenerate
    sg = SafeGenerate(model, tokenizer)
    result = sg.generate(
        full_prompt if use_cot else f"{prompt}\n\n",
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        enable_safety=True,
        anti_repeat=True,
    )
    return result["text"]


# ============================================================
# 6. 一键提升
# ============================================================

def improve(
    model,
    tokenizer=None,
    data=None,
    steps: int = 5000,
    output_dir: str = "./my_improved_model",
):
    """
    一键应用所有效果提升技术。

    自动: EMA + Model Soup + Self-Consistency + 数据增强
    """
    from transformers import AutoTokenizer

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token

    print("=== 一键提升: 应用所有效果技术 ===")

    # 1. 数据增强
    if data:
        print("1/3: 数据增强 (Evol-Instruct)...")
        try:
            from .AdvancedTechniques import EvolInstruct
            evolver = EvolInstruct(model, tokenizer)
            enhanced = evolver.evolve_batch(
                data if isinstance(data, list) else [data], rounds=1
            )
            print(f"  数据: {len(data) if isinstance(data, list) else 1} → {len(enhanced)} 条")
        except Exception as e:
            print(f"  跳过: {e}")
            enhanced = data

        # 2. 继续训练 + EMA
        print("2/3: 训练 + EMA...")
        trainer = train(model, tokenizer, data=enhanced, steps=steps,
                        output_dir=output_dir, use_ema=True)
    else:
        print("2/3: 跳过 (无新数据)")

    # 3. 自检
    print("3/3: Self-Consistency 验证...")
    from .AdvancedTechniques import SelfConsistency
    sc = SelfConsistency(model, tokenizer, num_samples=5)
    test = sc.solve("1+1=?")
    print(f"  自检: answer={test['answer']}, confidence={test['confidence']}")

    print(f"=== 提升完成! ===")
    return model


# ============================================================
# 7. 评估
# ============================================================

def evaluate(model, tokenizer=None, tasks=None):
    """快速评估模型。"""
    from transformers import AutoTokenizer
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token

    tasks = tasks or ["perplexity"]

    from .BenchmarkRunner import BenchmarkRunner
    runner = BenchmarkRunner(
        model_path=None, tasks=tasks, max_samples=200,
    )
    runner._model = model
    runner._tokenizer = tokenizer

    results = runner.run()
    for r in results:
        print(f"  {r.task}: {r.score:.4f} ({r.metric})")
    return results


# ============================================================
# 8. 列出可用模型
# ============================================================

def list_models():
    """列出所有可用的模型架构。"""
    from .ModelRegistry import ModelRegistry
    print(ModelRegistry.compare_specs())


# ============================================================
# 内部工具
# ============================================================

# ============================================================
# 0. 环境自检 (最先调用)
# ============================================================

def check():
    """运行环境自检。"""
    from .TrainerUtils import check_environment
    return check_environment()

# ============================================================
# 0b. 一键最佳模型
# ============================================================

def auto(model_size: str = "300m", lang: str = "zh", output_dir: str = "./my_model",
         use_api: bool = False):
    """一键训练最佳小模型。"""
    from .TrainerUtils import auto_train_best_small_model
    return auto_train_best_small_model(model_size, output_dir=output_dir, lang=lang,
                                        use_api_teacher=use_api)

# ============================================================
# 9. 自动配置 — 根据硬件推荐最佳设置
# ============================================================

def auto_config(model=None, model_size: str = None):
    """
    自动检测 GPU 并推荐最佳训练配置。

    示例:
        config = llm.auto_config()
        # → {"strategy": "single", "batch_size": 8, "grad_accum": 4, ...}

        config = llm.auto_config(model_size="1b")
    """
    import torch

    # 检测 GPU
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    gpu_name = ""
    gpu_mem = 0
    if gpu_count > 0:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_mem / (1024**3)
        gpu_tflops = {"A100": 312, "H100": 990, "H800": 990, "A800": 312, "V100": 125,
                       "RTX 4090": 165, "RTX 3090": 71, "T4": 65}.get(
                           next((k for k in ["A100", "H100", "H800", "A800", "V100", "RTX 4090", "RTX 3090", "T4"] if k in gpu_name), ""), 100
                       )
    else:
        gpu_tflops = 0
        print("未检测到 GPU, 使用 CPU 训练 (只适合 100M 以下模型)")

    # 估算模型参数量
    total_params = 0
    if model is not None:
        total_params = sum(p.numel() for p in model.parameters())
    elif model_size:
        from .ModelRegistry import ModelRegistry, MODEL_SPECS
        size_map = {"100m": "gpt2-small-124m", "300m": "llama-300m", "1b": "llama-1b",
                     "3b": "llama-3b", "7b": "llama-7b", "moe-1b": "moe-1b-8e"}
        spec_name = size_map.get(model_size, model_size)
        if spec_name in MODEL_SPECS:
            total_params = MODEL_SPECS[spec_name].estimated_params

    if total_params == 0:
        total_params = 3e8  # 默认 300M

    # 策略推荐
    if gpu_count == 0:
        strategy = "cpu"
        batch_size = 1
    elif gpu_count == 1:
        strategy = "single"
        batch_size = max(2, int(gpu_mem / 4))  # 每 4GB 显存 = 1 batch
    elif gpu_count <= 4:
        strategy = "deepspeed_zero2"
        batch_size = max(2, int(gpu_mem / 6))
    else:
        strategy = "deepspeed_zero3"
        batch_size = max(1, int(gpu_mem / 8))

    # 学习率
    if total_params < 5e8:
        lr = 3e-4
    elif total_params < 2e9:
        lr = 2e-4
    else:
        lr = 1e-4

    # 序列长度
    if total_params < 5e8:
        seq_len = 1024
    elif total_params < 2e9:
        seq_len = 2048
    else:
        seq_len = 4096

    # 梯度累积 (凑 global batch)
    grad_accum = max(1, int(256 / (batch_size * max(1, gpu_count))))

    config = {
        "gpu_count": gpu_count,
        "gpu_name": gpu_name,
        "gpu_memory_gb": round(gpu_mem, 1),
        "gpu_tflops": gpu_tflops,
        "strategy": strategy,
        "per_device_batch_size": batch_size,
        "gradient_accumulation_steps": grad_accum,
        "global_batch_tokens": batch_size * max(1, gpu_count) * grad_accum * seq_len,
        "learning_rate": lr,
        "max_seq_length": seq_len,
        "model_params": total_params,
        "model_params_human": f"{total_params/1e9:.1f}B" if total_params >= 1e9 else f"{total_params/1e6:.0f}M",
        "estimated_peak_vram_gb": round(total_params * 2 / (1024**3) * 4, 1),  # 模型+优化器+激活 ≈ 4x
        "recommendation": (
            "完美, 可以直接训练!" if gpu_count > 0 and total_params < 3e9
            else "建议减小 model size 或增加 GPU" if gpu_count > 0
            else "需要 GPU 才能训练 > 100M 模型"
        ),
    }

    print(f"""自动配置:
  GPU: {gpu_count}x {gpu_name} ({gpu_mem:.0f}GB, {gpu_tflops}TFLOPS)
  模型: {config['model_params_human']}
  策略: {strategy}
  Batch: {batch_size}/device × {grad_accum} grad_accum = {config['global_batch_tokens']:,} tokens/step
  学习率: {lr}
  序列长度: {seq_len}
  预估显存: {config['estimated_peak_vram_gb']}GB (可用 {gpu_mem:.0f}GB)
  → {config['recommendation']}""")

    return config


def _load_data(data_path: str, tokenizer):
    """加载和预处理数据。"""
    from .DataPipeline import DataPipeline, PipelineConfig

    config = PipelineConfig(
        sources=[{"type": "local", "path": data_path}],
        max_seq_length=2048,
    )
    pipeline = DataPipeline(config)
    pipeline.collect().clean().deduplicate().filter_quality()
    pipeline.tokenize_and_pack(tokenizer)
    return pipeline.to_iterable_dataset()
