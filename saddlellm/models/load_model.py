
import os
def unsloth_load_lora_model(MODEL_PATH):
    print("begin unsloth load_lora_model")
    max_seq_length  = 2048
    dtype           = None
    load_in_4bit    = True

    from unsloth import FastLanguageModel
    import torch

    # 确保GPU可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        # 加载模型和分词器
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=MODEL_PATH,
            max_seq_length=max_seq_length,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
        )

        # 将模型移至设备并启用推理优化
        model = model.to(device)
        _ = FastLanguageModel.for_inference(model)

        # 修正提示模板，只需要两个占位符
        alpaca_prompt = """### Question:
        {}

        ### Response:
        {}"""

        # 准备输入
        inputs = tokenizer(
            [
                alpaca_prompt.format(
                    "What are the key principles of deep learning?",  # 问题
                    ""  # 回答留空用于生成
                )
            ],
            return_tensors="pt"
        ).to(device)

        # 设置生成参数
        generate_kwargs = {
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.9,
            "max_new_tokens": 128,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id
        }

        # 执行生成
        from transformers import TextStreamer
        text_streamer = TextStreamer(tokenizer)
        with torch.no_grad():
            _ = model.generate(**inputs, streamer=text_streamer, **generate_kwargs)

    except Exception as e:
        print(f"生成过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
    print("finish unsloth load_lora_model")
    return




def transfomer_load_lora( MODEL_PATH ):
    print("begin transfomer_load_lora")
    import torch
    from peft import AutoPeftModelForCausalLM
    from transformers import AutoTokenizer, BitsAndBytesConfig
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,  # 计算时使用的数据类型
        bnb_4bit_quant_type="nf4",  # 量化类型，nf4或fp4
        bnb_4bit_use_double_quant=True  # 是否使用双重量化
    )

    model = AutoPeftModelForCausalLM.from_pretrained(
        MODEL_PATH,  # YOUR MODEL YOU USED FOR TRAINING
        quantization_config=quantization_config,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    alpaca_prompt = """### Question:
        {}

        ### Response:
        {}"""

    prompt = alpaca_prompt.format("Explain the concept of machine learning in simple terms.", "")

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=1024,
        temperature=0.7,
        top_p=0.95,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id
    )
    # outputs
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(decoded_output)
    print("finish## transfomer_load_lora")
    return

def src_mode_transformer_load(MODEL_PATH  = r'E:\reactflow_test\backend\model\lora_model_step_120'):
    print("begin src_mode_transformer_load")
    import os
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import os
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'device: {device}')
    # 模型的绝对路径

    abs_model_path  = os.path.abspath(                MODEL_PATH )
    model           = AutoModelForCausalLM.from_pretrained(     MODEL_PATH,torch_dtype=torch.bfloat16,device_map=device )
    tokenizer       = AutoTokenizer.from_pretrained(        MODEL_PATH)
    pipeline        = transformers.pipeline(  "text-generation",  model=model,    tokenizer=tokenizer,    device_map=device)

    messages = [
        {"role": "system", "content": "You are a helpful AI assistant"},
        {"role": "user", "content": "Explain the concept of machine learning in simple terms."},
    ]

    prompt = tokenizer.apply_chat_template(     messages,    tokenize=False,     add_generation_prompt=True )
    outputs = pipeline( prompt,max_new_tokens=256,    eos_token_id=tokenizer.eos_token_id,    do_sample=True,    temperature=0.6,top_p=0.9,)
    print(outputs[0]["generated_text"][len(prompt):])
    print("finish src_mode_transformer_load")
    return


if __name__ == '__main__':
    MODEL_PATH = r"E:\reactflow_test\backend\model\Qwen\Qwen2___5-0___5B-Instruct"
    DATASET_PATH = r'E:\ml_data\medical_zh\train_zh_0.jsonl'
    OUTPUT_PATH = r'E:\reactflow_test\backend\model\lora_model_2424'
    transfomer_load_lora(MODEL_PATH=r'E:\reactflow_test\backend\model\lora_model_2423_step_120')
    unsloth_load_lora_model( MODEL_PATH = r'E:\reactflow_test\backend\model\lora_model_2423_step_120' )
    src_mode_transformer_load()
