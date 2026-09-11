import argparse
import os
import json
import torch
import logging
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GPTQConfig
)
from datasets import load_dataset, Dataset
import tempfile

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# 本地模型和数据路径
MODEL_PATH = r"E:\reactflow_test\backend\model\Qwen\Qwen2___5-0___5B-Instruct"
DATASET_PATH = r'E:\ml_data\medical_zh\train_zh_1000.jsonl'


def load_model(model_path, device):
    """加载原始模型"""
    logger.info(f"加载模型: {model_path}")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        return model.eval()
    except Exception as e:
        logger.error(f"加载模型失败: {str(e)}")
        raise


def prepare_calib_data(tokenizer, data_path, num_samples=128, seq_len=128):
    """准备校准数据，返回文本列表"""
    logger.info(f"准备校准数据，使用本地数据集: {data_path}")

    with open(data_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    texts = []
    for i in tqdm(range(min(num_samples, len(lines)))):
        try:
            data = json.loads(lines[i])
            # 合并instruction/input/output
            text = f"{data.get('instruction', '')} {data.get('input', '')} {data.get('output', '')}"
            text = text.strip()

            if len(text) > 10:
                texts.append(text)

        except Exception as e:
            logger.warning(f"跳过第{i}行: {str(e)}")
            continue

    logger.info(f"成功加载 {len(texts)} 个校准文本")
    return texts


def quantize_model(args, model, tokenizer):
    """执行模型量化，改进数据集格式"""
    quant_method = args.method.lower()
    output_path = os.path.abspath(args.output_path)
    os.makedirs(output_path, exist_ok=True)

    try:
        if quant_method in ["int8", "int4"]:
            logger.info(f"使用bitsandbytes进行{quant_method.upper()}量化")

            # 配置量化参数
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=quant_method == "int4",
                load_in_8bit=quant_method == "int8",
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            )

            # 释放原始模型内存
            del model
            torch.cuda.empty_cache()

            # 重新加载量化后的模型
            model = AutoModelForCausalLM.from_pretrained(
                args.model_path,
                device_map="auto",
                trust_remote_code=True,
                quantization_config=bnb_config
            )

            # 保存量化模型
            model.save_pretrained(output_path)
            tokenizer.save_pretrained(output_path)

        elif quant_method == "gptq":
            logger.info("使用transformers原生GPTQ量化（针对Qwen2优化）")

            # 准备校准文本
            calib_texts = prepare_calib_data(
                tokenizer,
                args.data_path,
                args.calib_samples,
                args.seq_len
            )

            # 释放原始模型内存
            del model
            torch.cuda.empty_cache()

            # 使用transformers原生的GPTQ量化方法
            try:
                logger.info("使用transformers>=4.30.0的GPTQ量化方法")

                # 直接使用文本列表作为校准数据集
                gptq_config = GPTQConfig(
                    bits=4,
                    group_size=args.group_size,
                    dataset=calib_texts,  # 直接传递文本列表
                    desc_act=not args.sym_quant,
                    damp_percent=args.damp_percent,
                )

                # 加载并量化模型
                model = AutoModelForCausalLM.from_pretrained(
                    args.model_path,
                    device_map="auto",
                    torch_dtype=torch.float16,
                    trust_remote_code=True,
                    quantization_config=gptq_config
                )

                # 保存量化模型
                model.save_pretrained(output_path)
                tokenizer.save_pretrained(output_path)

            except Exception as e:
                # 旧版本的量化方法或其他错误
                logger.warning(f"使用GPTQ量化方法失败，尝试替代方案: {str(e)}")

                # 创建临时目录存储校准数据
                with tempfile.TemporaryDirectory() as tmpdir:
                    # 将文本转换为数据集并保存
                    calib_dataset = Dataset.from_dict({"text": calib_texts})
                    calib_dataset.save_to_disk(os.path.join(tmpdir, "calib_data"))

                    # 使用shell命令调用transformers-cli进行量化
                    logger.info("开始执行GPTQ量化命令行工具...")
                    cmd = f"""
                    transformers-cli quantize \
                        --model_name_or_path {args.model_path} \
                        --output_dir {output_path} \
                        --quantization_method gptq \
                        --bits 4 \
                        --dataset {os.path.join(tmpdir, "calib_data")} \
                        --trust_remote_code \
                        --group_size {args.group_size} \
                        --symmetric {args.sym_quant}
                    """

                    import subprocess
                    result = subprocess.run(
                        cmd,
                        shell=True,
                        capture_output=True,
                        text=True
                    )

                    if result.returncode != 0:
                        logger.error("量化命令执行失败")
                        logger.error(result.stderr)
                        raise RuntimeError("GPTQ量化失败")
                    else:
                        logger.info("量化命令执行成功")
                        logger.info(result.stdout)

        logger.info(f"量化完成，模型已保存到: {output_path}")
        return True

    except Exception as e:
        logger.error(f"量化过程中出错: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def test_model(model_path, tokenizer, prompt=None):
    """测试量化模型"""
    logger.info("测试模型生成...")

    try:
        # 检查是否为GPTQ量化模型
        is_gptq_quantized = os.path.exists(os.path.join(model_path, "gptq_config.json"))
        # 检查是否为bitsandbytes量化模型
        is_bnb_quantized = os.path.exists(os.path.join(model_path, "quantization_config.json"))

        if is_gptq_quantized:
            # 加载GPTQ量化模型
            logger.info("加载GPTQ量化模型")
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
        elif is_bnb_quantized:
            # 加载bitsandbytes量化模型
            logger.info("加载bitsandbytes量化模型")
            bnb_config = BitsAndBytesConfig.from_pretrained(model_path)
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto",
                trust_remote_code=True,
                quantization_config=bnb_config
            )
        else:
            # 加载原始模型
            logger.info("加载原始模型")
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto",
                trust_remote_code=True
            )

        test_prompt = prompt if prompt else "请解释一下糖尿病的主要症状有哪些？"
        inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=200,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.pad_token_id
            )

        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"生成结果:\n{result}")
        return result
    except Exception as e:
        logger.error(f"模型测试失败: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise


def main():
    parser = argparse.ArgumentParser(
        description="大模型量化工具 (支持int8/int4/gptq量化，适配Qwen2模型)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # 模型参数
    parser.add_argument("--model-path", default=MODEL_PATH, help="模型路径")
    parser.add_argument("--data-path", default=DATASET_PATH, help="校准数据集路径")

    # 量化参数
    parser.add_argument("--method",
                        choices=["int8", "int4", "gptq"],
                        default="gptq",
                        help="量化方法")
    parser.add_argument("--group-size",
                        type=int,
                        default=128,
                        help="GPTQ分组大小")
    parser.add_argument("--damp-percent",
                        type=float,
                        default=0.1,
                        help="GPTQ阻尼系数(0-1)")
    parser.add_argument("--sym-quant",
                        action="store_true",
                        help="GPTQ使用对称量化")
    parser.add_argument("--use-triton",
                        action="store_true",
                        help="GPTQ使用Triton加速（此选项在新版本中可能自动处理）")
    parser.add_argument("--batch-size",
                        type=int,
                        default=4,
                        help="量化批大小")

    # 数据参数
    parser.add_argument("--calib-samples",
                        type=int,
                        default=64,
                        help="校准样本数")
    parser.add_argument("--seq-len",
                        type=int,
                        default=1024,
                        help="序列长度")

    # 输出参数
    parser.add_argument("--output-path",
                        default="./quantized_qwen2",
                        help="量化模型输出路径")
    parser.add_argument("--test-prompt",
                        default=None,
                        help="测试使用的提示词")

    args = parser.parse_args()

    # 验证参数
    if not os.path.exists(args.model_path):
        logger.error(f"模型路径不存在: {args.model_path}")
        return

    if not os.path.exists(args.data_path):
        logger.error(f"数据路径不存在: {args.data_path}")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"使用设备: {device}")

    try:
        # 加载原始模型
        model = load_model(args.model_path, device)
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

        # 原始模型测试
        logger.info("\n=== 原始模型测试 ===")
        test_model(args.model_path, tokenizer, args.test_prompt)

        # 量化模型
        logger.info("\n=== 开始量化 ===")
        success = quantize_model(args, model, tokenizer)

        if success:
            # 量化后模型测试
            logger.info("\n=== 量化后模型测试 ===")
            test_model(args.output_path, tokenizer, args.test_prompt)

    except Exception as e:
        logger.error(f"主程序运行出错: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    main()


# import argparse
# import os
# import json
# import torch
# import logging
# from tqdm import tqdm
# from transformers import (
#     AutoModelForCausalLM,
#     AutoTokenizer,
#     BitsAndBytesConfig
# )
# from datasets import load_dataset
# import tempfile
#
# # 配置日志
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s - %(levelname)s - %(message)s"
# )
# logger = logging.getLogger(__name__)
#
# # 本地模型和数据路径
# MODEL_PATH = r"E:\reactflow_test\backend\model\Qwen\Qwen2___5-0___5B-Instruct"
# DATASET_PATH = r'E:\ml_data\medical_zh\train_zh_1000.jsonl'
#
#
# def load_model(model_path, device):
#     """加载原始模型"""
#     logger.info(f"加载模型: {model_path}")
#     try:
#         model = AutoModelForCausalLM.from_pretrained(
#             model_path,
#             device_map="auto",
#             torch_dtype=torch.float16,
#             trust_remote_code=True,
#             low_cpu_mem_usage=True
#         )
#         return model.eval()
#     except Exception as e:
#         logger.error(f"加载模型失败: {str(e)}")
#         raise
#
#
# def prepare_calib_data(tokenizer, data_path, num_samples=128, seq_len=128):
#     """准备校准数据，确保数据在CPU上"""
#     logger.info(f"准备校准数据，使用本地数据集: {data_path}")
#
#     with open(data_path, 'r', encoding='utf-8') as f:
#         lines = f.readlines()
#
#     samples = []
#     for i in tqdm(range(min(num_samples, len(lines)))):
#         try:
#             data = json.loads(lines[i])
#             # 合并instruction/input/output
#             text = f"{data.get('instruction', '')} {data.get('input', '')} {data.get('output', '')}"
#             text = text.strip()
#
#             if len(text) > 10:
#                 tokenized = tokenizer(
#                     text,
#                     truncation=True,
#                     max_length=seq_len,
#                     return_tensors="pt",
#                     padding="max_length"
#                 )
#                 # 确保所有张量在CPU上
#                 samples.append({
#                     "input_ids": tokenized["input_ids"][0].cpu(),
#                     "attention_mask": tokenized["attention_mask"][0].cpu()
#                 })
#
#         except Exception as e:
#             logger.warning(f"跳过第{i}行: {str(e)}")
#             continue
#
#     logger.info(f"成功加载 {len(samples)} 个校准样本")
#     return samples
#
#
# def quantize_model(args, model, tokenizer):
#     """执行模型量化，改进设备管理"""
#     quant_method = args.method.lower()
#     output_path = os.path.abspath(args.output_path)
#     os.makedirs(output_path, exist_ok=True)
#
#     try:
#         if quant_method in ["int8", "int4"]:
#             logger.info(f"使用bitsandbytes进行{quant_method.upper()}量化")
#
#             # 配置量化参数
#             bnb_config = BitsAndBytesConfig(
#                 load_in_4bit=quant_method == "int4",
#                 load_in_8bit=quant_method == "int8",
#                 llm_int8_threshold=6.0,
#                 llm_int8_has_fp16_weight=False,
#                 bnb_4bit_compute_dtype=torch.float16,
#                 bnb_4bit_quant_type="nf4",
#                 bnb_4bit_use_double_quant=True
#             )
#
#             # 释放原始模型内存
#             del model
#             torch.cuda.empty_cache()
#
#             # 重新加载量化后的模型
#             model = AutoModelForCausalLM.from_pretrained(
#                 args.model_path,
#                 device_map="auto",
#                 trust_remote_code=True,
#                 quantization_config=bnb_config
#             )
#
#             # 保存量化模型
#             model.save_pretrained(output_path)
#             tokenizer.save_pretrained(output_path)
#
#         elif quant_method == "gptq":
#             logger.info("使用transformers原生GPTQ量化（针对Qwen2优化）")
#
#             # 准备校准数据
#             calib_data = prepare_calib_data(
#                 tokenizer,
#                 args.data_path,
#                 args.calib_samples,
#                 args.seq_len
#             )
#
#             # 释放原始模型内存
#             del model
#             torch.cuda.empty_cache()
#
#             # 使用transformers原生的GPTQ量化方法
#             try:
#                 # 尝试导入最新的量化方法
#                 from transformers import GPTQConfig
#
#                 # 创建一个简单的数据集包装器
#                 class CalibrationDataset:
#                     def __init__(self, samples):
#                         self.samples = samples
#
#                     def __len__(self):
#                         return len(self.samples)
#
#                     def __getitem__(self, idx):
#                         return {
#                             "input_ids": self.samples[idx]["input_ids"],
#                             "attention_mask": self.samples[idx]["attention_mask"]
#                         }
#
#                 calibration_dataset = CalibrationDataset(calib_data)
#
#                 gptq_config = GPTQConfig(
#                     bits=4,
#                     group_size=args.group_size,
#                     dataset=calibration_dataset,
#                     desc_act=not args.sym_quant,
#                     damp_percent=args.damp_percent,
#                 )
#
#                 logger.info("使用transformers>=4.30.0的GPTQ量化方法")
#
#                 # 加载并量化模型
#                 model = AutoModelForCausalLM.from_pretrained(
#                     args.model_path,
#                     device_map="auto",
#                     torch_dtype=torch.float16,
#                     trust_remote_code=True,
#                     quantization_config=gptq_config
#                 )
#
#                 # 保存量化模型
#                 model.save_pretrained(output_path)
#                 tokenizer.save_pretrained(output_path)
#
#             except ImportError:
#                 # 旧版本的量化方法
#                 logger.info("使用旧版GPTQ量化方法")
#
#                 # 创建临时目录存储校准数据
#                 with tempfile.TemporaryDirectory() as tmpdir:
#                     # 保存校准数据为Hugging Face数据集格式
#                     from datasets import Dataset
#                     calib_dataset = Dataset.from_dict({
#                         "input_ids": [item["input_ids"] for item in calib_data],
#                         "attention_mask": [item["attention_mask"] for item in calib_data]
#                     })
#                     calib_dataset.save_to_disk(os.path.join(tmpdir, "calib_data"))
#
#                     # 使用shell命令调用transformers-cli进行量化
#                     logger.info("开始执行GPTQ量化命令行工具...")
#                     cmd = f"""
#                     transformers-cli quantize \
#                         --model_name_or_path {args.model_path} \
#                         --output_dir {output_path} \
#                         --quantization_method gptq \
#                         --bits 4 \
#                         --dataset {os.path.join(tmpdir, "calib_data")} \
#                         --trust_remote_code \
#                         --group_size {args.group_size} \
#                         --symmetric {args.sym_quant}
#                     """
#
#                     import subprocess
#                     result = subprocess.run(
#                         cmd,
#                         shell=True,
#                         capture_output=True,
#                         text=True
#                     )
#
#                     if result.returncode != 0:
#                         logger.error("量化命令执行失败")
#                         logger.error(result.stderr)
#                         raise RuntimeError("GPTQ量化失败")
#                     else:
#                         logger.info("量化命令执行成功")
#                         logger.info(result.stdout)
#
#         logger.info(f"量化完成，模型已保存到: {output_path}")
#         return True
#
#     except Exception as e:
#         logger.error(f"量化过程中出错: {str(e)}")
#         import traceback
#         logger.error(traceback.format_exc())
#         return False
#
#
# def test_model(model_path, tokenizer, prompt=None):
#     """测试量化模型"""
#     logger.info("测试模型生成...")
#
#     try:
#         # 检查是否为GPTQ量化模型
#         is_gptq_quantized = os.path.exists(os.path.join(model_path, "gptq_config.json"))
#         # 检查是否为bitsandbytes量化模型
#         is_bnb_quantized = os.path.exists(os.path.join(model_path, "quantization_config.json"))
#
#         if is_gptq_quantized:
#             # 加载GPTQ量化模型
#             logger.info("加载GPTQ量化模型")
#             model = AutoModelForCausalLM.from_pretrained(
#                 model_path,
#                 device_map="auto",
#                 trust_remote_code=True,
#                 low_cpu_mem_usage=True
#             )
#         elif is_bnb_quantized:
#             # 加载bitsandbytes量化模型
#             logger.info("加载bitsandbytes量化模型")
#             bnb_config = BitsAndBytesConfig.from_pretrained(model_path)
#             model = AutoModelForCausalLM.from_pretrained(
#                 model_path,
#                 device_map="auto",
#                 trust_remote_code=True,
#                 quantization_config=bnb_config
#             )
#         else:
#             # 加载原始模型
#             logger.info("加载原始模型")
#             model = AutoModelForCausalLM.from_pretrained(
#                 model_path,
#                 device_map="auto",
#                 trust_remote_code=True
#             )
#
#         test_prompt = prompt if prompt else "请解释一下糖尿病的主要症状有哪些？"
#         inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)
#
#         with torch.no_grad():
#             outputs = model.generate(
#                 **inputs,
#                 max_length=200,
#                 temperature=0.7,
#                 top_p=0.9,
#                 do_sample=True,
#                 pad_token_id=tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.pad_token_id
#             )
#
#         result = tokenizer.decode(outputs[0], skip_special_tokens=True)
#         logger.info(f"生成结果:\n{result}")
#         return result
#     except Exception as e:
#         logger.error(f"模型测试失败: {str(e)}")
#         import traceback
#         logger.error(traceback.format_exc())
#         raise
#
#
# def main():
#     parser = argparse.ArgumentParser(
#         description="大模型量化工具 (支持int8/int4/gptq量化，适配Qwen2模型)",
#         formatter_class=argparse.ArgumentDefaultsHelpFormatter
#     )
#
#     # 模型参数
#     parser.add_argument("--model-path", default=MODEL_PATH, help="模型路径")
#     parser.add_argument("--data-path", default=DATASET_PATH, help="校准数据集路径")
#
#     # 量化参数
#     parser.add_argument("--method",
#                         choices=["int8", "int4", "gptq"],
#                         default="gptq",
#                         help="量化方法")
#     parser.add_argument("--group-size",
#                         type=int,
#                         default=128,
#                         help="GPTQ分组大小")
#     parser.add_argument("--damp-percent",
#                         type=float,
#                         default=0.1,
#                         help="GPTQ阻尼系数(0-1)")
#     parser.add_argument("--sym-quant",
#                         action="store_true",
#                         help="GPTQ使用对称量化")
#     parser.add_argument("--use-triton",
#                         action="store_true",
#                         help="GPTQ使用Triton加速（此选项在新版本中可能自动处理）")
#     parser.add_argument("--batch-size",
#                         type=int,
#                         default=4,
#                         help="量化批大小")
#
#     # 数据参数
#     parser.add_argument("--calib-samples",
#                         type=int,
#                         default=64,
#                         help="校准样本数")
#     parser.add_argument("--seq-len",
#                         type=int,
#                         default=1024,
#                         help="序列长度")
#
#     # 输出参数
#     parser.add_argument("--output-path",
#                         default="./quantized_qwen2",
#                         help="量化模型输出路径")
#     parser.add_argument("--test-prompt",
#                         default=None,
#                         help="测试使用的提示词")
#
#     args = parser.parse_args()
#
#     # 验证参数
#     if not os.path.exists(args.model_path):
#         logger.error(f"模型路径不存在: {args.model_path}")
#         return
#
#     if not os.path.exists(args.data_path):
#         logger.error(f"数据路径不存在: {args.data_path}")
#         return
#
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     logger.info(f"使用设备: {device}")
#
#     try:
#         # 加载原始模型
#         model = load_model(args.model_path, device)
#         tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
#
#         # 原始模型测试
#         logger.info("\n=== 原始模型测试 ===")
#         test_model(args.model_path, tokenizer, args.test_prompt)
#
#         # 量化模型
#         logger.info("\n=== 开始量化 ===")
#         success = quantize_model(args, model, tokenizer)
#
#         if success:
#             # 量化后模型测试
#             logger.info("\n=== 量化后模型测试 ===")
#             test_model(args.output_path, tokenizer, args.test_prompt)
#
#     except Exception as e:
#         logger.error(f"主程序运行出错: {str(e)}")
#         import traceback
#         logger.error(traceback.format_exc())
#
#
# if __name__ == "__main__":
#     main()


# import argparse
# import os
# import json
# import torch
# import logging
# from tqdm import tqdm
# from transformers import (
#     AutoModelForCausalLM,
#     AutoTokenizer,
#     BitsAndBytesConfig
# )
# from datasets import load_dataset
# import tempfile
#
# # 配置日志
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s - %(levelname)s - %(message)s"
# )
# logger = logging.getLogger(__name__)
#
# # 本地模型和数据路径
# MODEL_PATH = r"E:\reactflow_test\backend\model\Qwen\Qwen2___5-0___5B-Instruct"
# DATASET_PATH = r'E:\ml_data\medical_zh\train_zh_1000.jsonl'
#
#
# def load_model(model_path, device):
#     """加载原始模型"""
#     logger.info(f"加载模型: {model_path}")
#     try:
#         model = AutoModelForCausalLM.from_pretrained(
#             model_path,
#             device_map="auto",
#             torch_dtype=torch.float16,
#             trust_remote_code=True,
#             low_cpu_mem_usage=True
#         )
#         return model.eval()
#     except Exception as e:
#         logger.error(f"加载模型失败: {str(e)}")
#         raise
#
#
# def prepare_calib_data(tokenizer, data_path, num_samples=128, seq_len=128):
#     """准备校准数据"""
#     logger.info(f"准备校准数据，使用本地数据集: {data_path}")
#
#     with open(data_path, 'r', encoding='utf-8') as f:
#         lines = f.readlines()
#
#     samples = []
#     for i in tqdm(range(min(num_samples, len(lines)))):
#         try:
#             data = json.loads(lines[i])
#             # 合并instruction/input/output
#             text = f"{data.get('instruction', '')} {data.get('input', '')} {data.get('output', '')}"
#             text = text.strip()
#
#             if len(text) > 10:
#                 tokenized = tokenizer(
#                     text,
#                     truncation=True,
#                     max_length=seq_len,
#                     return_tensors="pt",
#                     padding="max_length"
#                 )
#                 # 确保数据在CPU上
#                 samples.append({
#                     "input_ids": tokenized["input_ids"][0].cpu(),
#                     "attention_mask": tokenized["attention_mask"][0].cpu()
#                 })
#
#         except Exception as e:
#             logger.warning(f"跳过第{i}行: {str(e)}")
#             continue
#
#     logger.info(f"成功加载 {len(samples)} 个校准样本")
#     return samples
#
#
# def quantize_model(args, model, tokenizer):
#     """执行模型量化，使用更稳定的方法"""
#     quant_method = args.method.lower()
#     output_path = os.path.abspath(args.output_path)
#     os.makedirs(output_path, exist_ok=True)
#
#     try:
#         if quant_method in ["int8", "int4"]:
#             logger.info(f"使用bitsandbytes进行{quant_method.upper()}量化")
#
#             # 配置量化参数
#             bnb_config = BitsAndBytesConfig(
#                 load_in_4bit=quant_method == "int4",
#                 load_in_8bit=quant_method == "int8",
#                 llm_int8_threshold=6.0,
#                 llm_int8_has_fp16_weight=False,
#                 bnb_4bit_compute_dtype=torch.float16,
#                 bnb_4bit_quant_type="nf4",
#                 bnb_4bit_use_double_quant=True
#             )
#
#             # 释放原始模型内存
#             del model
#             torch.cuda.empty_cache()
#
#             # 重新加载量化后的模型
#             model = AutoModelForCausalLM.from_pretrained(
#                 args.model_path,
#                 device_map="auto",
#                 trust_remote_code=True,
#                 quantization_config=bnb_config
#             )
#
#             # 保存量化模型
#             model.save_pretrained(output_path)
#             tokenizer.save_pretrained(output_path)
#
#         elif quant_method == "gptq":
#             logger.info("使用transformers原生GPTQ量化（针对Qwen2优化）")
#
#             # 准备校准数据
#             calib_data = prepare_calib_data(
#                 tokenizer,
#                 args.data_path,
#                 args.calib_samples,
#                 args.seq_len
#             )
#
#             # 确保数据在正确的设备上
#             if torch.cuda.is_available():
#                 for i in range(len(calib_data)):
#                     calib_data[i]["input_ids"] = calib_data[i]["input_ids"].to("cuda")
#                     calib_data[i]["attention_mask"] = calib_data[i]["attention_mask"].to("cuda")
#
#             # 释放原始模型内存
#             del model
#             torch.cuda.empty_cache()
#
#             # 使用transformers原生的GPTQ量化方法
#             try:
#                 # 尝试导入最新的量化方法
#                 from transformers import GPTQConfig
#
#                 gptq_config = GPTQConfig(
#                     bits=4,
#                     group_size=args.group_size,
#                     dataset=calib_data,
#                     desc_act=not args.sym_quant,
#                     damp_percent=args.damp_percent,
#                     tokenizer=tokenizer
#                 )
#
#                 logger.info("使用transformers>=4.30.0的GPTQ量化方法")
#
#                 # 加载并量化模型
#                 model = AutoModelForCausalLM.from_pretrained(
#                     args.model_path,
#                     device_map="auto",
#                     torch_dtype=torch.float16,
#                     trust_remote_code=True,
#                     quantization_config=gptq_config
#                 )
#
#                 # 保存量化模型
#                 model.save_pretrained(output_path)
#                 tokenizer.save_pretrained(output_path)
#
#             except ImportError:
#                 # 旧版本的量化方法
#                 logger.info("使用旧版GPTQ量化方法")
#
#                 # 创建临时目录存储校准数据
#                 with tempfile.TemporaryDirectory() as tmpdir:
#                     # 保存校准数据为Hugging Face数据集格式
#                     from datasets import Dataset
#                     calib_dataset = Dataset.from_dict({
#                         "input_ids": [item["input_ids"] for item in calib_data],
#                         "attention_mask": [item["attention_mask"] for item in calib_data]
#                     })
#                     calib_dataset.save_to_disk(os.path.join(tmpdir, "calib_data"))
#
#                     # 使用shell命令调用transformers-cli进行量化
#                     logger.info("开始执行GPTQ量化命令行工具...")
#                     cmd = f"""
#                     transformers-cli quantize \
#                         --model_name_or_path {args.model_path} \
#                         --output_dir {output_path} \
#                         --quantization_method gptq \
#                         --bits 4 \
#                         --dataset {os.path.join(tmpdir, "calib_data")} \
#                         --trust_remote_code \
#                         --group_size {args.group_size} \
#                         --symmetric {args.sym_quant}
#                     """
#
#                     import subprocess
#                     result = subprocess.run(
#                         cmd,
#                         shell=True,
#                         capture_output=True,
#                         text=True
#                     )
#
#                     if result.returncode != 0:
#                         logger.error("量化命令执行失败")
#                         logger.error(result.stderr)
#                         raise RuntimeError("GPTQ量化失败")
#                     else:
#                         logger.info("量化命令执行成功")
#                         logger.info(result.stdout)
#
#         logger.info(f"量化完成，模型已保存到: {output_path}")
#         return True
#
#     except Exception as e:
#         logger.error(f"量化过程中出错: {str(e)}")
#         import traceback
#         logger.error(traceback.format_exc())
#         return False
#
#
# def test_model(model_path, tokenizer, prompt=None):
#     """测试量化模型"""
#     logger.info("测试模型生成...")
#
#     try:
#         # 检查是否为GPTQ量化模型
#         is_gptq_quantized = os.path.exists(os.path.join(model_path, "gptq_config.json"))
#         # 检查是否为bitsandbytes量化模型
#         is_bnb_quantized = os.path.exists(os.path.join(model_path, "quantization_config.json"))
#
#         if is_gptq_quantized:
#             # 加载GPTQ量化模型
#             logger.info("加载GPTQ量化模型")
#             model = AutoModelForCausalLM.from_pretrained(
#                 model_path,
#                 device_map="auto",
#                 trust_remote_code=True,
#                 low_cpu_mem_usage=True
#             )
#         elif is_bnb_quantized:
#             # 加载bitsandbytes量化模型
#             logger.info("加载bitsandbytes量化模型")
#             bnb_config = BitsAndBytesConfig.from_pretrained(model_path)
#             model = AutoModelForCausalLM.from_pretrained(
#                 model_path,
#                 device_map="auto",
#                 trust_remote_code=True,
#                 quantization_config=bnb_config
#             )
#         else:
#             # 加载原始模型
#             logger.info("加载原始模型")
#             model = AutoModelForCausalLM.from_pretrained(
#                 model_path,
#                 device_map="auto",
#                 trust_remote_code=True
#             )
#
#         test_prompt = prompt if prompt else "请解释一下糖尿病的主要症状有哪些？"
#         inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)
#
#         with torch.no_grad():
#             outputs = model.generate(
#                 **inputs,
#                 max_length=200,
#                 temperature=0.7,
#                 top_p=0.9,
#                 do_sample=True,
#                 pad_token_id=tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.pad_token_id
#             )
#
#         result = tokenizer.decode(outputs[0], skip_special_tokens=True)
#         logger.info(f"生成结果:\n{result}")
#         return result
#     except Exception as e:
#         logger.error(f"模型测试失败: {str(e)}")
#         import traceback
#         logger.error(traceback.format_exc())
#         raise
#
#
# def main():
#     parser = argparse.ArgumentParser(
#         description="大模型量化工具 (支持int8/int4/gptq量化，适配Qwen2模型)",
#         formatter_class=argparse.ArgumentDefaultsHelpFormatter
#     )
#
#     # 模型参数
#     parser.add_argument("--model-path", default=MODEL_PATH, help="模型路径")
#     parser.add_argument("--data-path", default=DATASET_PATH, help="校准数据集路径")
#
#     # 量化参数
#     parser.add_argument("--method",
#                         choices=["int8", "int4", "gptq"],
#                         default="gptq",
#                         help="量化方法")
#     parser.add_argument("--group-size",
#                         type=int,
#                         default=128,
#                         help="GPTQ分组大小")
#     parser.add_argument("--damp-percent",
#                         type=float,
#                         default=0.1,
#                         help="GPTQ阻尼系数(0-1)")
#     parser.add_argument("--sym-quant",
#                         action="store_true",
#                         help="GPTQ使用对称量化")
#     parser.add_argument("--use-triton",
#                         action="store_true",
#                         help="GPTQ使用Triton加速（此选项在新版本中可能自动处理）")
#     parser.add_argument("--batch-size",
#                         type=int,
#                         default=4,
#                         help="量化批大小")
#
#     # 数据参数
#     parser.add_argument("--calib-samples",
#                         type=int,
#                         default=64,
#                         help="校准样本数")
#     parser.add_argument("--seq-len",
#                         type=int,
#                         default=1024,
#                         help="序列长度")
#
#     # 输出参数
#     parser.add_argument("--output-path",
#                         default="./quantized_qwen2",
#                         help="量化模型输出路径")
#     parser.add_argument("--test-prompt",
#                         default=None,
#                         help="测试使用的提示词")
#
#     args = parser.parse_args()
#
#     # 验证参数
#     if not os.path.exists(args.model_path):
#         logger.error(f"模型路径不存在: {args.model_path}")
#         return
#
#     if not os.path.exists(args.data_path):
#         logger.error(f"数据路径不存在: {args.data_path}")
#         return
#
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     logger.info(f"使用设备: {device}")
#
#     try:
#         # 加载原始模型
#         model = load_model(args.model_path, device)
#         tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
#
#         # 原始模型测试
#         logger.info("\n=== 原始模型测试 ===")
#         test_model(args.model_path, tokenizer, args.test_prompt)
#
#         # 量化模型
#         logger.info("\n=== 开始量化 ===")
#         success = quantize_model(args, model, tokenizer)
#
#         if success:
#             # 量化后模型测试
#             logger.info("\n=== 量化后模型测试 ===")
#             test_model(args.output_path, tokenizer, args.test_prompt)
#
#     except Exception as e:
#         logger.error(f"主程序运行出错: {str(e)}")
#         import traceback
#         logger.error(traceback.format_exc())
#
#
# if __name__ == "__main__":
#     main()


# import argparse
# import os
# import json
# import torch
# import logging
# from tqdm import tqdm
# from transformers import (
#     AutoModelForCausalLM,
#     AutoTokenizer,
#     BitsAndBytesConfig
# )
# from datasets import load_dataset
# from packaging import version
# import optimum
#
# # 配置日志
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s - %(levelname)s - %(message)s"
# )
# logger = logging.getLogger(__name__)
#
# # 检查optimum版本
# optimum_version = version.parse(optimum.__version__)
# logger.info(f"检测到optimum版本: {optimum.__version__}")
#
# # 本地模型和数据路径
# MODEL_PATH = r"E:\reactflow_test\backend\model\Qwen\Qwen2___5-0___5B-Instruct"
# DATASET_PATH = r'E:\ml_data\medical_zh\train_zh_1000.jsonl'
#
#
# def load_model(model_path, device):
#     """加载原始模型"""
#     logger.info(f"加载模型: {model_path}")
#     try:
#         model = AutoModelForCausalLM.from_pretrained(
#             model_path,
#             device_map="auto",
#             torch_dtype=torch.float16,
#             trust_remote_code=True,
#             low_cpu_mem_usage=True
#         )
#         return model.eval()
#     except Exception as e:
#         logger.error(f"加载模型失败: {str(e)}")
#         raise
#
#
# def prepare_calib_data(tokenizer, data_path, num_samples=128, seq_len=128):
#     """准备校准数据（适配optimum格式）"""
#     logger.info(f"准备校准数据，使用本地数据集: {data_path}")
#
#     with open(data_path, 'r', encoding='utf-8') as f:
#         lines = f.readlines()
#
#     samples = []
#     for i in tqdm(range(min(num_samples, len(lines)))):
#         try:
#             data = json.loads(lines[i])
#             # 合并instruction/input/output
#             text = f"{data.get('instruction', '')} {data.get('input', '')} {data.get('output', '')}"
#             text = text.strip()
#
#             if len(text) > 10:
#                 tokenized = tokenizer(
#                     text,
#                     truncation=True,
#                     max_length=seq_len,
#                     return_tensors="pt",
#                     padding="max_length"
#                 )
#                 # 转换为optimum期望的格式：只需要input_ids
#                 samples.append({
#                     "input_ids": tokenized["input_ids"][0]  # 保留batch维度
#                 })
#
#         except Exception as e:
#             logger.warning(f"跳过第{i}行: {str(e)}")
#             continue
#
#     logger.info(f"成功加载 {len(samples)} 个校准样本")
#     return samples  # 返回字典列表
#
#
# def quantize_model(args, model, tokenizer):
#     """使用optimum.gptq执行量化，适配Qwen2模型并兼容不同版本"""
#     quant_method = args.method.lower()
#     output_path = os.path.abspath(args.output_path)
#     os.makedirs(output_path, exist_ok=True)
#
#     try:
#         # 准备校准数据
#         calib_data = None
#         if quant_method in ["gptq", "int8", "int4"]:
#             calib_data = prepare_calib_data(
#                 tokenizer,
#                 args.data_path,
#                 args.calib_samples,
#                 args.seq_len
#             )
#
#         if quant_method in ["int8", "int4"]:
#             logger.info(f"使用bitsandbytes进行{quant_method.upper()}量化")
#             # 保持原有int8/int4量化逻辑不变
#             bnb_config = BitsAndBytesConfig(
#                 load_in_4bit=quant_method == "int4",
#                 load_in_8bit=quant_method == "int8",
#                 llm_int8_threshold=6.0,
#                 llm_int8_has_fp16_weight=False,
#                 bnb_4bit_compute_dtype=torch.float16,
#                 bnb_4bit_quant_type="nf4",
#                 bnb_4bit_use_double_quant=True
#             )
#
#             # 释放原始模型内存
#             del model
#             torch.cuda.empty_cache()
#
#             # 重新加载量化后的模型
#             model = AutoModelForCausalLM.from_pretrained(
#                 args.model_path,
#                 device_map="auto",
#                 trust_remote_code=True,
#                 quantization_config=bnb_config
#             )
#             model.save_pretrained(output_path)
#             tokenizer.save_pretrained(output_path)
#
#         elif quant_method == "gptq":
#             logger.info("使用optimum.gptq进行4bit量化（针对Qwen2模型优化）")
#
#             # 确保数据在正确的设备上
#             if torch.cuda.is_available():
#                 for i in range(len(calib_data)):
#                     calib_data[i]["input_ids"] = calib_data[i]["input_ids"].to("cuda")
#
#             # 根据optimum版本选择正确的导入方式
#             if optimum_version >= version.parse("1.10.0"):
#                 from optimum.gptq import GPTQConfig, GPTQQuantizer
#             else:
#                 from optimum.gptq import GPTQQuantizer
#                 from optimum.gptq.quantization_config import GPTQConfig
#
#             # 定义GPTQ配置
#             gptq_config = GPTQConfig(
#                 bits=4,
#                 group_size=args.group_size,
#                 desc_act=not args.sym_quant,
#                 damp_percent=args.damp_percent,
#                 use_triton=args.use_triton
#             )
#
#             # 释放原始模型内存
#             del model
#             torch.cuda.empty_cache()
#
#             # 初始化量化器
#             quantizer = GPTQQuantizer.from_pretrained(
#                 args.model_path,
#                 gptq_config,
#                 trust_remote_code=True,
#                 device_map="auto"
#             )
#
#             # 执行量化
#             logger.info("开始执行GPTQ量化...")
#             quantizer.quantize(
#                 calib_data,
#                 batch_size=args.batch_size,
#                 max_seq_len=args.seq_len
#             )
#
#             # 保存量化模型
#             quantizer.save_quantized(output_path, use_safetensors=True)
#             tokenizer.save_pretrained(output_path)
#
#         logger.info(f"量化完成，模型已保存到: {output_path}")
#         return True
#
#     except Exception as e:
#         logger.error(f"量化过程中出错: {str(e)}")
#         import traceback
#         logger.error(traceback.format_exc())
#         return False
#
#
# def test_model(model_path, tokenizer, prompt=None):
#     """测试量化模型"""
#     logger.info("测试模型生成...")
#
#     try:
#         # 检查是否为量化模型
#         is_quantized = os.path.exists(os.path.join(model_path, "gptq_config.json"))
#
#         if is_quantized:
#             # 加载量化模型
#             model = AutoModelForCausalLM.from_quantized(
#                 model_path,
#                 device_map="auto",
#                 trust_remote_code=True,
#                 use_safetensors=True
#             )
#         else:
#             # 加载原始模型
#             model = AutoModelForCausalLM.from_pretrained(
#                 model_path,
#                 device_map="auto",
#                 trust_remote_code=True
#             )
#
#         test_prompt = prompt if prompt else "请解释一下糖尿病的主要症状有哪些？"
#         inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)
#
#         with torch.no_grad():
#             outputs = model.generate(
#                 **inputs,
#                 max_length=200,
#                 temperature=0.7,
#                 top_p=0.9,
#                 do_sample=True,
#                 pad_token_id=tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.pad_token_id
#             )
#
#         result = tokenizer.decode(outputs[0], skip_special_tokens=True)
#         logger.info(f"生成结果:\n{result}")
#         return result
#     except Exception as e:
#         logger.error(f"模型测试失败: {str(e)}")
#         import traceback
#         logger.error(traceback.format_exc())
#         raise
#
#
# def main():
#     parser = argparse.ArgumentParser(
#         description="大模型量化工具 (支持int8/int4/gptq量化，适配Qwen2模型)",
#         formatter_class=argparse.ArgumentDefaultsHelpFormatter
#     )
#
#     # 模型参数
#     parser.add_argument("--model-path", default=MODEL_PATH, help="模型路径")
#     parser.add_argument("--data-path", default=DATASET_PATH, help="校准数据集路径")
#
#     # 量化参数
#     parser.add_argument("--method",
#                         choices=["int8", "int4", "gptq"],
#                         default="gptq",
#                         help="量化方法")
#     parser.add_argument("--group-size",
#                         type=int,
#                         default=128,
#                         help="GPTQ分组大小")
#     parser.add_argument("--damp-percent",
#                         type=float,
#                         default=0.1,
#                         help="GPTQ阻尼系数(0-1)")
#     parser.add_argument("--sym-quant",
#                         action="store_true",
#                         help="GPTQ使用对称量化")
#     parser.add_argument("--use-triton",
#                         action="store_true",
#                         help="GPTQ使用Triton加速")
#     parser.add_argument("--batch-size",
#                         type=int,
#                         default=4,
#                         help="量化批大小")
#
#     # 数据参数
#     parser.add_argument("--calib-samples",
#                         type=int,
#                         default=64,
#                         help="校准样本数")
#     parser.add_argument("--seq-len",
#                         type=int,
#                         default=1024,
#                         help="序列长度")
#
#     # 输出参数
#     parser.add_argument("--output-path",
#                         default="./quantized_qwen2",
#                         help="量化模型输出路径")
#     parser.add_argument("--test-prompt",
#                         default=None,
#                         help="测试使用的提示词")
#
#     args = parser.parse_args()
#
#     # 验证参数
#     if not os.path.exists(args.model_path):
#         logger.error(f"模型路径不存在: {args.model_path}")
#         return
#
#     if not os.path.exists(args.data_path):
#         logger.error(f"数据路径不存在: {args.data_path}")
#         return
#
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     logger.info(f"使用设备: {device}")
#
#     try:
#         # 加载原始模型
#         model = load_model(args.model_path, device)
#         tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
#
#         # 原始模型测试
#         logger.info("\n=== 原始模型测试 ===")
#         test_model(args.model_path, tokenizer, args.test_prompt)
#
#         # 量化模型
#         logger.info("\n=== 开始量化 ===")
#         success = quantize_model(args, model, tokenizer)
#
#         if success:
#             # 量化后模型测试
#             logger.info("\n=== 量化后模型测试 ===")
#             test_model(args.output_path, tokenizer, args.test_prompt)
#
#     except Exception as e:
#         logger.error(f"主程序运行出错: {str(e)}")
#         import traceback
#         logger.error(traceback.format_exc())
#
#
# if __name__ == "__main__":
#     main()
#
# # import argparse
# # import os
# # import json
# # import torch
# # import logging
# # from tqdm import tqdm
# # from transformers import (
# #     AutoModelForCausalLM,
# #     AutoTokenizer,
# #     BitsAndBytesConfig
# # )
# # from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
# # from datasets import load_dataset
# # from packaging import version
# #
# # # 配置日志
# # logging.basicConfig(
# #     level=logging.INFO,
# #     format="%(asctime)s - %(levelname)s - %(message)s"
# # )
# # logger = logging.getLogger(__name__)
# #
# # # 本地模型和数据路径
# # MODEL_PATH = r"E:\reactflow_test\backend\model\Qwen\Qwen2___5-0___5B-Instruct"
# # DATASET_PATH = r'E:\ml_data\medical_zh\train_zh_1000.jsonl'
# #
# #
# # def load_model(model_path, device):
# #     """加载原始模型"""
# #     logger.info(f"加载模型: {model_path}")
# #     try:
# #         model = AutoModelForCausalLM.from_pretrained(
# #             model_path,
# #             device_map="auto",
# #             torch_dtype=torch.float16,
# #             trust_remote_code=True,
# #             low_cpu_mem_usage=True
# #         )
# #         return model.eval()
# #     except Exception as e:
# #         logger.error(f"加载模型失败: {str(e)}")
# #         raise
# #
# #
# # def prepare_calib_data(tokenizer, data_path, num_samples=128, seq_len=128):
# #     """准备校准数据（包含attention_mask）"""
# #     logger.info(f"准备校准数据，使用本地数据集: {data_path}")
# #
# #
# #     with open(data_path, 'r', encoding='utf-8') as f:
# #         lines = f.readlines()
# #
# #     samples = []
# #     for i in tqdm(range(min(num_samples, len(lines)))):
# #         try:
# #             data = json.loads(lines[i])
# #             # 合并instruction/input/output
# #             text = f"{data.get('instruction', '')} {data.get('input', '')} {data.get('output', '')}"
# #             text = text.strip()
# #
# #             if len(text) > 10:
# #                 tokenized = tokenizer(
# #                     text,
# #                     truncation=True,
# #                     max_length=seq_len,
# #                     return_tensors="pt",
# #                     padding="max_length"
# #                 )
# #                 # 转换为AutoGPTQ期望的格式：包含input_ids和attention_mask
# #                 samples.append({
# #                     "input_ids": tokenized["input_ids"][0],  # 保留batch维度
# #                     "attention_mask": tokenized["attention_mask"][0]  # 添加attention_mask
# #                 })
# #
# #         except Exception as e:
# #             logger.warning(f"跳过第{i}行: {str(e)}")
# #             continue
# #
# #     logger.info(f"成功加载 {len(samples)} 个校准样本")
# #     return samples  # 返回字典列表
# #
# #
# # def quantize_model(args, model, tokenizer):
# #     """执行量化（支持int8/int4/gptq，针对Qwen2特殊处理）"""
# #     quant_method = args.method.lower()
# #     output_path = os.path.abspath(args.output_path)
# #     os.makedirs(output_path, exist_ok=True)
# #
# #     try:
# #         # 准备校准数据
# #         calib_data = None
# #         if quant_method in ["gptq", "int8", "int4"]:
# #             calib_data = prepare_calib_data(
# #                 tokenizer,
# #                 args.data_path,
# #                 args.calib_samples,
# #                 args.seq_len
# #             )
# #
# #         if quant_method in ["int8", "int4"]:
# #             logger.info(f"使用bitsandbytes进行{quant_method.upper()}量化")
# #             # ... [保持原有int8/int4量化代码不变] ...
# #
# #         elif quant_method == "gptq":
# #             logger.info("使用GPTQ进行4bit量化（针对Qwen2模型特殊处理）")
# #
# #             # 确保数据在GPU上（如果可用）
# #             if torch.cuda.is_available():
# #                 for i in range(len(calib_data)):
# #                     calib_data[i]["input_ids"] = calib_data[i]["input_ids"].to("cuda")
# #                     calib_data[i]["attention_mask"] = calib_data[i]["attention_mask"].to("cuda")
# #
# #             # 定义GPTQ配置
# #             gptq_config = BaseQuantizeConfig(
# #                 bits=4,
# #                 group_size=args.group_size,
# #                 desc_act=not args.sym_quant,
# #                 damp_percent=args.damp_percent
# #             )
# #
# #             del model
# #             torch.cuda.empty_cache()
# #
# #             # 特殊处理：完全绕过attention_type检查
# #             from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
# #             original_forward = Qwen2DecoderLayer.forward
# #
# #             def patched_forward(self, hidden_states, attention_mask=None, *args, **kwargs):
# #                 # 完全绕过attention_type检查，直接使用causal attention mask
# #                 if attention_mask is not None:
# #                     attention_mask = attention_mask.to(hidden_states.device)
# #                     attention_mask = attention_mask.bool()
# #                     attention_mask = torch.where(attention_mask, 0.0, torch.finfo(hidden_states.dtype).min)
# #                     attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
# #
# #                 return original_forward(
# #                     self,
# #                     hidden_states,
# #                     attention_mask=attention_mask,
# #                     *args,
# #                     **kwargs
# #                 )
# #
# #             # 应用补丁
# #             Qwen2DecoderLayer.forward = patched_forward
# #
# #             try:
# #                 # 加载模型并量化
# #                 gptq_model = AutoGPTQForCausalLM.from_pretrained(
# #                     args.model_path,
# #                     quantize_config=gptq_config,
# #                     trust_remote_code=True,
# #                     device_map="auto"
# #                 )
# #
# #                 # 量化前确保模型在eval模式
# #                 gptq_model.eval()
# #
# #                 gptq_model.quantize(
# #                     calib_data,
# #                     use_triton=args.use_triton,
# #                     batch_size=args.batch_size
# #                 )
# #
# #                 gptq_model.save_quantized(
# #                     output_path,
# #                     use_safetensors=True
# #                 )
# #                 tokenizer.save_pretrained(output_path)
# #             finally:
# #                 # 恢复原始forward方法
# #                 Qwen2DecoderLayer.forward = original_forward
# #
# #         logger.info(f"量化完成，模型已保存到: {output_path}")
# #         return True
# #
# #     except Exception as e:
# #         logger.error(f"量化过程中出错: {str(e)}")
# #         import traceback
# #         logger.error(traceback.format_exc())
# #         return False
# #
# # def test_model(model_path, tokenizer, prompt=None):
# #     """测试量化模型"""
# #     logger.info("测试模型生成...")
# #
# #     try:
# #         if os.path.exists(os.path.join(model_path, "quantize_config.json")):
# #             model = AutoGPTQForCausalLM.from_quantized(
# #                 model_path,
# #                 device="cuda:0",
# #                 trust_remote_code=True,
# #                 use_safetensors=True
# #             )
# #         else:
# #             model = AutoModelForCausalLM.from_pretrained(
# #                 model_path,
# #                 device_map="auto",
# #                 trust_remote_code=True
# #             )
# #
# #         test_prompt = prompt if prompt else "请解释一下糖尿病的主要症状有哪些？"
# #         inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)
# #
# #         with torch.no_grad():
# #             outputs = model.generate(
# #                 **inputs,
# #                 max_length=200,
# #                 temperature=0.7,
# #                 top_p=0.9,
# #                 do_sample=True,
# #                 pad_token_id=tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.pad_token_id
# #             )
# #
# #         result = tokenizer.decode(outputs[0], skip_special_tokens=True)
# #         logger.info(f"生成结果:\n{result}")
# #         return result
# #     except Exception as e:
# #         logger.error(f"模型测试失败: {str(e)}")
# #         import traceback
# #         logger.error(traceback.format_exc())
# #         raise
# #
# #
# # def main():
# #     parser = argparse.ArgumentParser(
# #         description="大模型量化工具 (支持int8/int4/gptq量化，适配Qwen2模型)",
# #         formatter_class=argparse.ArgumentDefaultsHelpFormatter
# #     )
# #
# #     # 模型参数
# #     parser.add_argument("--model-path", default=MODEL_PATH, help="模型路径")
# #     parser.add_argument("--data-path", default=DATASET_PATH, help="校准数据集路径")
# #
# #     # 量化参数
# #     parser.add_argument("--method",
# #                         choices=["int8", "int4", "gptq"],
# #                         default="gptq",
# #                         help="量化方法")
# #     parser.add_argument("--group-size",
# #                         type=int,
# #                         default=128,
# #                         help="GPTQ分组大小")
# #     parser.add_argument("--damp-percent",
# #                         type=float,
# #                         default=0.1,
# #                         help="GPTQ阻尼系数(0-1)")
# #     parser.add_argument("--sym-quant",
# #                         action="store_true",
# #                         help="GPTQ使用对称量化")
# #     parser.add_argument("--use-triton",
# #                         action="store_true",
# #                         help="GPTQ使用Triton加速")
# #     parser.add_argument("--batch-size",
# #                         type=int,
# #                         default=4,  # 调小默认batch_size减少内存压力
# #                         help="量化批大小")
# #
# #     # 数据参数
# #     parser.add_argument("--calib-samples",
# #                         type=int,
# #                         default=64,  # 调小默认校准样本数
# #                         help="校准样本数")
# #     parser.add_argument("--seq-len",
# #                         type=int,
# #                         default=1024,  # 调小默认序列长度
# #                         help="序列长度")
# #
# #     # 输出参数
# #     parser.add_argument("--output-path",
# #                         default="./quantized_qwen2",
# #                         help="量化模型输出路径")
# #     parser.add_argument("--test-prompt",
# #                         default=None,
# #                         help="测试使用的提示词")
# #
# #     args = parser.parse_args()
# #
# #     # 验证参数
# #     if not os.path.exists(args.model_path):
# #         logger.error(f"模型路径不存在: {args.model_path}")
# #         return
# #
# #     if not os.path.exists(args.data_path):
# #         logger.error(f"数据路径不存在: {args.data_path}")
# #         return
# #
# #     device = "cuda" if torch.cuda.is_available() else "cpu"
# #     logger.info(f"使用设备: {device}")
# #
# #     try:
# #         # 加载原始模型
# #         model = load_model(args.model_path, device)
# #         tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
# #
# #         # 原始模型测试
# #         logger.info("\n=== 原始模型测试 ===")
# #         test_model(args.model_path, tokenizer, args.test_prompt)
# #
# #         # 量化模型
# #         logger.info("\n=== 开始量化 ===")
# #         success = quantize_model(args, model, tokenizer)
# #
# #         if success:
# #             # 量化后模型测试
# #             logger.info("\n=== 量化后模型测试 ===")
# #             test_model(args.output_path, tokenizer, args.test_prompt)
# #
# #     except Exception as e:
# #         logger.error(f"主程序运行出错: {str(e)}")
# #         import traceback
# #         logger.error(traceback.format_exc())
# #
# #
# # if __name__ == "__main__":
# #     main()
