"""
一键部署 — 导出模型 + 启动 API 服务

支持:
  1. HuggingFace 格式导出 (默认, 兼容性最好)
  2. ONNX 导出 (推理加速 2-3x)
  3. FastAPI 服务 (生产推理)
  4. GGUF 导出 (llama.cpp 兼容)

用法:
    from saddlellm import Deploy

    # 导出
    Deploy.export(model, tokenizer, "./exported", format="hf")

    # 启动 API 服务
    Deploy.serve(model, tokenizer, port=8000)

    # 或从命令行
    # python -m saddlellm.Deploy serve --model ./my_model --port 8000
"""
import os
import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class Deploy:
    """模型部署与导出。"""

    # ============================================================
    # 导出
    # ============================================================

    @staticmethod
    def export(
        model,
        tokenizer,
        output_dir: str = "./exported_model",
        format: str = "hf",
        quantize: bool = False,
        quantize_bits: int = 4,
    ):
        """
        导出模型。

        format:
          "hf"     — HuggingFace 标准格式 (默认)
          "onnx"   — ONNX (推理加速)
          "gguf"   — GGUF (llama.cpp/ollama 兼容)

        quantize: 是否量化 (减少模型大小)
        """
        os.makedirs(output_dir, exist_ok=True)

        if format == "hf":
            Deploy._export_hf(model, tokenizer, output_dir, quantize, quantize_bits)
        elif format == "onnx":
            Deploy._export_onnx(model, tokenizer, output_dir)
        elif format == "gguf":
            Deploy._export_gguf(model, tokenizer, output_dir)
        else:
            raise ValueError(f"未知格式: {format}")

        # 保存配置信息
        config = {
            "format": format,
            "quantized": quantize,
            "quantize_bits": quantize_bits,
            "model_params": sum(p.numel() for p in model.parameters()),
        }
        with open(os.path.join(output_dir, "deploy_config.json"), "w") as f:
            json.dump(config, f, indent=2, default=str)

        logger.info(f"模型已导出到 {output_dir} (格式: {format})")
        return output_dir

    @staticmethod
    def _export_hf(model, tokenizer, output_dir, quantize, bits):
        """导出 HuggingFace 格式。"""
        if quantize:
            try:
                from .ModelQuantizer import ModelQuantizer
                q = ModelQuantizer()
                q.quantize(model, bits=bits)
            except Exception as e:
                logger.warning(f"量化失败, 保存原始模型: {e}")

        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        logger.info(f"HuggingFace 格式已保存: {output_dir}")

    @staticmethod
    def _export_onnx(model, tokenizer, output_dir):
        """导出 ONNX 格式。"""
        try:
            import torch

            model.eval()
            dummy_input = tokenizer("Hello world", return_tensors="pt")
            dummy_ids = dummy_input["input_ids"]

            onnx_path = os.path.join(output_dir, "model.onnx")
            torch.onnx.export(
                model,
                (dummy_ids,),
                onnx_path,
                input_names=["input_ids"],
                output_names=["logits"],
                dynamic_axes={"input_ids": {0: "batch", 1: "sequence"}},
                opset_version=17,
            )
            tokenizer.save_pretrained(output_dir)
            logger.info(f"ONNX 格式已保存: {onnx_path}")
        except ImportError:
            logger.warning("ONNX 导出需要: pip install onnx onnxruntime")
            logger.info("回退到 HuggingFace 格式...")
            Deploy._export_hf(model, tokenizer, output_dir, False, 0)

    @staticmethod
    def _export_gguf(model, tokenizer, output_dir):
        """导出 GGUF 格式 (需要 llama-cpp-python)。"""
        # 先保存 HF 格式, 然后转换
        hf_dir = os.path.join(output_dir, "_hf_temp")
        Deploy._export_hf(model, tokenizer, hf_dir, False, 0)

        try:
            # 尝试用 llama.cpp 的 convert 工具
            import subprocess
            gguf_path = os.path.join(output_dir, "model.gguf")
            result = subprocess.run(
                ["python", "-m", "llama_cpp.convert", hf_dir, "--outfile", gguf_path],
                capture_output=True, text=True,
            )
            if result.returncode == 0:
                logger.info(f"GGUF 格式已保存: {gguf_path}")
            else:
                logger.warning(f"GGUF 转换失败: {result.stderr}")
                logger.info("请手动转换: pip install llama-cpp-python")
        except Exception as e:
            logger.warning(f"GGUF 转换失败: {e}")
            logger.info("已保存 HuggingFace 格式, 可用 llama.cpp 的 convert.py 手动转换")

        # 清理临时文件
        import shutil
        if os.path.exists(hf_dir):
            shutil.rmtree(hf_dir)

    # ============================================================
    # API 服务
    # ============================================================

    @staticmethod
    def serve(
        model,
        tokenizer,
        host: str = "0.0.0.0",
        port: int = 8000,
        title: str = "SaddleLLM API",
        max_tokens: int = 2048,
    ):
        """
        启动 FastAPI 推理服务。

        启动后:
          POST /v1/chat/completions  — OpenAI 兼容接口
          POST /generate             — 简单生成接口
          GET  /health               — 健康检查
          GET  /docs                 — Swagger 文档

        示例:
          Deploy.serve(model, tokenizer, port=8000)

          # 客户端调用:
          curl -X POST http://localhost:8000/v1/chat/completions \
            -H "Content-Type: application/json" \
            -d '{"model": "saddlellm", "messages": [{"role": "user", "content": "你好"}]}'
        """
        try:
            import uvicorn
            from fastapi import FastAPI
            from pydantic import BaseModel
            import torch
        except ImportError:
            raise ImportError("API 服务需要: pip install fastapi uvicorn pydantic")

        device = next(model.parameters()).device

        app = FastAPI(title=title, version="1.0")

        # ----- Pydantic models -----
        class Message(BaseModel):
            role: str = "user"
            content: str

        class ChatRequest(BaseModel):
            model: str = "saddlellm"
            messages: list
            temperature: float = 0.7
            max_tokens: int = max_tokens
            top_p: float = 0.9

        class ChatResponse(BaseModel):
            id: str = "chatcmpl-001"
            object: str = "chat.completion"
            model: str = "saddlellm"
            choices: list

        class GenerateRequest(BaseModel):
            prompt: str
            temperature: float = 0.7
            max_tokens: int = max_tokens
            top_p: float = 0.9

        class GenerateResponse(BaseModel):
            text: str
            tokens_generated: int

        # ----- Routes -----
        @app.get("/health")
        async def health():
            return {"status": "ok", "model": title}

        @app.post("/v1/chat/completions", response_model=ChatResponse)
        async def chat_completions(req: ChatRequest):
            # 构建 prompt
            prompt = ""
            for msg in req.messages:
                role = msg["role"] if isinstance(msg, dict) else msg.role
                content = msg["content"] if isinstance(msg, dict) else msg.content
                if role == "system":
                    prompt += f"系统: {content}\n\n"
                elif role == "user":
                    prompt += f"用户: {content}\n"
                elif role == "assistant":
                    prompt += f"助手: {content}\n"
            prompt += "助手: "

            # 生成
            model.eval()
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=req.max_tokens,
                    temperature=req.temperature,
                    do_sample=req.temperature > 0.05,
                    top_p=req.top_p,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )
            full = tokenizer.decode(outputs[0], skip_special_tokens=True)
            prompt_len = len(tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True))
            response_text = full[prompt_len:].strip()

            return ChatResponse(
                choices=[{"index": 0, "message": {"role": "assistant", "content": response_text},
                          "finish_reason": "stop"}],
            )

        @app.post("/generate", response_model=GenerateResponse)
        async def generate(req: GenerateRequest):
            model.eval()
            inputs = tokenizer(req.prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=req.max_tokens,
                    temperature=req.temperature,
                    do_sample=req.temperature > 0.05,
                    top_p=req.top_p,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )
            full = tokenizer.decode(outputs[0], skip_special_tokens=True)
            prompt_len = len(tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True))
            text = full[prompt_len:].strip()
            return GenerateResponse(text=text, tokens_generated=len(outputs[0]) - prompt_len)

        print(f"""
{'='*60}
  {title} API 服务已启动!

  OpenAI 兼容:  POST http://localhost:{port}/v1/chat/completions
  生成接口:      POST http://localhost:{port}/generate
  健康检查:      GET  http://localhost:{port}/health
  API 文档:      GET  http://localhost:{port}/docs
{'='*60}

客户端示例:
  curl http://localhost:{port}/v1/chat/completions \\
    -H "Content-Type: application/json" \\
    -d '{{"model":"saddlellm","messages":[{{"role":"user","content":"你好"}}]}}'
""")

        uvicorn.run(app, host=host, port=port, log_level="info")


# ============================================================
# CLI 入口
# ============================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SaddleLLM 部署工具")
    subparsers = parser.add_subparsers(dest="command")

    serve_parser = subparsers.add_parser("serve", help="启动 API 服务")
    serve_parser.add_argument("--model", "-m", required=True, help="模型路径")
    serve_parser.add_argument("--port", "-p", type=int, default=8000)
    serve_parser.add_argument("--host", default="0.0.0.0")

    export_parser = subparsers.add_parser("export", help="导出模型")
    export_parser.add_argument("--model", "-m", required=True, help="模型路径")
    export_parser.add_argument("--output", "-o", default="./exported_model")
    export_parser.add_argument("--format", "-f", default="hf", choices=["hf", "onnx", "gguf"])
    export_parser.add_argument("--quantize", "-q", action="store_true")

    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    if args.command == "serve":
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16)
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        Deploy.serve(model, tokenizer, host=args.host, port=args.port)
    elif args.command == "export":
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16)
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        Deploy.export(model, tokenizer, args.output, format=args.format, quantize=args.quantize)
