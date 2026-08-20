"""
Chat 界面 — 一键启动 Web 聊天 UI

支持:
  - Gradio 界面 (推荐, 功能全)
  - 终端聊天 (轻量, 无依赖)
  - 与 easy 模块无缝集成

用法:
    # 方式1: 用 easy 模块
    import saddlellm.easy as llm
    llm.chat_ui(model)           # 启动 Gradio

    # 方式2: 直接启动
    from saddlellm import ChatUI
    ChatUI.launch(model, tokenizer)

    # 方式3: 终端聊天
    ChatUI.terminal(model, tokenizer)
"""
import logging

logger = logging.getLogger(__name__)


class ChatUI:
    """一键启动模型聊天界面。"""

    @staticmethod
    def launch(
        model,
        tokenizer=None,
        title: str = "SaddleLLM Chat",
        port: int = 7860,
        share: bool = False,
        use_cot: bool = True,
        max_tokens: int = 1024,
    ):
        """
        启动 Gradio Web 聊天界面。

        示例:
            ChatUI.launch(model, tokenizer, title="我的 300M 模型")
        """
        import torch
        from transformers import AutoTokenizer

        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
            tokenizer.pad_token = tokenizer.eos_token
        device = next(model.parameters()).device

        # 对话历史
        history = []

        def chat_fn(message, history_list, temperature, top_p, do_cot):
            # 构建 prompt
            if do_cot:
                prompt = f"问题: {message}\n\n让我们一步步思考。\n\n"
            else:
                prompt = f"用户: {message}\n助手: "

            # 多轮对话
            if history_list:
                context = ""
                for h in history_list[-3:]:  # 最近 3 轮
                    context += f"用户: {h[0]}\n助手: {h[1]}\n"
                prompt = context + prompt

            # 生成
            model.eval()
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0.05,
                    top_p=top_p,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )

            full = tokenizer.decode(outputs[0], skip_special_tokens=True)
            prompt_len = len(tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True))
            response = full[prompt_len:].strip()

            history_list.append((message, response))
            return "", history_list

        try:
            import gradio as gr

            with gr.Blocks(title=title, theme=gr.themes.Soft()) as demo:
                gr.Markdown(f"# {title}")
                gr.Markdown("模型已加载, 开始对话吧!")

                chatbot = gr.Chatbot(height=500)
                msg = gr.Textbox(placeholder="输入你的问题...", label="消息")
                clear = gr.Button("清除对话")

                with gr.Row():
                    temperature = gr.Slider(0.05, 1.5, value=0.7, step=0.05, label="Temperature")
                    top_p = gr.Slider(0.1, 1.0, value=0.9, step=0.05, label="Top-p")
                    do_cot = gr.Checkbox(value=True, label="Chain-of-Thought 推理")

                msg.submit(chat_fn, [msg, chatbot, temperature, top_p, do_cot], [msg, chatbot])
                clear.click(lambda: [], None, chatbot)

            print(f"\n{'='*50}")
            print(f"  {title}")
            print(f"  打开 http://localhost:{port}")
            print(f"{'='*50}\n")
            demo.launch(server_port=port, share=share)

        except ImportError:
            print("Gradio 未安装。安装: pip install gradio")
            print("回退到终端聊天模式...")
            ChatUI.terminal(model, tokenizer)

    @staticmethod
    def terminal(model, tokenizer=None, use_cot: bool = True):
        """
        终端聊天模式 (无额外依赖)。

        命令:
          /cot       切换 CoT 推理
          /clear     清除历史
          /quit      退出
        """
        from transformers import AutoTokenizer

        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
            tokenizer.pad_token = tokenizer.eos_token
        device = next(model.parameters()).device

        history = []
        print("\n" + "=" * 50)
        print("  SaddleLLM 终端聊天")
        print("  命令: /cot(切换推理) /clear(清除) /quit(退出)")
        print("=" * 50 + "\n")

        while True:
            try:
                msg = input("你: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n再见!")
                break

            if not msg:
                continue
            if msg == "/quit":
                print("再见!")
                break
            if msg == "/clear":
                history = []
                print("[历史已清除]")
                continue
            if msg == "/cot":
                use_cot = not use_cot
                print(f"[CoT 推理: {'开' if use_cot else '关'}]")
                continue

            # 构建 prompt
            if use_cot:
                prompt = f"问题: {msg}\n\n让我们一步步思考。\n\n"
            else:
                prompt = f"用户: {msg}\n助手: "

            if history:
                context = ""
                for h in history[-3:]:
                    context += f"用户: {h['user']}\n助手: {h['assistant']}\n"
                prompt = context + prompt

            # 生成
            model.eval()
            import torch
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)

            print("助手: ", end="", flush=True)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )
            full = tokenizer.decode(outputs[0], skip_special_tokens=True)
            prompt_len = len(tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True))
            response = full[prompt_len:].strip()
            print(response)
            print()

            history.append({"user": msg, "assistant": response})


# 挂载到 easy 模块
def _patch_easy():
    """给 easy 模块添加 chat_ui 函数。"""
    import saddlellm.easy as easy_mod
    easy_mod.chat_ui = lambda model, tokenizer=None, **kw: ChatUI.launch(model, tokenizer, **kw)
    easy_mod.terminal_chat = lambda model, tokenizer=None: ChatUI.terminal(model, tokenizer)


_patch_easy()
