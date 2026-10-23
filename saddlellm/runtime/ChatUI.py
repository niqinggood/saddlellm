"""Compatibility helpers for launching SaddleLLM chat interfaces."""
from __future__ import annotations

import logging
from typing import Any

from .InferenceServer import (
    ChatCompletionRequest,
    ChatMessage,
    InferenceServerSettings,
    LocalGenerationEngine,
    create_inference_app,
)


logger = logging.getLogger(__name__)


class ChatUI:
    """Launch the same chat runtime used by ``saddlellm serve-model``."""

    @staticmethod
    def launch(
        model: Any,
        tokenizer: Any,
        title: str = "SaddleLLM 对话",
        port: int = 7860,
        share: bool = False,
        use_cot: bool = False,
        max_tokens: int = 256,
        host: str = "127.0.0.1",
    ) -> None:
        """Serve a model object with the built-in Web UI and OpenAI API.

        ``share`` remains for source compatibility but is deliberately ignored:
        SaddleLLM does not create an external public tunnel implicitly.
        """
        if tokenizer is None:
            raise ValueError("tokenizer is required; implicit GPT-2 fallback was unsafe for chat models.")
        if share:
            logger.warning("share=True is no longer implicit; expose the service through your approved gateway.")
        try:
            import uvicorn
        except ImportError as exc:
            raise ImportError("ChatUI requires the SaddleLLM serve extra.") from exc
        settings = InferenceServerSettings(
            model_name=getattr(getattr(model, "config", None), "name_or_path", None),
            default_max_new_tokens=max_tokens,
            max_new_tokens=max_tokens,
            enable_thinking=use_cot,
            ui_title=title,
        )
        app = create_inference_app(settings, model=model, tokenizer=tokenizer)
        print(f"SaddleLLM chat and API: http://{host}:{port}", flush=True)
        uvicorn.run(app, host=host, port=port, log_level="info")

    @staticmethod
    def terminal(
        model: Any,
        tokenizer: Any,
        use_cot: bool = False,
        max_tokens: int = 256,
    ) -> None:
        """Run terminal chat through the exact same prompt and generation path."""
        if tokenizer is None:
            raise ValueError("tokenizer is required.")
        settings = InferenceServerSettings(
            default_max_new_tokens=max_tokens,
            max_new_tokens=max_tokens,
            enable_thinking=use_cot,
            web_ui=False,
        )
        engine = LocalGenerationEngine(model, tokenizer, settings)
        history = []
        print("SaddleLLM terminal chat · /clear · /thinking · /quit")
        while True:
            try:
                message = input("你: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n再见")
                return
            if not message:
                continue
            if message == "/quit":
                return
            if message == "/clear":
                history.clear()
                print("[历史已清除]")
                continue
            if message == "/thinking":
                settings.enable_thinking = not settings.enable_thinking
                print(f"[思考模式: {'开' if settings.enable_thinking else '关'}]")
                continue
            history.append(ChatMessage(role="user", content=message))
            request = ChatCompletionRequest(
                messages=history,
                max_tokens=max_tokens,
                temperature=0.2,
            )
            print("助手: ", end="", flush=True)
            response = ""
            for piece in engine.stream_chat(request):
                response += piece
                print(piece, end="", flush=True)
            print("\n")
            history.append(ChatMessage(role="assistant", content=response))


def _patch_easy() -> None:
    import saddlellm.easy as easy_mod

    easy_mod.chat_ui = lambda model, tokenizer=None, **kwargs: ChatUI.launch(
        model, tokenizer, **kwargs
    )
    easy_mod.terminal_chat = lambda model, tokenizer=None, **kwargs: ChatUI.terminal(
        model, tokenizer, **kwargs
    )


_patch_easy()


__all__ = ["ChatUI"]
