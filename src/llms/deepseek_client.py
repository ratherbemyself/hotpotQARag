# -*- coding: utf-8 -*-
"""DeepSeek LLM 客户端

提供 DeepSeek API 的调用接口，兼容 OpenAI API 格式。
"""

from typing import Any, Dict, List, Optional, Iterator
import time

from openai import OpenAI

from src.llms.base_client import BaseLLMClient, Message, LLMResponse
from src.utils.config import get_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DeepSeekClient(BaseLLMClient):
    """DeepSeek LLM 客户端
    
    支持 DeepSeek API 的调用，包括 chat 和 completion 模式。
    兼容 OpenAI API 格式，可同时用于 DeepSeek、通义千问等。
    """
    
    def __init__(
        self,
        model: str = None,
        api_key: Optional[str] = None,
        base_url: str = None,
        **kwargs: Any,
    ):
        """初始化 DeepSeek 客户端
        
        Args:
            model: 模型名称 (deepseek-chat, deepseek-coder, qwen-plus 等)
            api_key: API 密钥
            base_url: API 基础 URL
            **kwargs: 其他参数
        """
        config = get_config()
        
        model = config.llm_model
        api_key = config.openai_api_key
        base_url = config.openai_api_base
        
        super().__init__(model, api_key, base_url, **kwargs)
        
        self._client = None
        self._init_client()
    
    def _init_client(self) -> None:
        """初始化 OpenAI 客户端（DeepSeek 兼容 OpenAI API）"""
        self._client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
        )
        logger.debug(f"LLM 客户端初始化: model={self.model}")
    
    def generate(
        self,
        messages: List[Message],
        temperature: float = None,
        max_tokens: int = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """生成响应
        
        Args:
            messages: 消息列表
            temperature: 温度参数
            max_tokens: 最大 token 数
            **kwargs: 其他参数
            
        Returns:
            LLMResponse 实例
        """
        config = get_config()
        temperature = config.llm_temperature if temperature is None else temperature
        max_tokens = config.llm_max_tokens if max_tokens is None else max_tokens
        
        start_time = time.time()
        
        try:
            request_kwargs = dict(kwargs)
            extra_body = {"thinking": {"type": "disabled"}}
            extra_body.update(request_kwargs.pop("extra_body", {}) or {})
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[m.to_dict() for m in messages],
                temperature=temperature,
                max_tokens=max_tokens,
                extra_body=extra_body,
                **request_kwargs,
            )
            
            latency = time.time() - start_time
            
            content = response.choices[0].message.content or ""
            usage_obj = getattr(response, "usage", None)
            usage = {
                "prompt_tokens": getattr(usage_obj, "prompt_tokens", 0),
                "completion_tokens": getattr(usage_obj, "completion_tokens", 0),
                "total_tokens": getattr(usage_obj, "total_tokens", 0),
            }
            success = bool(content.strip())
            
            logger.debug(
                f"LLM 生成完成: tokens={usage['total_tokens']}, "
                f"latency={latency:.2f}s"
            )
            
            return LLMResponse(
                content=content,
                model=self.model,
                usage=usage,
                latency=latency,
                success=success,
                error=None if success else "empty response",
            )
            
        except Exception as e:
            latency = time.time() - start_time
            logger.error(f"LLM 生成失败: {e}")
            
            return LLMResponse(
                content="",
                model=self.model,
                latency=latency,
                success=False,
                error=str(e),
            )
    
    def stream_generate(
        self,
        messages: List[Message],
        temperature: float = None,
        max_tokens: int = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        """流式生成响应
        
        Args:
            messages: 消息列表
            temperature: 温度参数
            max_tokens: 最大 token 数
            **kwargs: 其他参数
            
        Yields:
            生成的文本片段
        """
        config = get_config()
        temperature = config.llm_temperature if temperature is None else temperature
        max_tokens = config.llm_max_tokens if max_tokens is None else max_tokens
        
        try:
            request_kwargs = dict(kwargs)
            request_kwargs["stream"] = True
            extra_body = {"thinking": {"type": "disabled"}}
            extra_body.update(request_kwargs.pop("extra_body", {}) or {})
            stream = self._client.chat.completions.create(
                model=self.model,
                messages=[m.to_dict() for m in messages],
                temperature=temperature,
                max_tokens=max_tokens,
                extra_body=extra_body,
                **request_kwargs,
            )
            
            for chunk in stream:
                choices = getattr(chunk, "choices", [])
                if not choices:
                    continue
                content = getattr(choices[0].delta, "content", None)
                if content:
                    yield content
                    
        except Exception as e:
            logger.error(f"LLM 流式生成失败: {e}")
            raise


if __name__ == "__main__":
    print("=" * 50)
    print("测试 LLM 客户端")
    print("=" * 50)
    
    client = DeepSeekClient()
    
    print(f"✓ 模型: {client.model}")
    print(f"✓ Base URL: {client.base_url}")
    
    messages = [
        Message(role="system", content="你是一个有帮助的AI助手。"),
        Message(role="user", content="你好，请用一句话介绍你自己。"),
    ]
    
    response = client.generate(messages)
    
    if response.success:
        print(f"✓ 响应: {response.content[:100]}...")
        print(f"✓ Token 使用: {response.usage}")
        print(f"✓ 延迟: {response.latency:.2f}s")
    else:
        print(f"✗ 错误: {response.error}")
    
    print("\n测试完成!")
