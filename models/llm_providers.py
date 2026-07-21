"""LLM Providers"""

import os
import logging
from typing import List, Optional, Dict
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from openai import OpenAI

logger = logging.getLogger(__name__)


class LLMProvider(Enum):
    HUGGINGFACE = "huggingface"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    DEEPSEEK = "deepseek"
    GROK = "grok"
    KIMI = "kimi"
    MINIMAX = "minimax"
    QWEN = "qwen"
    GLM = "glm"


@dataclass
class LLMConfig:
    """Configuration for LLM provider"""
    provider: LLMProvider
    model_name: str
    temperature: float = 0.7
    max_tokens: int = 8192
    top_p: float = 0.9
    additional_params: Dict = field(default_factory=dict)


@dataclass
class LLMResponse:
    """Response from LLM provider"""
    content: str
    model: str
    provider: str
    metadata: Dict = field(default_factory=dict)


class BaseLLMProvider(ABC):
    """Base class for LLM providers"""

    # Maximum number of prior turns (user+assistant pairs) to include
    MAX_HISTORY_TURNS = 6

    def __init__(self, config: LLMConfig):
        self.config = config
        self.provider_name = config.provider.value

    @abstractmethod
    def generate_response(
        self,
        system_prompt: str,
        user_prompt: str,
        context: Optional[str] = None,
        conversation_history: Optional[List[Dict]] = None,
    ) -> LLMResponse:
        """Generate response from LLM"""
        pass

    def _build_messages(
        self,
        system_prompt: str,
        user_prompt: str,
        context: Optional[str],
        conversation_history: Optional[List[Dict]],
    ) -> List[Dict[str, str]]:
        """Build the chat messages array with optional multi-turn history.

        History entries are dicts with keys 'type' (user/assistant) and 'message'.
        Returns the messages list ready to send to any OpenAI-compatible API.
        """
        messages: List[Dict[str, str]] = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # Inject prior turns (capped to avoid token bloat)
        if conversation_history:
            recent = conversation_history[-self.MAX_HISTORY_TURNS:]
            for turn in recent:
                role = turn.get("type", turn.get("role", ""))
                content = turn.get("message", turn.get("content", ""))
                if not content:
                    continue
                if role == "user":
                    messages.append({"role": "user", "content": content})
                elif role in ("assistant", "ai"):
                    messages.append({"role": "assistant", "content": content})

        # Current user message — prepend RAG context if present
        if context:
            user_content = f"Context from literature:\n{context}\n\nQuestion: {user_prompt}"
        else:
            user_content = user_prompt

        messages.append({"role": "user", "content": user_content})
        return messages

    def _generate_fallback_response(self, system_prompt: str, user_prompt: str, context: Optional[str] = None) -> str:
        """Generate a meaningful fallback response"""
        return f"""I apologize, but I'm unable to generate a response at the moment. 

However, for your question about the corpus: instead of guessing, you might re-read the passage and consider:

1. **Language & Style** - Word choice, rhythm, syntax
2. **Imagery & Symbolism** - Recurring images, metaphors
3. **Character Development** - Actions, dialogue, inner thoughts
4. **Narrative Structure** - POV, timeline, pacing
5. **Thematic Elements** - Central themes and their expression

For more detailed analysis, please try again or rephrase your question."""


class HuggingFaceProvider(BaseLLMProvider):
    """HuggingFace Inference API provider - calls models hosted on HF servers"""

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.client = None
        self._initialize_client()

    def _initialize_client(self):
        """Initialize HuggingFace router client using OpenAI-compatible API"""
        try:
            # Prefer HUGGINGFACE_API_KEY to avoid conflict with HF Spaces auto-injected HF_TOKEN
            hf_token = os.getenv("HUGGINGFACE_API_KEY") or os.getenv("HF_TOKEN")
            
            if not hf_token:
                raise ValueError("HUGGINGFACE_API_KEY or HF_TOKEN environment variable is required")
            
            self.client = OpenAI(
                base_url="https://router.huggingface.co/v1",
                api_key=hf_token,
                timeout=120
            )
            
            logger.info(f"HF Router API initialized for {self.config.model_name} (token prefix: {hf_token[:8]}...)")
            
        except Exception as e:
            logger.error(f"Failed to initialize HF Router client: {str(e)}")
            self.client = None

    def _format_chat_messages(self, messages: List[Dict[str, str]]) -> str:
        """Format messages for the model"""
        model_lower = self.config.model_name.lower()
        
        # Qwen format
        if "qwen" in model_lower:
            formatted = ""
            for msg in messages:
                role = msg["role"]
                content = msg["content"]
                if role == "system":
                    formatted += f"<|im_start|>system\n{content}<|im_end|>\n"
                elif role == "user":
                    formatted += f"<|im_start|>user\n{content}<|im_end|>\n"
                elif role == "assistant":
                    formatted += f"<|im_start|>assistant\n{content}<|im_end|>\n"
            return formatted + "<|im_start|>assistant\n"
        
        # Llama 3 format
        elif "llama" in model_lower:
            formatted = ""
            for msg in messages:
                role = msg["role"]
                content = msg["content"]
                if role == "system":
                    formatted += f"<|start_header_id|>system<|end_header_id|>\n\n{content}<|eot_id|>"
                elif role == "user":
                    formatted += f"<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|>"
                elif role == "assistant":
                    formatted += f"<|start_header_id|>assistant<|end_header_id|>\n\n{content}<|eot_id|>"
            return formatted + "<|start_header_id|>assistant<|end_header_id|>\n\n"
        
        # Mistral format
        elif "mistral" in model_lower or "mixtral" in model_lower:
            formatted = ""
            for msg in messages:
                role = msg["role"]
                content = msg["content"]
                if role == "system":
                    formatted += f"[INST] <<SYS>>\n{content}\n<</SYS>>\n\n"
                elif role == "user":
                    formatted += f"[INST]{content}[/INST]"
                elif role == "assistant":
                    formatted += f" {content}"
            return formatted + " "
        
        # Generic format
        else:
            formatted = ""
            for msg in messages:
                role = msg["role"]
                content = msg["content"]
                if role == "system":
                    formatted += f"System: {content}\n\n"
                elif role == "user":
                    formatted += f"User: {content}\n\n"
                elif role == "assistant":
                    formatted += f"Assistant: {content}\n\n"
            return formatted + "Assistant:"

    def generate_response(
        self,
        system_prompt: str,
        user_prompt: str,
        context: Optional[str] = None,
        conversation_history: Optional[List[Dict]] = None,
    ) -> LLMResponse:
        """Generate response using HuggingFace Router API with OpenAI-compatible interface"""
        if not self.client:
            raise RuntimeError("HuggingFace Router client not initialized")

        try:
            messages = self._build_messages(system_prompt, user_prompt, context, conversation_history)
            
            logger.info(f"Calling HF Router API with {self.config.model_name} ({len(messages)} messages)")
            
            response = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p
            )
            
            # Extract response using OpenAI format
            response_text = response.choices[0].message.content
            response_text = response_text.strip() if response_text else ""
            
            if len(response_text) < 5:
                response_text = self._generate_fallback_response(system_prompt, user_prompt, context)

            return LLMResponse(
                content=response_text,
                model=self.config.model_name,
                provider=self.provider_name,
                metadata={"api": "huggingface_router"}
            )

        except Exception as e:
            logger.error(f"Error calling HF Router API: {str(e)}")
            fallback = self._generate_fallback_response(system_prompt, user_prompt, context)
            return LLMResponse(
                content=fallback,
                model=self.config.model_name,
                provider=self.provider_name,
                metadata={"error": str(e), "fallback": True}
            )


class OpenAIProvider(BaseLLMProvider):
    """OpenAI API provider - supports GPT-4o, GPT-4o-mini, etc."""

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.client = None
        self._initialize_client()

    def _initialize_client(self):
        """Initialize OpenAI client"""
        try:
            api_key = self.config.additional_params.get("api_key") or os.getenv("OPENAI_API_KEY")
            
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable is required")
            
            self.client = OpenAI(
                api_key=api_key,
                timeout=120
            )
            
            logger.info(f"OpenAI client initialized for {self.config.model_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {str(e)}")
            self.client = None

    def generate_response(
        self,
        system_prompt: str,
        user_prompt: str,
        context: Optional[str] = None,
        conversation_history: Optional[List[Dict]] = None,
    ) -> LLMResponse:
        """Generate response using OpenAI API"""
        if not self.client:
            raise RuntimeError("OpenAI client not initialized")

        try:
            messages = self._build_messages(system_prompt, user_prompt, context, conversation_history)
            
            logger.info(f"Calling OpenAI API with {self.config.model_name} ({len(messages)} messages)")
            
            response = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p
            )
            
            response_text = response.choices[0].message.content
            response_text = response_text.strip() if response_text else ""

            if len(response_text) < 5:
                response_text = self._generate_fallback_response(system_prompt, user_prompt, context)

            return LLMResponse(
                content=response_text,
                model=self.config.model_name,
                provider=self.provider_name,
                metadata={"api": "openai"}
            )

        except Exception as e:
            logger.error(f"Error calling OpenAI API: {str(e)}")
            fallback = self._generate_fallback_response(system_prompt, user_prompt, context)
            return LLMResponse(
                content=fallback,
                model=self.config.model_name,
                provider=self.provider_name,
                metadata={"error": str(e), "fallback": True}
            )


class AnthropicProvider(BaseLLMProvider):
    """Anthropic API provider - supports Claude 3.5, Claude 3, etc."""

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.client = None
        self._initialize_client()

    def _initialize_client(self):
        """Initialize Anthropic client"""
        try:
            api_key = self.config.additional_params.get("api_key") or os.getenv("ANTHROPIC_API_KEY")
            
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY environment variable is required")
            
            # Use OpenAI-compatible client for Anthropic
            self.client = OpenAI(
                base_url="https://api.anthropic.com/v1",
                api_key=api_key,
                timeout=120
            )
            
            logger.info(f"Anthropic client initialized for {self.config.model_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Anthropic client: {str(e)}")
            self.client = None

    def generate_response(
        self,
        system_prompt: str,
        user_prompt: str,
        context: Optional[str] = None,
        conversation_history: Optional[List[Dict]] = None,
    ) -> LLMResponse:
        """Generate response using Anthropic API"""
        if not self.client:
            raise RuntimeError("Anthropic client not initialized")

        try:
            messages = self._build_messages(system_prompt, user_prompt, context, conversation_history)
            
            logger.info(f"Calling Anthropic API with {self.config.model_name} ({len(messages)} messages)")
            
            response = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p
            )
            
            response_text = response.choices[0].message.content
            response_text = response_text.strip() if response_text else ""

            if len(response_text) < 5:
                response_text = self._generate_fallback_response(system_prompt, user_prompt, context)

            return LLMResponse(
                content=response_text,
                model=self.config.model_name,
                provider=self.provider_name,
                metadata={"api": "anthropic"}
            )

        except Exception as e:
            logger.error(f"Error calling Anthropic API: {str(e)}")
            fallback = self._generate_fallback_response(system_prompt, user_prompt, context)
            return LLMResponse(
                content=fallback,
                model=self.config.model_name,
                provider=self.provider_name,
                metadata={"error": str(e), "fallback": True}
            )


class DeepSeekProvider(BaseLLMProvider):
    """DeepSeek API provider - supports DeepSeek Chat, DeepSeek Coder, etc."""

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.client = None
        self._initialize_client()

    def _initialize_client(self):
        """Initialize DeepSeek client"""
        try:
            api_key = self.config.additional_params.get("api_key") or os.getenv("DEEPSEEK_API_KEY")
            base_url = self.config.additional_params.get("base_url") or "https://api.deepseek.com"
            
            if not api_key:
                raise ValueError("DEEPSEEK_API_KEY environment variable is required")
            
            self.client = OpenAI(
                base_url=base_url,
                api_key=api_key,
                timeout=120
            )
            
            logger.info(f"DeepSeek client initialized for {self.config.model_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize DeepSeek client: {str(e)}")
            self.client = None

    def generate_response(
        self,
        system_prompt: str,
        user_prompt: str,
        context: Optional[str] = None,
        conversation_history: Optional[List[Dict]] = None,
    ) -> LLMResponse:
        """Generate response using DeepSeek API"""
        if not self.client:
            raise RuntimeError("DeepSeek client not initialized")

        try:
            messages = self._build_messages(system_prompt, user_prompt, context, conversation_history)
            
            logger.info(f"Calling DeepSeek API with {self.config.model_name} ({len(messages)} messages)")
            
            response = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p
            )
            
            response_text = response.choices[0].message.content
            response_text = response_text.strip() if response_text else ""

            if len(response_text) < 5:
                response_text = self._generate_fallback_response(system_prompt, user_prompt, context)

            return LLMResponse(
                content=response_text,
                model=self.config.model_name,
                provider=self.provider_name,
                metadata={"api": "deepseek"}
            )

        except Exception as e:
            logger.error(f"Error calling DeepSeek API: {str(e)}")
            fallback = self._generate_fallback_response(system_prompt, user_prompt, context)
            return LLMResponse(
                content=fallback,
                model=self.config.model_name,
                provider=self.provider_name,
                metadata={"error": str(e), "fallback": True}
            )


class GrokProvider(BaseLLMProvider):
    """Grok (xAI) API provider - supports Grok models"""

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.client = None
        self._initialize_client()

    def _initialize_client(self):
        """Initialize Grok client"""
        try:
            api_key = self.config.additional_params.get("api_key") or os.getenv("GROK_API_KEY")
            base_url = self.config.additional_params.get("base_url") or "https://api.x.ai/v1"
            
            if not api_key:
                raise ValueError("GROK_API_KEY environment variable is required")
            
            self.client = OpenAI(
                base_url=base_url,
                api_key=api_key,
                timeout=120
            )
            
            logger.info(f"Grok client initialized for {self.config.model_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Grok client: {str(e)}")
            self.client = None

    def generate_response(
        self,
        system_prompt: str,
        user_prompt: str,
        context: Optional[str] = None,
        conversation_history: Optional[List[Dict]] = None,
    ) -> LLMResponse:
        """Generate response using Grok API"""
        if not self.client:
            raise RuntimeError("Grok client not initialized")

        try:
            messages = self._build_messages(system_prompt, user_prompt, context, conversation_history)
            
            logger.info(f"Calling Grok API with {self.config.model_name} ({len(messages)} messages)")
            
            response = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p
            )
            
            response_text = response.choices[0].message.content
            response_text = response_text.strip() if response_text else ""

            if len(response_text) < 5:
                response_text = self._generate_fallback_response(system_prompt, user_prompt, context)

            return LLMResponse(
                content=response_text,
                model=self.config.model_name,
                provider=self.provider_name,
                metadata={"api": "grok"}
            )

        except Exception as e:
            logger.error(f"Error calling Grok API: {str(e)}")
            fallback = self._generate_fallback_response(system_prompt, user_prompt, context)
            return LLMResponse(
                content=fallback,
                model=self.config.model_name,
                provider=self.provider_name,
                metadata={"error": str(e), "fallback": True}
            )


class KimiProvider(BaseLLMProvider):
    """Kimi (Moonshot AI) API provider - supports Moonshot models"""

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.client = None
        self._initialize_client()

    def _initialize_client(self):
        """Initialize Kimi client"""
        try:
            api_key = self.config.additional_params.get("api_key") or os.getenv("KIMI_API_KEY")
            base_url = self.config.additional_params.get("base_url") or "https://api.moonshot.cn/v1"
            
            if not api_key:
                raise ValueError("KIMI_API_KEY environment variable is required")
            
            self.client = OpenAI(
                base_url=base_url,
                api_key=api_key,
                timeout=120
            )
            
            logger.info(f"Kimi client initialized for {self.config.model_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Kimi client: {str(e)}")
            self.client = None

    def generate_response(
        self,
        system_prompt: str,
        user_prompt: str,
        context: Optional[str] = None,
        conversation_history: Optional[List[Dict]] = None,
    ) -> LLMResponse:
        """Generate response using Kimi API"""
        if not self.client:
            raise RuntimeError("Kimi client not initialized")

        try:
            messages = self._build_messages(system_prompt, user_prompt, context, conversation_history)
            
            logger.info(f"Calling Kimi API with {self.config.model_name} ({len(messages)} messages)")
            
            response = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p
            )
            
            response_text = response.choices[0].message.content
            response_text = response_text.strip() if response_text else ""

            if len(response_text) < 5:
                response_text = self._generate_fallback_response(system_prompt, user_prompt, context)

            return LLMResponse(
                content=response_text,
                model=self.config.model_name,
                provider=self.provider_name,
                metadata={"api": "kimi"}
            )

        except Exception as e:
            logger.error(f"Error calling Kimi API: {str(e)}")
            fallback = self._generate_fallback_response(system_prompt, user_prompt, context)
            return LLMResponse(
                content=fallback,
                model=self.config.model_name,
                provider=self.provider_name,
                metadata={"error": str(e), "fallback": True}
            )


class MinimaxProvider(BaseLLMProvider):
    """Minimax API provider - supports Minimax models"""

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.client = None
        self._initialize_client()

    def _initialize_client(self):
        """Initialize Minimax client"""
        try:
            api_key = self.config.additional_params.get("api_key") or os.getenv("MINIMAX_API_KEY")
            base_url = self.config.additional_params.get("base_url") or "https://api.minimax.chat/v1"
            
            if not api_key:
                raise ValueError("MINIMAX_API_KEY environment variable is required")
            
            self.client = OpenAI(
                base_url=base_url,
                api_key=api_key,
                timeout=120
            )
            
            logger.info(f"Minimax client initialized for {self.config.model_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Minimax client: {str(e)}")
            self.client = None

    def generate_response(
        self,
        system_prompt: str,
        user_prompt: str,
        context: Optional[str] = None,
        conversation_history: Optional[List[Dict]] = None,
    ) -> LLMResponse:
        """Generate response using Minimax API"""
        if not self.client:
            raise RuntimeError("Minimax client not initialized")

        try:
            messages = self._build_messages(system_prompt, user_prompt, context, conversation_history)
            
            logger.info(f"Calling Minimax API with {self.config.model_name} ({len(messages)} messages)")
            
            response = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p
            )
            
            response_text = response.choices[0].message.content
            response_text = response_text.strip() if response_text else ""

            if len(response_text) < 5:
                response_text = self._generate_fallback_response(system_prompt, user_prompt, context)

            return LLMResponse(
                content=response_text,
                model=self.config.model_name,
                provider=self.provider_name,
                metadata={"api": "minimax"}
            )

        except Exception as e:
            logger.error(f"Error calling Minimax API: {str(e)}")
            fallback = self._generate_fallback_response(system_prompt, user_prompt, context)
            return LLMResponse(
                content=fallback,
                model=self.config.model_name,
                provider=self.provider_name,
                metadata={"error": str(e), "fallback": True}
            )


class QwenProvider(BaseLLMProvider):
    """Qwen (Alibaba Cloud) API provider - supports Qwen models"""

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.client = None
        self._initialize_client()

    def _initialize_client(self):
        """Initialize Qwen client"""
        try:
            api_key = self.config.additional_params.get("api_key") or os.getenv("QWEN_API_KEY")
            base_url = self.config.additional_params.get("base_url") or "https://dashscope.aliyuncs.com/compatible-mode/v1"
            
            if not api_key:
                raise ValueError("QWEN_API_KEY environment variable is required")
            
            self.client = OpenAI(
                base_url=base_url,
                api_key=api_key,
                timeout=120
            )
            
            logger.info(f"Qwen client initialized for {self.config.model_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Qwen client: {str(e)}")
            self.client = None

    def generate_response(
        self,
        system_prompt: str,
        user_prompt: str,
        context: Optional[str] = None,
        conversation_history: Optional[List[Dict]] = None,
    ) -> LLMResponse:
        """Generate response using Qwen API"""
        if not self.client:
            raise RuntimeError("Qwen client not initialized")

        try:
            messages = self._build_messages(system_prompt, user_prompt, context, conversation_history)
            
            logger.info(f"Calling Qwen API with {self.config.model_name} ({len(messages)} messages)")
            
            response = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p
            )
            
            response_text = response.choices[0].message.content
            response_text = response_text.strip() if response_text else ""

            if len(response_text) < 5:
                response_text = self._generate_fallback_response(system_prompt, user_prompt, context)

            return LLMResponse(
                content=response_text,
                model=self.config.model_name,
                provider=self.provider_name,
                metadata={"api": "qwen"}
            )

        except Exception as e:
            logger.error(f"Error calling Qwen API: {str(e)}")
            fallback = self._generate_fallback_response(system_prompt, user_prompt, context)
            return LLMResponse(
                content=fallback,
                model=self.config.model_name,
                provider=self.provider_name,
                metadata={"error": str(e), "fallback": True}
            )


class GLMProvider(BaseLLMProvider):
    """GLM (Zhipu AI) API provider - supports ChatGLM models"""

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.client = None
        self._initialize_client()

    def _initialize_client(self):
        """Initialize GLM client"""
        try:
            api_key = self.config.additional_params.get("api_key") or os.getenv("GLM_API_KEY")
            base_url = self.config.additional_params.get("base_url") or "https://open.bigmodel.cn/api/paas/v4"
            
            if not api_key:
                raise ValueError("GLM_API_KEY environment variable is required")
            
            self.client = OpenAI(
                base_url=base_url,
                api_key=api_key,
                timeout=120
            )
            
            logger.info(f"GLM client initialized for {self.config.model_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize GLM client: {str(e)}")
            self.client = None

    def generate_response(
        self,
        system_prompt: str,
        user_prompt: str,
        context: Optional[str] = None,
        conversation_history: Optional[List[Dict]] = None,
    ) -> LLMResponse:
        """Generate response using GLM API"""
        if not self.client:
            raise RuntimeError("GLM client not initialized")

        try:
            messages = self._build_messages(system_prompt, user_prompt, context, conversation_history)
            
            logger.info(f"Calling GLM API with {self.config.model_name} ({len(messages)} messages)")
            
            response = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p
            )
            
            response_text = response.choices[0].message.content
            response_text = response_text.strip() if response_text else ""

            if len(response_text) < 5:
                response_text = self._generate_fallback_response(system_prompt, user_prompt, context)

            return LLMResponse(
                content=response_text,
                model=self.config.model_name,
                provider=self.provider_name,
                metadata={"api": "glm"}
            )

        except Exception as e:
            logger.error(f"Error calling GLM API: {str(e)}")
            fallback = self._generate_fallback_response(system_prompt, user_prompt, context)
            return LLMResponse(
                content=fallback,
                model=self.config.model_name,
                provider=self.provider_name,
                metadata={"error": str(e), "fallback": True}
            )


def create_llm_provider(config: LLMConfig) -> BaseLLMProvider:
    """Factory function to create the appropriate LLM provider"""
    provider_map = {
        LLMProvider.HUGGINGFACE: HuggingFaceProvider,
        LLMProvider.OPENAI: OpenAIProvider,
        LLMProvider.ANTHROPIC: AnthropicProvider,
        LLMProvider.DEEPSEEK: DeepSeekProvider,
        LLMProvider.GROK: GrokProvider,
        LLMProvider.KIMI: KimiProvider,
        LLMProvider.MINIMAX: MinimaxProvider,
        LLMProvider.QWEN: QwenProvider,
        LLMProvider.GLM: GLMProvider,
    }
    
    provider_class = provider_map.get(config.provider)
    if not provider_class:
        raise ValueError(f"Unsupported LLM provider: {config.provider}")
    
    return provider_class(config)