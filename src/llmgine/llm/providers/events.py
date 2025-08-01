from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from openai.types.chat import ChatCompletion

from llmgine.llm.providers.providers import Providers
from llmgine.messages.events import Event


@dataclass
class LLMResponseEvent(Event):
    call_id: str = ""
    raw_response: Optional[ChatCompletion] = None
    error: Optional[Exception] = None


@dataclass
class LLMCallEvent(Event):
    model_id: str = ""
    call_id: str = ""
    provider: Optional[Providers] = None
    payload: Dict[str, Any] = field(default_factory=dict)
