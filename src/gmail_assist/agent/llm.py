"""Chat models for the graph, built on demand from "<provider>:<model>" specs.

Which model handles a message is decided inside the graph (see
nodes.choose_model); .env only says how to reach each provider:

    ANTHROPIC_API_KEY=sk-ant-...                          # enables anthropic:* models
    GMAIL_ASSIST_LLM_BASE_URL=http://localhost:11434/v1   # enables local:* models (Ollama here;
                                                          # llama-server/vLLM use their own port)

Specs look like "local:qwen3:8b" or "anthropic:claude-opus-5-5".
"""
import os
from dataclasses import dataclass, field

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field

PROVIDERS = ('local', 'anthropic')
# Claude thinks before answering and that counts toward max_tokens, so leave
# room beyond the few dozen tokens a Classification needs.
CLAUDE_MAX_TOKENS = 8000


class Classification(BaseModel):
    # TODO: tune these descriptions -- the model reads them as instructions.
    label: str = Field(description='The Gmail label this email belongs under, e.g. "Finance/Receipts".')
    is_new_label: bool = Field(description='True if no existing label fits and this is a new sub-label.')
    confidence: float = Field(ge=0, le=1, description='How sure you are, from 0 to 1.')
    reason: str = Field(description='One short sentence explaining the choice.')


@dataclass
class LLM:
    chat: BaseChatModel
    provider: str
    model: str

    @property
    def name(self) -> str:
        """Recorded on each decision, so you can compare models later."""
        return f'{self.provider}:{self.model}'

    def structured(self, schema: type[BaseModel]) -> Runnable:
        """The chat model constrained to return `schema`.

        Claude uses its native structured outputs: LangChain's default
        ('function_calling') forces a tool call, which current Claude models
        reject. Local servers keep langchain-openai's default (JSON schema
        response_format); if yours doesn't support it, try
        method='function_calling' here.
        """
        if self.provider == 'anthropic':
            return self.chat.with_structured_output(schema, method='json_schema')
        return self.chat.with_structured_output(schema)


def _local(model: str, effort: str | None) -> BaseChatModel:
    from langchain_openai import ChatOpenAI

    # Local servers ignore the key, but the client requires one. `effort` is
    # Claude-only and ignored here.
    return ChatOpenAI(base_url=os.environ['GMAIL_ASSIST_LLM_BASE_URL'], model=model,
                      api_key='local', temperature=0)


def _anthropic(model: str, effort: str | None) -> BaseChatModel:
    from langchain_anthropic import ChatAnthropic

    # No temperature: current Claude models reject sampling parameters.
    return ChatAnthropic(
        model=model,
        max_tokens=CLAUDE_MAX_TOKENS,
        output_config={'effort': effort} if effort else None,
    )


_BUILDERS = {'local': _local, 'anthropic': _anthropic}
_CREDENTIALS = {'local': 'GMAIL_ASSIST_LLM_BASE_URL', 'anthropic': 'ANTHROPIC_API_KEY'}


@dataclass
class Models:
    """Builds and caches chat models so each message doesn't open a new client."""
    _cache: dict[tuple[str, str | None], LLM] = field(default_factory=dict)

    @staticmethod
    def configured(provider: str) -> bool:
        """Whether .env has what this provider needs."""
        return bool(os.environ.get(_CREDENTIALS.get(provider, '')))

    def get(self, spec: str, effort: str | None = None) -> LLM:
        """`effort` (low | medium | high | xhigh | max) applies to Claude only."""
        key = (spec, effort)
        if key not in self._cache:
            # Split on the first colon only: Ollama tags contain one ("qwen3:8b").
            provider, _, model = spec.partition(':')
            if provider not in _BUILDERS or not model:
                raise ValueError(f'bad model spec {spec!r}; use "<{"|".join(PROVIDERS)}>:<model>"')
            if not self.configured(provider):
                raise ValueError(f'{spec} needs {_CREDENTIALS[provider]} in .env')
            self._cache[key] = LLM(_BUILDERS[provider](model, effort), provider, model)
        return self._cache[key]
