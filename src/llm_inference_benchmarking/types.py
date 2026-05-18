import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class GatewayRequest:
    prompt: Any
    role: str = "agent"
    tier: str = "auto"
    metadata: dict[str, Any] = field(default_factory=dict)
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class GatewayDecision:
    tier: str
    backend: str
    model: str
    reason: str


@dataclass
class GatewayUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0
    latency_ms: int = 0


@dataclass
class GatewayResult:
    content: Any
    backend: str
    model: str
    tier: str
    usage: GatewayUsage
    raw: Any = None


@dataclass
class ABVariantResult:
    tier: str
    model: str
    avg_score: float
    avg_latency_ms: float
    total_cost_usd: float


@dataclass
class ABResult:
    variant_a: ABVariantResult
    variant_b: ABVariantResult
    win_rate_a: float  # fraction of prompts where A scored strictly higher than B
    n_prompts: int
    judge_model: str
