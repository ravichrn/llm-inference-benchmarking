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
