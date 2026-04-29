import os
import sqlite3
from pathlib import Path

from llm_inference_benchmarking.types import GatewayDecision, GatewayUsage


def get_ledger_db_path() -> Path:
    """SQLite path for usage ledger (standalone default under user home)."""
    env = os.getenv("GATEWAY_LEDGER_DB", "").strip()
    if env:
        return Path(env).expanduser()
    base = Path.home() / ".llm_inference_benchmarking"
    base.mkdir(parents=True, exist_ok=True)
    return base / "gateway_usage.db"


def init_ledger() -> None:
    db_path = get_ledger_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS gateway_usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                tier TEXT NOT NULL,
                backend TEXT NOT NULL,
                model TEXT NOT NULL,
                input_tokens INTEGER NOT NULL,
                output_tokens INTEGER NOT NULL,
                total_tokens INTEGER NOT NULL,
                latency_ms INTEGER NOT NULL,
                estimated_cost_usd REAL NOT NULL,
                ok INTEGER NOT NULL,
                error TEXT,
                request_id TEXT NOT NULL DEFAULT ''
            )
            """
        )
        try:
            conn.execute(
                "ALTER TABLE gateway_usage ADD COLUMN request_id TEXT NOT NULL DEFAULT ''"
            )
        except sqlite3.OperationalError:
            pass
        conn.commit()


_ledger_initialized = False


def _ensure_ledger() -> None:
    global _ledger_initialized
    if not _ledger_initialized:
        init_ledger()
        _ledger_initialized = True


def log_usage(
    decision: GatewayDecision,
    usage: GatewayUsage,
    ok: bool,
    error: str = "",
    request_id: str = "",
) -> None:
    _ensure_ledger()
    db_path = get_ledger_db_path()
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO gateway_usage (
                tier, backend, model, input_tokens, output_tokens, total_tokens,
                latency_ms, estimated_cost_usd, ok, error, request_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                decision.tier,
                decision.backend,
                decision.model,
                usage.input_tokens,
                usage.output_tokens,
                usage.total_tokens,
                usage.latency_ms,
                usage.estimated_cost_usd,
                1 if ok else 0,
                error[:1000],
                request_id,
            ),
        )
        conn.commit()
