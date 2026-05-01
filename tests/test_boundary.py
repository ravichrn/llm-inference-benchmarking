"""Enforce that the gateway library has no coupling to application-layer modules."""

from pathlib import Path

_FORBIDDEN = (
    "from agents",
    "import agents",
    "from ingestion",
    "import ingestion",
    "from databases",
    "import databases",
    "from guardrails",
    "import guardrails",
)

_SRC = Path(__file__).resolve().parent.parent / "src" / "llm_inference_benchmarking"


def test_library_has_no_app_coupling_strings():
    py_files = list(_SRC.glob("*.py"))
    assert py_files, f"No Python files found in {_SRC}"
    combined = "\n".join(f.read_text() for f in py_files)
    violations = [t for t in _FORBIDDEN if t in combined]
    assert not violations, "Library must not reference application modules:\n" + "\n".join(
        violations
    )
