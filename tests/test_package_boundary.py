from pathlib import Path

import llm_inference_benchmarking

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


def test_package_has_no_app_coupling_strings():
    pkg_dir = Path(llm_inference_benchmarking.__file__).resolve().parent
    py_files = sorted(pkg_dir.glob("*.py"))
    assert py_files

    violations: list[str] = []
    for file_path in py_files:
        text = file_path.read_text()
        for token in _FORBIDDEN:
            if token in text:
                violations.append(f"{file_path.name}: {token}")

    assert not violations, "Package must not reference application modules:\n" + "\n".join(
        violations
    )
