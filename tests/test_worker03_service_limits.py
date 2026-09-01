from pathlib import Path


def test_system_runner_has_a_bounded_memlock_allowance():
    unit = Path("infra/worker03/sensetrace.service").read_text()

    assert "LimitMEMLOCK=64M" in unit
