from pathlib import Path


def test_native_makefile_test_is_independent_of_project_installation():
    makefile = Path("native/Makefile").read_text()

    assert "PYTHON ?= python3" in makefile
    assert "ctypes.CDLL('./libsensetrace_measurement.so')" in makefile
    assert "from sensetrace" not in makefile
