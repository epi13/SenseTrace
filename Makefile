PYTHON ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)

test-bounded:
	$(PYTHON) tools/test_bounded.py

native-test:
	$(MAKE) -C native test

.PHONY: test-bounded native-test
