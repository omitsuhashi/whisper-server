PYTHON ?= python3
UVICORN := $(PYTHON) -m uvicorn
HTTP_HOST ?= 127.0.0.1
HTTP_PORT ?= 8000
HTTP_RELOAD ?= 0
LOG_LEVEL ?= DEBUG

.PHONY: help http test shell

help:
	@echo "Available targets:"
	@echo "  make http                     # FastAPI起動"
	@echo "  make test                     # unit tests"
	@echo "  make shell                    # .venv でシェル起動"

http:
	@RELOAD_FLAG=""; \
	if [ "$(HTTP_RELOAD)" != "0" ]; then RELOAD_FLAG="--reload"; fi; \
	LOWER_LOG_LEVEL=$$(printf '%s' "$(LOG_LEVEL)" | tr '[:upper:]' '[:lower:]'); \
	ENVV="LOG_LEVEL=$(LOG_LEVEL)"; \
	if [ "$(MEM_DIAG)" = "1" ]; then ENVV="$$ENVV MEM_DIAG=1"; fi; \
	echo "env $$ENVV $(UVICORN) src.cmd.http:create_app $$RELOAD_FLAG --host $(HTTP_HOST) --port $(HTTP_PORT) --factory --log-level $$LOWER_LOG_LEVEL"; \
	env $$ENVV $(UVICORN) src.cmd.http:create_app $$RELOAD_FLAG --host $(HTTP_HOST) --port $(HTTP_PORT) --factory --log-level $$LOWER_LOG_LEVEL

test:
	$(PYTHON) -m unittest discover -s tests

shell:
	@if [ ! -d ".venv" ]; then \
		echo ".venv ディレクトリが見つかりません。作成するには 'python3 -m venv .venv' を実行してください。"; \
		exit 1; \
	fi
	@echo "source .venv/bin/activate"
	@SHELL=$$SHELL; \
	if [ -z "$$SHELL" ]; then SHELL=/bin/bash; fi; \
	$$SHELL --login -i -c "source .venv/bin/activate && exec $$SHELL"
