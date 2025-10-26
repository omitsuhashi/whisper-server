PYTHON ?= python3
CLI := $(PYTHON) -m src.cmd.cli
UVICORN := $(PYTHON) -m uvicorn
HTTP_HOST ?= 127.0.0.1
HTTP_PORT ?= 8000
HTTP_RELOAD ?= 1

.PHONY: help cli cli-help cli-files cli-stream audio-streaming http test shell

help:
	@echo "Available targets:"
	@echo "  make cli-help                 # CLI のヘルプを表示"
	@echo "  make cli-files AUDIO=\"file.wav\" [MODEL=...] [LANGUAGE=...] [TASK=...]"
	@echo "                               # ファイル入力で書き起こし"
	@echo "  make cli-stream [MODEL=...] [LANGUAGE=...] [TASK=...] [NAME=...] [STREAM_INTERVAL=...]"
	@echo "                               # 標準入力経由でストリーム書き起こし (音声バイトをパイプで供給すること)"
	@echo "  make audio-streaming [DEVICE=idx] [SECONDS=] [MODEL=...] [LANGUAGE=...] [TASK=...] [NAME=...] [STREAM_INTERVAL=...]"
	@echo "                               # ffmpeg でマイク録音→CLI ストリームへパイプ"
	@echo "  make http [HTTP_HOST=...] [HTTP_PORT=...] [HTTP_RELOAD=0|1]"
	@echo "                               # FastAPI サーバーを uvicorn で起動"

cli:
	$(CLI)

cli-help:
	$(CLI) --help

cli-files:
	@if [ -z "$(AUDIO)" ]; then \
		echo "Usage: make $@ AUDIO=\"path/to/audio.wav [more.wav]\" [MODEL=...] [LANGUAGE=...] [TASK=...]"; \
		exit 1; \
	fi
	@FLAGS=""; \
	if [ -n "$(MODEL)" ]; then FLAGS="$$FLAGS --model $(MODEL)"; fi; \
	if [ -n "$(LANGUAGE)" ]; then FLAGS="$$FLAGS --language $(LANGUAGE)"; fi; \
	if [ -n "$(TASK)" ]; then FLAGS="$$FLAGS --task $(TASK)"; fi; \
	echo "$(CLI) files $$FLAGS $(AUDIO)"; \
	$(CLI) files $$FLAGS $(AUDIO)
cli-stream:
	@FLAGS=""; \
	if [ -n "$(MODEL)" ]; then FLAGS="$$FLAGS --model $(MODEL)"; fi; \
	if [ -n "$(LANGUAGE)" ]; then FLAGS="$$FLAGS --language $(LANGUAGE)"; fi; \
	if [ -n "$(TASK)" ]; then FLAGS="$$FLAGS --task $(TASK)"; fi; \
	if [ -n "$(STREAM_INTERVAL)" ]; then FLAGS="$$FLAGS --stream-interval $(STREAM_INTERVAL)"; fi; \
	if [ -n "$(NAME)" ]; then FLAGS="$$FLAGS --name $(NAME)"; fi; \
	echo "$(CLI) stream $$FLAGS"; \
	$(CLI) stream $$FLAGS

audio-streaming:
	@DEVICE_VAL="$(if $(DEVICE),$(DEVICE),:1)"; \
	SECONDS_OPT=""; \
	if [ -n "$(SECONDS)" ]; then SECONDS_OPT="-t $(SECONDS)"; fi; \
	FLAGS=""; \
	if [ -n "$(MODEL)" ]; then FLAGS="$$FLAGS --model $(MODEL)"; fi; \
	if [ -n "$(LANGUAGE)" ]; then FLAGS="$$FLAGS --language $(LANGUAGE)"; fi; \
	if [ -n "$(TASK)" ]; then FLAGS="$$FLAGS --task $(TASK)"; fi; \
	if [ -n "$(NAME)" ]; then FLAGS="$$FLAGS --name $(NAME)"; fi; \
	if [ -n "$(STREAM_INTERVAL)" ]; then FLAGS="$$FLAGS --stream-interval $(STREAM_INTERVAL)"; fi; \
	echo "ffmpeg -hide_banner -loglevel error -f avfoundation -i $${DEVICE_VAL} -ac 1 -ar 16000 $$SECONDS_OPT -f wav - | $(CLI) stream $$FLAGS"; \
	ffmpeg -hide_banner -loglevel error -f avfoundation -i "$${DEVICE_VAL}" -ac 1 -ar 16000 $$SECONDS_OPT -f wav - \
		| $(CLI) stream $$FLAGS

http:
	@RELOAD_FLAG=""; \
	if [ "$(HTTP_RELOAD)" != "0" ]; then RELOAD_FLAG="--reload"; fi; \
	echo "$(UVICORN) src.cmd.http:create_app $$RELOAD_FLAG --host $(HTTP_HOST) --port $(HTTP_PORT)"; \
	$(UVICORN) src.cmd.http:create_app $$RELOAD_FLAG --host $(HTTP_HOST) --port $(HTTP_PORT) --factory

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
