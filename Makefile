PYTHON ?= python3
CLI := $(PYTHON) -m src.cmd.cli
UVICORN := $(PYTHON) -m uvicorn
HTTP_HOST ?= 127.0.0.1
HTTP_PORT ?= 8000
HTTP_RELOAD ?= 1

ifeq ($(firstword $(MAKECMDGOALS)),cli-files)
  CLI_FILES_EXTRA := $(wordlist 2,$(words $(MAKECMDGOALS)),$(MAKECMDGOALS))
  ifndef AUDIO
    AUDIO := $(firstword $(CLI_FILES_EXTRA))
    CLI_FILES_EXTRA := $(wordlist 2,$(words $(CLI_FILES_EXTRA)),$(CLI_FILES_EXTRA))
  endif
  CLI_FILES_ARGS := $(CLI_FILES_EXTRA)
  $(foreach extra,$(CLI_FILES_EXTRA),$(eval $(extra):;@:))
endif

ifeq ($(firstword $(MAKECMDGOALS)),cli-frames)
  CLI_FRAMES_EXTRA := $(wordlist 2,$(words $(MAKECMDGOALS)),$(MAKECMDGOALS))
  ifndef VIDEO
    VIDEO := $(firstword $(CLI_FRAMES_EXTRA))
    CLI_FRAMES_EXTRA := $(wordlist 2,$(words $(CLI_FRAMES_EXTRA)),$(CLI_FRAMES_EXTRA))
  endif
  CLI_FRAMES_ARGS := $(CLI_FRAMES_EXTRA)
  $(foreach extra,$(CLI_FRAMES_EXTRA),$(eval $(extra):;@:))
  $(if $(VIDEO),$(eval $(VIDEO):;@:))
endif

.PHONY: help cli cli-help cli-files cli-frames cli-stream audio-streaming http test shell

help:
	@echo "Available targets:"
	@echo "  make cli-help                 # CLI のヘルプを表示"
	@echo "  make cli-files AUDIO=\"file.wav\" [MODEL=...] [LANGUAGE=...] [TASK=...]"
	@echo "     または make -- cli-files file.wav [--diarize] [--show-segments]"
	@echo "                               # ファイル入力で書き起こし"
	@echo "  make cli-frames VIDEO=\"video.mp4\" [OUTPUT_DIR=...] [MAX_FRAMES=...]"
	@echo "     または make -- cli-frames video.mp4 [--output-dir snapshots] [--max-frames 10]"
	@echo "                               # 動画から代表フレームを抽出"
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
	EXTRA="$(CLI_FILES_ARGS)"; \
	echo "$(CLI) files $$FLAGS $(AUDIO) $$EXTRA"; \
	$(CLI) files $$FLAGS $(AUDIO) $$EXTRA

cli-frames:
	@if [ -z "$(VIDEO)" ]; then \
		echo "Usage: make $@ VIDEO=\"path/to/video.mp4\" [OUTPUT_DIR=...] [MAX_FRAMES=...]"; \
		echo "       または make -- $@ video.mp4 [--output-dir snapshots] [--max-frames 10]"; \
		exit 1; \
	fi
	@FLAGS=""; \
	if [ -n "$(OUTPUT_DIR)" ]; then FLAGS="$$FLAGS --output-dir $(OUTPUT_DIR)"; fi; \
	if [ -n "$(MIN_SCENE_SPAN)" ]; then FLAGS="$$FLAGS --min-scene-span $(MIN_SCENE_SPAN)"; fi; \
	if [ -n "$(DIFF_THRESHOLD)" ]; then FLAGS="$$FLAGS --diff-threshold $(DIFF_THRESHOLD)"; fi; \
	if [ -n "$(MAX_FRAMES)" ]; then FLAGS="$$FLAGS --max-frames $(MAX_FRAMES)"; fi; \
	if [ -n "$(IMAGE_FORMAT)" ]; then FLAGS="$$FLAGS --image-format $(IMAGE_FORMAT)"; fi; \
	if [ -n "$(OVERWRITE)" ]; then \
		case "$(OVERWRITE)" in \
			1|true|TRUE|yes|YES) FLAGS="$$FLAGS --overwrite" ;; \
		esac; \
	fi; \
	EXTRA="$(CLI_FRAMES_ARGS)"; \
	echo "$(CLI) frames $$FLAGS $(VIDEO) $$EXTRA"; \
	$(CLI) frames $$FLAGS $(VIDEO) $$EXTRA
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
