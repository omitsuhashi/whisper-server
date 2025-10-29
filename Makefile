PYTHON ?= python3
CLI := $(PYTHON) -m src.cmd.cli
UVICORN := $(PYTHON) -m uvicorn
HTTP_HOST ?= 127.0.0.1
HTTP_PORT ?= 8000
HTTP_RELOAD ?= 0
LOG_LEVEL ?= INFO
LLM_POLISH_MODEL ?= mlx-community/Qwen3-1.7B-MLX-MXFP4

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

define RUN_AUDIO_STREAM
	@FLAGS=""; \
	if [ -n "$(MODEL)" ]; then FLAGS="$$FLAGS --model $(MODEL)"; fi; \
	if [ -n "$(LANGUAGE)" ]; then FLAGS="$$FLAGS --language $(LANGUAGE)"; fi; \
	if [ -n "$(TASK)" ]; then FLAGS="$$FLAGS --task $(TASK)"; fi; \
	if [ -n "$(STREAM_INTERVAL)" ]; then FLAGS="$$FLAGS --stream-interval $(STREAM_INTERVAL)"; fi; \
	if [ -n "$(NAME)" ]; then FLAGS="$$FLAGS --name $(NAME)"; fi; \
	FLAGS="$$FLAGS $(1)"; \
	if ! command -v ffmpeg >/dev/null 2>&1; then \
		echo "ffmpeg が見つかりません。'brew install ffmpeg' 等でインストールしてください。" >&2; \
		exit 1; \
	fi; \
	OS_NAME=$$(uname 2>/dev/null || echo Unknown); \
	case "$$OS_NAME" in \
		Darwin) DEFAULT_DEVICE=":1"; INPUT_FMT="avfoundation" ;; \
		Linux) DEFAULT_DEVICE="default"; INPUT_FMT="alsa" ;; \
		*) DEFAULT_DEVICE=""; INPUT_FMT="" ;; \
	esac; \
	if [ -n "$(DEVICE)" ]; then DEVICE_VAL="$(DEVICE)"; else DEVICE_VAL="$$DEFAULT_DEVICE"; fi; \
	if [ -n "$(AUDIO_FORMAT)" ]; then INPUT_FMT="$(AUDIO_FORMAT)"; fi; \
	if [ -z "$$DEVICE_VAL" ] || [ -z "$$INPUT_FMT" ]; then \
		echo "音声デバイスまたはフォーマットを決定できません。DEVICE と AUDIO_FORMAT を指定してください。" >&2; \
		exit 1; \
	fi; \
	if [ -n "$(SECONDS)" ]; then SECONDS_OPT="-t $(SECONDS)"; else SECONDS_OPT=""; fi; \
	echo "ffmpeg -hide_banner -loglevel error -f $$INPUT_FMT -i \"$$DEVICE_VAL\" -ac 1 -ar 16000 $$SECONDS_OPT -f wav - | $(CLI) stream $$FLAGS"; \
	ffmpeg -hide_banner -loglevel error -f "$$INPUT_FMT" -i "$$DEVICE_VAL" -ac 1 -ar 16000 $$SECONDS_OPT -f wav - | $(CLI) stream $$FLAGS
endef

.PHONY: help cli cli-help cli-files cli-frames cli-stream audio-stream audio-stream-polish audio-devices http test shell

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
	@echo "  make audio-stream [DEVICE=...] [SECONDS=...] [MODEL=...] [LANGUAGE=...] [TASK=...] [NAME=...] [STREAM_INTERVAL=...]"
	@echo "                               # マイク録音→CLI ストリーム (OS ごとに自動判定)"
	@echo "  make audio-stream-polish [DEVICE=...] [SECONDS=...] [MODEL=...] [LANGUAGE=...] [TASK=...] [NAME=...]"
	@echo "                               # マイク録音→CLI ストリーム (書き起こし後に校正、テキストのみ出力)"
	@echo "  make audio-devices           # 利用可能な録音デバイス一覧を表示"
	@echo "  make http [HTTP_HOST=...] [HTTP_PORT=...] [HTTP_RELOAD=1] [LOG_LEVEL=DEBUG] [MEM_DIAG=1]"
	@echo "                               # FastAPI サーバーを uvicorn で起動 (MEM_DIAG=1 でメモリ診断ログを有効化)"
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

audio-stream:
	$(call RUN_AUDIO_STREAM,)

audio-stream-polish:
	$(call RUN_AUDIO_STREAM,--polish --plain-text)

audio-devices:
	@OS_NAME=$$(uname 2>/dev/null || echo Unknown); \
	if [ "$$OS_NAME" = "Darwin" ]; then \
		if ! command -v ffmpeg >/dev/null 2>&1; then \
			echo "ffmpeg が見つかりません。'brew install ffmpeg' 等でインストールしてください。" >&2; exit 1; \
		fi; \
		echo "[macOS] avfoundation の入力デバイス"; \
		ffmpeg -hide_banner -f avfoundation -list_devices true -i "" 2>&1 | sed 's/^/  /'; \
	elif [ "$$OS_NAME" = "Linux" ]; then \
		if command -v arecord >/dev/null 2>&1; then \
			echo "[Linux] ALSA デバイス"; \
			arecord -l; \
		else \
			echo "arecord が見つかりません。'sudo apt install alsa-utils' 等でインストールしてください。" >&2; exit 1; \
		fi; \
	else \
		echo "この OS ($$OS_NAME) のデバイス列挙方法が未対応です。手動で確認してください。" >&2; exit 1; \
	fi

http:
	@RELOAD_FLAG=""; \
	if [ "$(HTTP_RELOAD)" != "0" ]; then RELOAD_FLAG="--reload"; fi; \
	LOWER_LOG_LEVEL=$$(printf '%s' "$(LOG_LEVEL)" | tr '[:upper:]' '[:lower:]'); \
	ENVV="LOG_LEVEL=$(LOG_LEVEL) LLM_POLISH_MODEL=$(LLM_POLISH_MODEL)"; \
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
