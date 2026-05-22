#!/usr/bin/env bash
set -euo pipefail

OLLAMA_MODEL="${OLLAMA_MODEL:-gemma4:latest}"
OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://127.0.0.1:11434}"
OLLAMA_LOG_DIR="${HOME}/.ollama"
OLLAMA_LOG_FILE="${OLLAMA_LOG_DIR}/swarmgpt-ollama.log"

log() {
    printf '[ollama-setup] %s\n' "$*"
}

command_exists() {
    command -v "$1" >/dev/null 2>&1
}

ollama_is_running() {
    curl -fsS "${OLLAMA_BASE_URL}/api/tags" >/dev/null 2>&1
}

wait_for_ollama() {
    for _ in {1..30}; do
        if ollama_is_running; then
            return 0
        fi
        sleep 1
    done
    return 1
}

install_ollama() {
    if command_exists ollama; then
        log "Ollama is already installed."
        return
    fi

    case "$(uname -s)" in
        Linux)
            if ! command_exists curl; then
                log "curl is required to install Ollama. Install curl and rerun this command."
                exit 1
            fi
            log "Installing Ollama with the official install script..."
            curl -fsSL https://ollama.com/install.sh | sh
            ;;
        Darwin)
            if ! command_exists brew; then
                log "Homebrew is required for automatic Ollama installation on macOS."
                log "Install Homebrew from https://brew.sh or install Ollama from https://ollama.com/download, then rerun this command."
                exit 1
            fi
            log "Installing Ollama with Homebrew..."
            brew install ollama
            ;;
        *)
            log "Unsupported OS: $(uname -s). Install Ollama manually from https://ollama.com/download, then rerun this command."
            exit 1
            ;;
    esac
}

start_ollama() {
    if ollama_is_running; then
        log "Ollama is already running."
        return
    fi

    mkdir -p "${OLLAMA_LOG_DIR}"

    case "$(uname -s)" in
        Linux)
            if command_exists systemctl && systemctl list-unit-files ollama.service >/dev/null 2>&1; then
                log "Starting Ollama with systemd..."
                sudo systemctl start ollama || true
            fi
            ;;
        Darwin)
            if command_exists brew; then
                log "Starting Ollama with Homebrew services..."
                brew services start ollama || true
            fi
            ;;
    esac

    if ! wait_for_ollama; then
        log "Starting Ollama in the background..."
        nohup ollama serve >"${OLLAMA_LOG_FILE}" 2>&1 &
    fi

    if ! wait_for_ollama; then
        log "Ollama did not become ready. Check ${OLLAMA_LOG_FILE} for details."
        exit 1
    fi

    log "Ollama is running at ${OLLAMA_BASE_URL}."
}

pull_model() {
    if ollama list | awk 'NR > 1 {print $1}' | grep -Fxq "${OLLAMA_MODEL}"; then
        log "Model ${OLLAMA_MODEL} is already installed."
        return
    fi

    log "Pulling Ollama model ${OLLAMA_MODEL}..."
    ollama pull "${OLLAMA_MODEL}"
}

install_ollama
start_ollama
pull_model

log "Done. Select Ollama (local) in the SwarmGPT UI and use ${OLLAMA_MODEL}."
