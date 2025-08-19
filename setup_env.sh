#!/bin/bash

# ✨ Скрипт для автоматической настройки окружения для эксперимента ✨
#
# Этот Shell-скрипт (для Linux/Ubuntu) подготавливает рабочее окружение для
# экспериментального проекта по тестированию производительности распознавания лиц.
#
# Его главная задача — корректно скомпилировать библиотеку dlib из исходного кода
# с поддержкой NVIDIA CUDA. Это ключевой шаг для использования мощностей GPU,
# что позволяет значительно ускорить процесс анализа видео.
#
# Данный скрипт является частью технического стенда и подготавливает среду для
# тестирования производительности, а не для создания работающей системы мониторинга.
#
# Функционал:
# - Проверка наличия системных зависимостей, необходимых для компиляции.
# - Создание изолированного виртуального окружения Python во избежание конфликтов.
# - Установка необходимых библиотек из requirements.txt.
# - Принудительная компиляция dlib с флагом для активации поддержки CUDA.
#
# Предварительные требования:
# Перед запуском необходимо установить NVIDIA CUDA Toolkit и cuDNN согласно
# официальным инструкциям NVIDIA для вашей системы.
#
# Автор: Михаил Шардин https://shardin.name/
# Дата создания: 19.08.2025
# Версия: 1.0
#
# Актуальная версия скрипта всегда здесь: https://github.com/empenoso/doorcam-face-report/
#

set -e

VENV_DIR=".venv"
PYTHON_VERSION_TARGET="3.11.9"

echo "=== Установка окружения и зависимостей ==="

# --- Проверка наличия pyenv ---
if ! command -v pyenv &> /dev/null; then
    echo -e "\n\033[1;31m[ERROR] pyenv не найден. Установи pyenv перед запуском.\033[0m"
    exit 1
fi

echo -e "\n[INFO] Выбор версии Python $PYTHON_VERSION_TARGET через pyenv..."
pyenv local $PYTHON_VERSION_TARGET
echo "[INFO] Текущая версия Python: $(python --version)"

# --- Проверка системных библиотек ---
echo -e "\n[INFO] Проверка и установка системных библиотек для dlib..."
sudo apt update
sudo apt install -y build-essential cmake libopenblas-dev liblapack-dev libjpeg-dev git

# --- Очистка и создание виртуального окружения ---
if [ -d "$VENV_DIR" ]; then
    echo "[INFO] Удаление старого виртуального окружения '$VENV_DIR'..."
    rm -rf "$VENV_DIR"
fi

echo "[INFO] Создание виртуального окружения '$VENV_DIR'..."
python -m venv "$VENV_DIR"

echo "[INFO] Активация окружения..."
source "$VENV_DIR/bin/activate"

echo "[INFO] Установка Python-зависимостей..."
pip install --upgrade pip
pip install -r requirements.txt

echo "[INFO] Установка dlib с поддержкой CUDA..."
pip install dlib \
    --no-binary :all: \
    --verbose \
    --config-settings="cmake.args=-DDLIB_USE_CUDA=1"

echo "[INFO] Установка face_recognition..."
pip install face_recognition

echo -e "\n\033[1;32m[OK] Окружение и зависимости успешно установлены.\033[0m"