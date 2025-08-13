#!/bin/bash

# ✨ Автоматическая настройка окружения для распознавания лиц ✨
#
# Этот Shell-скрипт (для Linux/Ubuntu) автоматизирует сложный процесс настройки рабочего окружения
# для проекта по распознаванию лиц. Он решает главную проблему, описанную в статье: 
# компилирует библиотеку dlib с поддержкой CUDA, что позволяет использовать мощность
# видеокарты (GPU) для ускорения анализа видео в десятки раз.
#
# Функционал:
# - Проверка наличия pyenv для управления версиями Python
# - Установка и активация нужной версии Python (3.11.9) во избежание конфликтов
# - Установка системных зависимостей, необходимых для компиляции (cmake, build-essential и т.д.)
# - Создание чистого и изолированного виртуального окружения (.venv)
# - Установка Python-библиотек из файла requirements.txt
# - Принудительная компиляция библиотеки dlib из исходного кода с включенной поддержкой CUDA
# - Установка библиотеки face_recognition после успешной сборки dlib
#
# Предварительные требования: Перед запуском этого скрипта необходимо установить 
# NVIDIA CUDA Toolkit и cuDNN согласно официальным инструкциям NVIDIA.
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