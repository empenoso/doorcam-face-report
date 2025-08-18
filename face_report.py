# face_report.py

# 🤖🔎 Анализ видео, распознавание лиц и создание отчета 🤖🔎
#
# Это главный Python-скрипт проекта, его "мозг". Он анализирует видеозаписи с камер,
# находит на них лица людей, сравнивает их между собой для выявления уникальных личностей
# и подсчитывает, как часто каждый человек появлялся. Результаты своей работы скрипт
# оформляет в виде наглядного и удобного HTML-файла с фотографиями всех найденных людей.
#
# Функционал:
# - Прием параметров из командной строки: папка с видео, модель детекции (CNN/HOG) и др.
# - Автоматический поиск всех видеофайлов (.mp4) в указанной директории
# - Последовательная обработка каждого видеофайла с пропуском кадров для ускорения
# - Детекция лиц на кадрах с использованием быстрой HOG или точной нейросетевой CNN-модели
# - Преобразование каждого лица в уникальный "цифровой отпечаток" (face encoding)
# - Сравнение "отпечатков" для идентификации уникальных людей и отслеживания их появлений
# - "Умный" подсчет: каждая личность учитывается только один раз за один видеофайл
# - Создание качественных иконок-портретов для отчета с автоматическим кадрированием
# - Генерация итогового HTML-отчета с отсортированным списком личностей по частоте появлений
#
# Установка зависимостей перед использованием: Запустить скрипт setup_env.sh,
# который установит все необходимое, включая dlib с поддержкой CUDA.
#
# Автор: Михаил Шардин https://shardin.name/
# Дата создания: 19.08.2025
# Версия: 1.0
#
# Актуальная версия скрипта всегда здесь: https://github.com/empenoso/doorcam-face-report/
#

import argparse
import base64
import cv2
import face_recognition
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys
import os

# --- КОНФИГУРАЦИЯ (значения по умолчанию) ---
FACE_COMPARISON_TOLERANCE = 0.6
THUMBNAIL_SIZE = (150, 150)

def create_thumbnail(frame, location):
    """
    Создает иконку лица с увеличенными отступами.
    """
    top, right, bottom, left = location
    
    # Увеличиваем вертикальный и горизонтальный отступы с 25% до 50% от размера лица.
    # Это позволяет захватить больше пространства вокруг лица (волосы, лоб, подбородок).
    pad_y = (bottom - top) // 2 
    pad_x = (right - left) // 2

    # Применяем отступы, но следим, чтобы не выйти за границы кадра
    top = max(0, top - pad_y)
    bottom = min(frame.shape[0], bottom + pad_y)
    left = max(0, left - pad_x)
    right = min(frame.shape[1], right + pad_x)
    
    face_image = frame[top:bottom, left:right]
    
    # Проверка, не является ли вырезанное изображение пустым
    if face_image.size == 0:
        return None

    resized_face = cv2.resize(face_image, THUMBNAIL_SIZE)
    is_success, buffer = cv2.imencode(".jpg", resized_face)
    
    if not is_success:
        return None
        
    return base64.b64encode(buffer).decode("utf-8")

def generate_html_report(face_metadata, output_path):
    """
    Генерирует HTML-отчет со списком найденных лиц.
    """
    if not face_metadata:
        print("Не найдено ни одного лица для создания отчета.")
        return

    # Сортировка личностей по количеству появлений (по убыванию)
    sorted_metadata = sorted(face_metadata, key=lambda x: x['count'], reverse=True)

    # Сначала определяем шаблон с именованным плейсхолдером {total_count}
    # и {person_cards}. CSS-стили теперь не будут конфликтовать с форматированием.
    html_template = """
    <!DOCTYPE html>
    <html lang="ru">
    <head>
        <meta charset="UTF-8">
        <title>Отчет по распознаванию лиц</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f4f4f9; }}
            h1 {{ text-align: center; color: #333; }}
            h2 {{ text-align: center; color: #666; font-weight: normal; margin-top: -10px; }}
            .container {{ display: flex; flex-wrap: wrap; gap: 20px; justify-content: center; }}
            .person-card {{
                background-color: white;
                border: 1px solid #ddd;
                border-radius: 8px;
                padding: 15px;
                text-align: center;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                transition: transform 0.2s;
                width: 200px;
            }}
            .person-card:hover {{ transform: translateY(-5px); }}
            .person-card img {{ border-radius: 8px; width: 150px; height: 150px; object-fit: cover; border: 3px solid #eee; }}
            .person-card h3 {{ margin: 10px 0 5px 0; color: #555; }}
            .person-card p {{ margin: 0; color: #777; }}
            .person-card .count {{ font-size: 1.2em; font-weight: bold; color: #333; }}
        </style>
    </head>
    <body>
        <h1>Отчет по распознаванию лиц</h1>
        <h2>Всего уникальных личностей: {total_count}</h2>
        <div class="container">
            {person_cards}
        </div>
    </body>
    </html>
    """

    # Генерируем HTML-код для каждой карточки личности
    cards_html = []
    for i, data in enumerate(sorted_metadata):
        person_id = f"Личность #{i + 1}"
        count_text = f"Обнаружен в {data['count']} файлах"
        thumbnail_b64 = data['thumbnail']
        cards_html.append(f"""
            <div class="person-card">
                <img src="data:image/jpeg;base64,{thumbnail_b64}" alt="{person_id}">
                <h3>{person_id}</h3>
                <p class="count">{count_text}</p>
            </div>
        """)

    # Собираем итоговый HTML, подставляя данные в шаблон
    final_html = html_template.format(
        total_count=len(sorted_metadata),
        person_cards="\n".join(cards_html)
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(final_html)
    print(f"\nОтчет успешно сохранен в: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Анализ видеозаписей для подсчета появлений уникальных лиц.")
    parser.add_argument("--input-dir", required=True, help="Корневой каталог с папками камер.")
    parser.add_argument("--output-html", default="face_report.html", help="Путь для сохранения HTML-отчета.")
    parser.add_argument("--model", default="hog", choices=["hog", "cnn"], help="Модель для детекции лиц: 'hog' (быстрая, CPU) или 'cnn' (точная, GPU).")
    parser.add_argument("--scale", type=float, default=0.5, help="Коэффициент масштабирования кадра перед детекцией (например, 0.5 для уменьшения в 2 раза, 1.0 для исходного размера).")
    parser.add_argument("--skip-frames", type=int, default=25, help="Количество пропускаемых кадров между анализами.")
    parser.add_argument("--debug", action="store_true", help="Включить режим отладки: сохраняет кадры, где должны быть лица.")

    args = parser.parse_args()

    input_path = Path(args.input_dir)
    if not input_path.is_dir():
        print(f"Ошибка: Директория не найдена: {args.input_dir}")
        sys.exit(1)

    print("Поиск видеофайлов .mp4...")
    video_files = sorted(list(input_path.rglob("*.mp4"))) # Сортируем для более предсказуемого порядка
    if not video_files:
        print(f"В директории {input_path} не найдено файлов .mp4")
        sys.exit(0)
    
    print(f"Найдено {len(video_files)} видеофайлов для анализа.")
    print(f"Используемые параметры: модель={args.model}, масштаб={args.scale}, пропуск кадров={args.skip_frames}")

    if args.debug:
        debug_dir = Path("debug_frames")
        debug_dir.mkdir(exist_ok=True)
        print(f"Режим отладки включен. Кадры будут сохраняться в папку: {debug_dir}")

    known_face_encodings = []
    known_face_metadata = [] 

    for video_path in tqdm(video_files, desc="Общий прогресс", unit="видео"):
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            tqdm.write(f"Не удалось открыть видео: {video_path.name}")
            continue

        # Этот set будет хранить индексы личностей, уже посчитанных для ТЕКУЩЕГО видеофайла.
        # Он сбрасывается для каждого нового видео.
        persons_found_in_this_file = set()

        frame_count = 0
        processed_one_frame_for_debug = False
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            if frame_count % args.skip_frames != 0:
                continue

            if args.scale != 1.0:
                small_frame = cv2.resize(frame, (0, 0), fx=args.scale, fy=args.scale)
            else:
                small_frame = frame
            
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_small_frame, model=args.model)
            
            if args.debug and not processed_one_frame_for_debug and face_locations:
                debug_frame = frame.copy()
                tqdm.write(f"[DEBUG] Найдено {len(face_locations)} лиц в {video_path.name}")
                for (top, right, bottom, left) in face_locations:
                    top, right, bottom, left = int(top/args.scale), int(right/args.scale), int(bottom/args.scale), int(left/args.scale)
                    cv2.rectangle(debug_frame, (left, top), (right, bottom), (0, 255, 0), 2)
                debug_filename = debug_dir / f"found_{video_path.stem}_{frame_count}.jpg"
                cv2.imwrite(str(debug_filename), debug_frame)
                processed_one_frame_for_debug = True

            if face_locations:
                current_face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations, num_jitters=1)
                
                scale_inv = 1.0 / args.scale
                for i, face_encoding in enumerate(current_face_encodings):
                    matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=FACE_COMPARISON_TOLERANCE)
                    
                    match_index = -1
                    if True in matches:
                        match_index = matches.index(True)
                    
                    # Если личность уже известна
                    if match_index != -1:
                        # --- ИЗМЕНЕНИЕ: Логика подсчета по файлам ---
                        # Увеличиваем счетчик только если мы видим эту личность В ЭТОМ ФАЙЛЕ впервые.
                        if match_index not in persons_found_in_this_file:
                            known_face_metadata[match_index]['count'] += 1
                            persons_found_in_this_file.add(match_index)
                            tqdm.write(f"Личность #{match_index + 1} повторно обнаружена в новом файле: {video_path.name}")

                    # Если это новая личность
                    else:
                        tqdm.write(f"Найдена новая личность в файле {video_path.name}")
                        known_face_encodings.append(face_encoding)
                        
                        (top, right, bottom, left) = face_locations[i]
                        original_location = (int(top*scale_inv), int(right*scale_inv), int(bottom*scale_inv), int(left*scale_inv))

                        thumbnail = create_thumbnail(frame, original_location)
                        if thumbnail:
                            known_face_metadata.append({
                                "count": 1,
                                "thumbnail": thumbnail
                            })
                            # --- ИЗМЕНЕНИЕ: Логика подсчета по файлам ---
                            # Добавляем новую личность в set для текущего файла, чтобы не посчитать ее дважды.
                            new_person_index = len(known_face_metadata) - 1
                            persons_found_in_this_file.add(new_person_index)
        cap.release()

    generate_html_report(known_face_metadata, args.output_html)

if __name__ == "__main__":
    main()