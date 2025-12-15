import telebot
from telebot import types
import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.append(PROJECT_ROOT)
from model.model import predict_disease, DISEASES_EN, DISEASES_RU

from model.model import retrain_model

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.append(PROJECT_ROOT)
OWNER_ID = 6957191587

SELF_LEARN_DIR = os.path.join(PROJECT_ROOT, "self_learn")
os.makedirs(SELF_LEARN_DIR, exist_ok=True)
SELF_LEARN_COUNTER = 0

TEMP_DIR = os.path.join(CURRENT_DIR, "tmp")
os.makedirs(TEMP_DIR, exist_ok=True)

user_last_result = {}     # chat_id -> последний предсказанный класс
user_result_repeats = {}  # chat_id -> количество повторов подряд

# ===== Настройки бота =====
TOKEN = "8285788264:AAHjTLJ5aWeelqyRUC2oA1K1PU62wDXtPb0"  # <- вставь сюда токен
bot = telebot.TeleBot(TOKEN)

# ===== Пользовательские состояния =====
user_lang = {}          # chat_id -> "EN" или "RU"
user_last_photo = {}    # chat_id -> путь к последнему фото

# ===== Функции =====
def get_text(text_en, text_ru, chat_id):
    return text_ru if user_lang.get(chat_id, "RU") == "RU" else text_en

def get_disease(photo_path, lang="RU"):
    disease_code, confidence = predict_disease(photo_path)
    if lang == "RU":
        disease = DISEASES_RU[disease_code]
    else:
        disease = DISEASES_EN[disease_code]
    return disease, confidence


# ===== /start =====
@bot.message_handler(commands=['start'])
def start(message):
    chat_id = message.chat.id

    markup = types.InlineKeyboardMarkup()
    btn_photo = types.InlineKeyboardButton(get_text("Send leaf photo", "Отправить фото листа", chat_id), callback_data="send_photo")
    btn_lang = types.InlineKeyboardButton(get_text("Язык/Language", "Язык/Language", chat_id), callback_data="language")
    btn_help = types.InlineKeyboardButton(get_text("Help", "Помощь", chat_id), callback_data="help")
    markup.add(btn_photo)
    markup.add(btn_lang)
    markup.add(btn_help)

    bot.send_message(chat_id, get_text(
        "Hello! 👋 I can help identify mango leaf diseases.\nChoose an action:",
        "Привет! 👋 Я помогу определить болезни листьев манго.\nВыбери действие:",
        chat_id
    ), reply_markup=markup)

@bot.message_handler(commands=['info'])
def info(message):
    if message.chat.id != OWNER_ID:
        return

    total_self = sum(
        len(files) for _, _, files in os.walk(SELF_LEARN_DIR)
    )

    bot.send_message(
        OWNER_ID,
        f"ℹ <b>Bot status</b>\n"
        f"Processed photos: {len(user_last_photo)}\n"
        f"Self-learn samples: {total_self}\n"
        f"Loaded model: mango_disease_model_pytorch.pth\n",
        parse_mode="HTML"
    )

# ===== Обработка нажатий кнопок =====
@bot.callback_query_handler(func=lambda call: True)
def callback(call):
    chat_id = call.message.chat.id

    # Сбрасываем нажатие кнопки (чтобы можно было нажимать снова)
    bot.answer_callback_query(call.id)

    # --- Отправка фото ---
    if call.data == "send_photo":
        chat_id = call.message.chat.id
        markup = types.InlineKeyboardMarkup()
        btn_back = types.InlineKeyboardButton(get_text("Back", "Назад", chat_id), callback_data="back")
        markup.add(btn_back)

        bot.send_message(chat_id, get_text(
            "Send me a photo of a mango leaf 📷",
            "Просто отправь фото листа манго 📷",
            chat_id
        ), reply_markup=markup)

    # --- Помощь ---
    elif call.data == "help":
        markup = types.InlineKeyboardMarkup()
        btn_back = types.InlineKeyboardButton(get_text("Back", "Назад", chat_id), callback_data="back")
        markup.add(btn_back)

        bot.send_message(chat_id,
                         get_text(
                             "Send a photo of a mango leaf and I will tell you the disease.\nSupported diseases:\n" +
                             "\n".join(DISEASES_EN),
                             "Просто отправь фото листа манго, и я скажу, какая болезнь у растения.\nПоддерживаемые болезни:\n" +
                             "\n".join(DISEASES_RU),
                             chat_id
                         ),
                         reply_markup=markup
                         )

    # --- Меню выбора языка ---
    elif call.data == "language":
        current_lang = user_lang.get(chat_id, "RU")
        markup = types.InlineKeyboardMarkup()

        ru_label = "🇷🇺 Русский"
        en_label = "🇬🇧 English"
        if current_lang == "RU":
            ru_label += " ✅"
        else:
            en_label += " ✅"

        btn_ru = types.InlineKeyboardButton(ru_label, callback_data="lang_ru")
        btn_en = types.InlineKeyboardButton(en_label, callback_data="lang_en")
        btn_back = types.InlineKeyboardButton(get_text("Back", "Назад", chat_id), callback_data="back")
        markup.add(btn_ru, btn_en)
        markup.add(btn_back)

        bot.send_message(chat_id, get_text("Choose your language:", "Выбери язык:", chat_id), reply_markup=markup)

    # --- Установка русского языка ---
    elif call.data == "lang_ru":
        user_lang[chat_id] = "RU"

        markup = types.InlineKeyboardMarkup()
        ru_label = "🇷🇺 Русский ✅"
        en_label = "🇬🇧 English"
        btn_ru = types.InlineKeyboardButton(ru_label, callback_data="lang_ru")
        btn_en = types.InlineKeyboardButton(en_label, callback_data="lang_en")
        btn_back = types.InlineKeyboardButton(get_text("Back", "Назад", chat_id), callback_data="back")
        markup.add(btn_ru, btn_en)
        markup.add(btn_back)

        bot.send_message(chat_id, "Язык установлен на Русский ✅", reply_markup=markup)

    # --- Установка английского языка ---
    elif call.data == "lang_en":
        user_lang[chat_id] = "EN"

        markup = types.InlineKeyboardMarkup()
        ru_label = "🇷🇺 Русский"
        en_label = "🇬🇧 English ✅"
        btn_ru = types.InlineKeyboardButton(ru_label, callback_data="lang_ru")
        btn_en = types.InlineKeyboardButton(en_label, callback_data="lang_en")
        btn_back = types.InlineKeyboardButton(get_text("Back", "Назад", chat_id), callback_data="back")
        markup.add(btn_ru, btn_en)
        markup.add(btn_back)

        bot.send_message(chat_id, "Language set to English ✅", reply_markup=markup)

    # --- Переанализировать ---
    elif call.data == "again":
        if chat_id in user_last_photo:
            process_photo(chat_id, user_last_photo[chat_id], force_full=True)
        else:
            bot.send_message(chat_id, get_text(
                "No photo found. Send a new one.",
                "Фото не найдено. Отправь новое.", chat_id
            ))

    # --- Назад в главное меню ---
    elif call.data == "back":
        start(call.message)

import uuid

@bot.message_handler(content_types=['photo'])
def handle_photo(message):
    chat_id = message.chat.id

    # загружаем фото
    file_info = bot.get_file(message.photo[-1].file_id)
    downloaded_file = bot.download_file(file_info.file_path)

    # сохраняем во временный файл
    filename = f"{uuid.uuid4().hex}.jpg"
    temp_path = os.path.join(TEMP_DIR, filename)

    with open(temp_path, "wb") as f:
        f.write(downloaded_file)

    user_last_photo[chat_id] = temp_path

    process_photo(chat_id, temp_path)

def process_photo(chat_id, photo_path, force_full=False):
    bot.send_message(chat_id, get_text("Analyzing...", "Анализирую...", chat_id))
    global SELF_LEARN_COUNTER

    try:
        lang = user_lang.get(chat_id, "RU")

        # === ПРЕДСКАЗАНИЕ ===
        if force_full:
            bot.send_message(chat_id, get_text(
                "Reanalyzing full image...", "Переанализирую всё изображение...", chat_id
            ))
            from model.model import predict_full_image
            class_idx, confidence = predict_full_image(photo_path)
        else:
            class_idx, confidence = predict_disease(photo_path)
        disease_en = DISEASES_EN[class_idx]

        ru_map = {
            "Anthracnose": "Антракноз",
            "Bacterial Canker": "Бактериальный рак",
            "Cutting Weevil": "Долгоносик",
            "Die Back": "Отмирание ветвей",
            "Gall Midge": "Галлица",
            "Healthy": "Здоровый",
            "Powdery Mildew": "Мучнистая роса",
            "Sooty Mould": "Сажа"
        }
        disease = ru_map[disease_en] if lang == "RU" else disease_en
        confidence_display = min(confidence + 0.25, 1.0)

        # === Логика «уверенности через повтор» ===
        prev_class = user_last_result.get(chat_id)
        repeats = user_result_repeats.get(chat_id, 0)

        if prev_class == disease and prev_class is not None:
            repeats += 1
        else:
            repeats = 1  # сбрасываем счётчик, если класс изменился
        user_last_result[chat_id] = disease
        user_result_repeats[chat_id] = repeats

        # === Формируем ответ в зависимости от уверенности и повторов ===
        if confidence < 0.3 and repeats < 2:
            text_msg = get_text(
                "Not sure 🤔 Try photographing the full sheet or reanalyzing it.",
                "Совсем не уверен 🤔 Попробуй сфотографировать полный лист или переанализировать.",
                chat_id
            )
        elif confidence < 0.5 and repeats < 2:
            text_msg = get_text(
                f"Looks like {disease}, but it's better to reanalyze 😅",
                f"Похоже на {disease}, но лучше переанализируй 😅",
                chat_id
            )
        else:
            # === Показываем финальный результат ===
            text_msg = (
                f"{get_text('Result', 'Результат', chat_id)}: {disease}\n"
                f"{get_text('Confidence', 'Вероятность', chat_id)}: {confidence_display * 100:.1f}%"
            )

        # кнопки
        markup = types.InlineKeyboardMarkup()
        markup.add(
            types.InlineKeyboardButton(get_text("Analyze again", "Переанализировать", chat_id), callback_data="again"),
            types.InlineKeyboardButton(get_text("Back", "Назад", chat_id), callback_data="back")
        )

        # отправляем фото и ответ
        with open(photo_path, 'rb') as img:
            bot.send_photo(chat_id, img, caption=text_msg, reply_markup=markup)

        # === ДО-ОБУЧЕНИЕ ===
        if confidence > 0.95:
            class_dir = os.path.join(SELF_LEARN_DIR, disease_en)
            os.makedirs(class_dir, exist_ok=True)

            import shutil
            save_path = os.path.join(class_dir, os.path.basename(photo_path))
            shutil.copy(photo_path, save_path)

            SELF_LEARN_COUNTER += 1
            print(f"[SELF-LEARN] Saved: {save_path}. Total: {SELF_LEARN_COUNTER}")

            if SELF_LEARN_COUNTER >= 20:
                SELF_LEARN_COUNTER = 0
                import threading
                threading.Thread(target=retrain_model, daemon=True).start()

    except Exception as e:
        print("Ошибка в process_photo:", e)
        bot.send_message(chat_id, get_text(
            "Error while processing the image 😥",
            "Ошибка при обработке изображения 😥",
            chat_id
        ))


# ===== Запуск =====
if __name__ == "__main__":
    print("Бот запущен...")
    while True:
        try:
            bot.infinity_polling(timeout=60, long_polling_timeout=10)
        except Exception as ex:
            print("⚠ Ошибка polling:", ex)
            import time

            time.sleep(5)