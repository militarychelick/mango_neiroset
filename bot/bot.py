import telebot
from telebot import types
from model.filters import predict_disease
import os
import sys

def retrain_model():
    print("[SELF-LEARN] Начало дообучения модели...")

    try:
        from model.model import fine_tune_model
        fine_tune_model(SELF_LEARN_DIR)
        print("[SELF-LEARN] Дообучение завершено.")
    except Exception as e:
        print("[SELF-LEARN] Ошибка при дообучении:", e)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.append(PROJECT_ROOT)
OWNER_ID = 6957191587

#Папка для обучения нейросети на новых фото
# === SELF-LEARNING переменные ===
SELF_LEARN_DIR = "self_learn"
os.makedirs(SELF_LEARN_DIR, exist_ok=True)

SELF_LEARN_COUNTER = 0


# ===== Настройки бота =====
TOKEN = "8285788264:AAHjTLJ5aWeelqyRUC2oA1K1PU62wDXtPb0"  # <- вставь сюда токен
bot = telebot.TeleBot(TOKEN)

# Папка для временного сохранения фото
# ===== Болезни =====
DISEASES_EN = [
    "Anthracnose",
    "Bacterial Canker",
    "Cutting Weevil",
    "Die Back",
    "Gall Midge",
    "Healthy",
    "Powdery Mildew",
    "Sooty Mould"
]

DISEASES_RU = [
    "Антракноз",
    "Бактериальный рак",
    "Долгоносик",
    "Отмирание ветвей",
    "Галлица",
    "Здоровый",
    "Мучнистая роса",
    "Сажа"
]

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

    bot.send_message(OWNER_ID,
                     "ℹ <b>Bot status</b>\n"
                     f"Processed photos: {len(user_last_photo)}\n"
                     f"Loaded model: mango_disease_model_pytorch.pth\n",
                     parse_mode="HTML")


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
            process_photo(chat_id, user_last_photo[chat_id])
        else:
            bot.send_message(chat_id, get_text("No photo found. Send a new one.",
                                               "Фото не найдено. Отправь новое.", chat_id))

    # --- Назад в главное меню ---
    elif call.data == "back":
        start(call.message)

@bot.message_handler(content_types=['photo'])
def handle_photo(message):
    chat_id = message.chat.id

    # Получаем файл в бинарном виде
    file_info = bot.get_file(message.photo[-1].file_id)
    downloaded_file = bot.download_file(file_info.file_path)

    # Передаем напрямую в процессинг без сохранения
    from io import BytesIO
    photo_bytes = BytesIO(downloaded_file)

    # Сохраняем в RAM
    user_last_photo[chat_id] = photo_bytes

    process_photo(chat_id, photo_bytes)


# ===== Обработка и вывод результата =====
def process_photo(chat_id, photo_path):
    bot.send_message(chat_id, get_text("Analyzing...", "Анализирую...", chat_id))

    global SELF_LEARN_COUNTER

    try:
        lang = user_lang.get(chat_id, "RU")

        # === ПРЕДСКАЗАНИЕ ===
        class_idx, confidence = predict_disease(photo_path)

        # Порог уверенности (всё ниже — "не лист манго")
        if confidence < 0.75:
            bot.send_message(chat_id, get_text(
                "Please send a photo of a mango leaf 🍃",
                "Пожалуйста, отправьте фото листа манго 🍃",
                chat_id
            ))
            return

        disease_en = DISEASES_EN[class_idx]
        disease = DISEASES_RU[class_idx] if lang == "RU" else disease_en

        # === КНОПКИ ===
        markup = types.InlineKeyboardMarkup()
        btn_again = types.InlineKeyboardButton(
            get_text("Analyze again", "Переанализировать", chat_id),
            callback_data="again"
        )
        btn_back = types.InlineKeyboardButton(
            get_text("Back", "Назад", chat_id),
            callback_data="back"
        )
        markup.add(btn_again, btn_back)

        # ========== ОТПРАВЛЯЕМ ФОТО + РЕЗУЛЬТАТ ==========
        with open(photo_path, 'rb') as img:
            bot.send_photo(
                chat_id,
                img,
                caption=(
                    f"{get_text('Result', 'Результат', chat_id)}: {disease}\n"
                    f"{get_text('Confidence', 'Вероятность', chat_id)}: {confidence*100:.1f}%"
                ),
                reply_markup=markup
            )

        # ========== АВТОСАМООБУЧЕНИЕ (только >95%) ==========
        if confidence > 0.95:
            class_dir = os.path.join("self_learn", disease_en)
            os.makedirs(class_dir, exist_ok=True)

            filename = os.path.basename(photo_path)
            save_path = os.path.join(class_dir, filename)

            # безопасно копируем, чтобы не ломать исходные файлы
            import shutil
            shutil.copy(photo_path, save_path)

            SELF_LEARN_COUNTER += 1
            print(f"[SELF-LEARN] Сохранено фото: {save_path}. Всего: {SELF_LEARN_COUNTER}")

            # Авто-дообучение каждые 20 фото
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
    bot.infinity_polling()