import requests
import logging
import sqlite3
import numpy as np
from datetime import datetime
import warnings
import json
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)



def get_embedding_from_ollama(text):
    try:
        response = requests.post("http://localhost:11434/api/embeddings", json={
            "model": "nomic-embed-text:v1.5",
            "prompt": text
        })
        response.raise_for_status()
        return np.array(response.json()["embedding"])
    except Exception as e:
        logger.error(f"خطا در گرفتن embedding: {str(e)}")
        return np.zeros(384)

def init_db():
    conn = sqlite3.connect("christopher.db")
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS interactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            task_type TEXT,
            user_input TEXT,
            language TEXT,
            prompt TEXT,
            response TEXT
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS memory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT,
            embedding BLOB,
            timestamp TEXT
        )
    ''')
    conn.commit()
    conn.close()

def embed_to_blob(embed):
    return embed.astype(np.float32).tobytes()



def save_to_longterm_memory(text):
    try:
        embedding = get_embedding_from_ollama(text)
        with sqlite3.connect("christopher.db") as conn:
            cursor = conn.cursor()
            cursor.execute("INSERT INTO memory (text, embedding, timestamp) VALUES (?, ?, ?)",
                         (text, embed_to_blob(embedding), datetime.now().isoformat()))
            conn.commit()
            logger.info("Saved to long-term memory.")
    except Exception as e:
        logger.error(f"Error saving to long-term memory: {str(e)}")



def gemma3(user_input, explain_language="فارسی"):
    prompt = (
        f"تو یک مدل زبانی به نام کریستوفر هستی.\n"
        f"کاربر این کد را داده:\n{user_input}\n"
        f"وظیفه‌ات اینه که این کد را خط به خط به زبان {explain_language} توضیح بدی."
    )
    try:
        response = requests.post("http://localhost:11434/api/generate", json={
            'model': "gemma3",
            'prompt': prompt,
            'num_predict': 2048,
            'stream': False
        })
        response.raise_for_status()
        return response.json().get("response", "")
    except Exception as e:
        logger.error(f"خطا در تولید پاسخ توسط gemma3: {e}")
        return "❌ خطا در توضیح کد با gemma3"






def detect(user_input, language):
    prompt = (
        f"شما یک مدل زبانی هستید که وظیفه‌تان تبدیل توضیحات کاربر به پرامپتی مناسب برای تولید کد با مدل Qwen-Coder است.\n"
        f"کاربر این را گفته:\n{user_input}\n"
        f"لطفاً به انگلیسی و با دقت، یک پرامپت دقیق و حرفه ای برای مدل Qwen بنویس. زبان مورد نظر: {language}. فقط پرامپت را خروجی بده."
    )
    try:
        response = requests.post("http://localhost:11434/api/generate", json={
            'model': "gemma3",
            'prompt': prompt,
            'stream': False
        })
        response.raise_for_status()
        data = response.json()
        return data.get("response", "")  
    except Exception as e:
        logger.error(f"خطا در ساخت پرامپت توسط gemma3: {e}")
        return "خطا در ساخت پرامپت"
    



def call_model(prompt):
    try:
        response = requests.post("http://localhost:11434/api/generate", json={
            'model': "qwen2.5-coder:7b",
            'prompt': prompt,
            'system': 'تو یک دستیار کدنویسی حرفه‌ای هستی. اسم تو کریستوفره. همیشه خروجی دقیق بده.',
            'num_predict': 16384,
            'temperature': 0.7,
            'stream': True
        }, stream=True)

        response.raise_for_status()

        output = ""
        for line in response.iter_lines():
            if line:
                try:
                    part = json.loads(line.decode("utf-8"))
                    chunk = part.get("response", "")
                    output += chunk
                except json.JSONDecodeError as e:
                    logger.warning(f"خطا در پردازش chunk: {e}")
        return output if output.strip() else "⚠️ مدل پاسخی تولید نکرد."
    except requests.RequestException as e:
        logger.error(f"خطا در ارتباط با مدل: {str(e)}")
        return "❌ خطا در تماس با مدل"

def log_to_db(task_type, user_input, language, prompt, response):
    try:
        with sqlite3.connect("christopher.db") as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO interactions (timestamp, task_type, user_input, language, prompt, response)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (datetime.now().isoformat(), task_type, user_input, language, prompt, response))
            conn.commit()
    except Exception as e:
        logger.error(f"Error logging interaction: {str(e)}")

def run_cli():
    print("=" * 45)
    print("🤖 خوش اومدی به کریستوفر — دستیار کدنویس آفلاین")
    print("Welcome to Christopher, the offline coding assistant 🤖")
    print("=" * 45)

    while True:
        print("\n🔧 چه کاری می‌خوای انجام بدم؟")
        print("1. تولید کد از توضیح")
        print("2. تحلیل و توضیح کد")
        print("3. تکمیل کد ناقص")
        print("4. دیباگ کد")
        print("5. خروج")

        choice = input("👉 انتخاب (1 تا 5): ").strip()

        if choice == "5":
            print("✅")
            break

        elif choice in ["1", "2", "3", "4"]:
            user_input = input("\n📝 لطفاً توضیح یا کدت رو وارد کن:\n")
            language = input("🌐 با چه زبان برنامه‌نویسی کار کنیم؟ (مثلاً python, c, java): ").strip()

            if choice == "1":
                prompt = detect(user_input, language) 
                print("⏳ در حال تولید کد توسط Qwen...")
                response = call_model(prompt)

            elif choice == "2":
                explain_lang = input("🌐 توضیح به چه زبانی باشه؟ (مثلاً فارسی، انگلیسی): ").strip()
                prompt = f"توضیح کد به زبان {explain_lang}:\n{user_input}"
                print("⏳ در حال تحلیل و توضیح کد توسط gemma3...")
                response = gemma3(user_input, explain_lang)

            elif choice == "3":
                prompt = (
                    f"This is an incomplete code in {language}:\n{user_input}\n"
                    f"Please complete the code properly."
                )
                print("⏳ در حال تکمیل کد توسط Qwen...")
                response = call_model(prompt)

            elif choice == "4":
                prompt = (
                    f"This code has errors in {language}:\n{user_input}\n"
                    f"Please debug and fix all issues."
                )
                print("⏳ در حال دیباگ کد توسط Qwen...")
                response = call_model(prompt)
            else:
                print("❌ گزینه نامعتبر. لطفاً عددی بین 1 تا 5 وارد کن.")
            


            print("\n📤 پاسخ مدل:\n")
            print(response)

            log_to_db(choice, user_input, language, prompt, response)
            save_to_longterm_memory(user_input + "\n" + response)

if __name__ == "__main__":
    init_db()
    run_cli()


