# -*- coding: utf-8 -*-
from flask import Flask, request, jsonify, render_template
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import gdown
import os
import requests
import zipfile
import shutil

app = Flask(__name__)

# --- 1. إعداد نموذج التضمين (Embedding Model) ---
print("Initializing embedding model...")
embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-en-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)
print("Embedding model initialized.")

# --- 2. إعداد قاعدة بيانات Chroma ---
# بيانات تحميل الملف
zip_url = "https://drive.google.com/uc?id=1TRCTZ_txfmdzSfEGr_YXS9h4Kx4ZWNEx"
db_directory = "chroma_db"  # اسم المجلد النهائي لقاعدة البيانات

# التحقق إذا كان مجلد قاعدة البيانات غير موجود، ثم نقوم بتحميله وإعداده
if not os.path.isdir(db_directory):
    print(f"Database directory '{db_directory}' not found. Starting download and setup...")
    temp_zip_path = "chroma_dataset.zip"
    temp_extract_path = "chroma_dataset_temp"

    # الخطوة أ: تحميل الملف
    try:
        print(f"Downloading data from Google Drive...")
        gdown.download(url=zip_url, output=temp_zip_path, quiet=False)
        print("Download complete.")
    except Exception as e:
        print(f"FATAL ERROR: Failed to download the file: {e}")
        exit()

    # الخطوة ب: فك ضغط الملف
    try:
        print(f"Extracting '{temp_zip_path}' to temporary directory '{temp_extract_path}'...")
        # التأكد من أن المجلد المؤقت فارغ ونظيف
        if os.path.exists(temp_extract_path):
            shutil.rmtree(temp_extract_path)
        os.makedirs(temp_extract_path)

        with zipfile.ZipFile(temp_zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_extract_path)
        print("Extraction complete.")
        # حذف الملف المضغوط بعد الانتهاء منه
        os.remove(temp_zip_path)
    except Exception as e:
        print(f"FATAL ERROR: Failed to extract the zip file: {e}")
        if os.path.exists(temp_zip_path):
            os.remove(temp_zip_path) # تنظيف
        exit()

    # الخطوة ج: البحث عن المجلد الصحيح وإعادة تسميته
    try:
        extracted_contents = os.listdir(temp_extract_path)
        print(f"Contents of temporary directory: {extracted_contents}")

        # إذا كان الملف المضغوط يحتوي على مجلد رئيسي واحد بداخله
        if len(extracted_contents) == 1 and os.path.isdir(os.path.join(temp_extract_path, extracted_contents[0])):
            source_path = os.path.join(temp_extract_path, extracted_contents[0])
            print(f"Detected nested directory: '{source_path}'. Moving it to '{db_directory}'.")
            shutil.move(source_path, db_directory)
            shutil.rmtree(temp_extract_path) # حذف المجلد المؤقت الفارغ
        else:
            # إذا كانت الملفات مباشرة في المجلد
            print(f"Data is directly in the root. Moving '{temp_extract_path}' to '{db_directory}'.")
            shutil.move(temp_extract_path, db_directory)

        print(f"Database successfully placed at '{db_directory}'.")
    except Exception as e:
        print(f"FATAL ERROR: Failed to organize the database directory: {e}")
        exit()
else:
    print(f"Database directory '{db_directory}' already exists. Skipping download.")

# --- 3. تحميل قاعدة البيانات إلى Chroma ---
try:
    print(f"Loading vector store from: '{db_directory}'...")
    # تحقق حاسم: هل يحتوي المجلد على الملف الأساسي لـ Chroma؟
    if not os.path.exists(os.path.join(db_directory, "chroma.sqlite3")):
         raise FileNotFoundError(f"Chroma database file (chroma.sqlite3) not found in '{db_directory}'. The directory structure is incorrect or the download failed.")

    vector_store = Chroma(
        persist_directory=db_directory,
        embedding_function=embedding_model
    )
    print("Vector store loaded successfully.")
except Exception as e:
    print(f"CRITICAL ERROR: Failed to load the Chroma database. {e}")
    exit()

# --- 4. إعداد الـ Retriever ---
retriever = vector_store.as_retriever(
    search_kwargs={"k": 5},
    search_type="mmr"
)
print("Retriever is ready.")

# --- 5. إعداد واجهة برمجة التطبيقات (API) للنموذج اللغوي ---
# انتبه: يجب حماية مفتاح الواجهة البرمجية في بيئة الإنتاج
HF_API_KEY = os.getenv("OPENROUTER_API_KEY", "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")

def call_llm_api(prompt: str):
    """Calls the OpenRouter API to get a response from the LLM."""
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {HF_API_KEY}",
    }
    payload = {
        "model": "deepseek/deepseek-chat-v3-0324:free", # استخدام موديل قياسي للتوافقية
        "messages": [{"role": "user", "content": prompt}],
    }
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()  # This will raise an HTTPError for bad responses (4xx or 5xx)
        return response.json()["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        print(f"API Request Error: {e}")
        return f"Error communicating with the API: {e}"
    except (KeyError, IndexError) as e:
        print(f"API Response Parsing Error: {e}")
        return "Error parsing the response from the API."
    except Exception as e:
        print(f"An unexpected error occurred in call_llm_api: {e}")
        return "An unexpected error occurred."


# --- 6. إعداد تطبيق Flask ---
@app.route('/')
def home():
    # تأكد من وجود ملف 'index.html' في مجلد 'templates'
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    """Handles the incoming question from the user."""
    try:
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({'error': 'Invalid request. "query" field is missing.'}), 400

        user_query = data['query']
        print(f"Received query: {user_query}")
        answer = ask_question(user_query)
        return jsonify({'answer': answer})
    except Exception as e:
        print(f"Error in /ask endpoint: {e}")
        return jsonify({'error': f'An internal server error occurred: {e}'}), 500

def ask_question(query: str) -> str:
    """Uses the RAG pipeline to answer a question."""
    # استرجاع المستندات ذات الصلة
    docs = retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in docs]) if docs else "No relevant context found."
    print(f"Retrieved context for query '{query}':\n---\n{context}\n---")

    # بناء الـ Prompt
    prompt = f"""You are an expert assistant. Your ONLY source of information is the provided "Context".
You MUST answer questions using ONLY the information explicitly given in the "Context".
If the question is a greeting (e.g., "hi", "hello", "hey", "greetings", "how are you"), reply with exactly: "Hello! How can I assist you today? 😊".
If the answer can be found directly or inferred clearly from the "Context", provide that answer concisely.
If the answer is NOT in the "Context" or cannot be directly inferred, you MUST reply with exactly: "Sorry, I don't have enough information about your question"
Do NOT add extra explanations, guesses, or unrelated information.

Context:
{context}

Question:
{query}
"""
    answer = call_llm_api(prompt)
    return answer

if __name__ == '__main__':
    # تأكد من أن debug=False في بيئة الإنتاج
    app.run(debug=True, port=5000)
