import os
import re
from typing import List, Dict
import chromadb
from chromadb.utils import embedding_functions
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

class GeminiEmbeddingFunction(embedding_functions.EmbeddingFunction):
    """Кастомный класс для получения эмбеддингов через стабильный google-generativeai SDK."""
    def __init__(self, api_key: str, model_name: str = "models/gemini-embedding-2"):
        genai.configure(api_key=api_key)
        self.model_name = model_name

    def __call__(self, input: List[str]) -> List[List[float]]:
        # Важно: google-generativeai возвращает список эмбеддингов в другом формате
        response = genai.embed_content(
            model=self.model_name,
            content=input,
            task_type="retrieval_document"
        )
        return response['embedding']

def split_markdown_by_headers(text: str) -> List[Dict[str, str]]:
    """Разбивает текст по заголовкам второго и третьего уровня (##, ###)."""
    sections = re.split(r'\n(?=#{2,3} )', text)
    chunks = []
    
    current_header = "Intro"
    for section in sections:
        lines = section.strip().split('\n')
        if lines and lines[0].startswith('##'):
            current_header = lines[0].replace('#', '').strip()
            content = '\n'.join(lines[1:])
        else:
            content = section
            
        if content.strip():
            chunks.append({
                "header": current_header,
                "content": content.strip()
            })
    return chunks

def ingest_knowledge(file_path: str, db_path: str):
    """Индексирует базу знаний в ChromaDB."""
    if not os.path.exists(file_path):
        print(f"Ошибка: Файл {file_path} не найден.")
        return

    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    chunks = split_markdown_by_headers(text)
    
    # Инициализация ChromaDB
    client = chromadb.PersistentClient(path=db_path)
    
    # Настройка функции эмбеддингов
    gemini_ef = GeminiEmbeddingFunction(api_key=os.getenv("GOOGLE_API_KEY"))
    
    collection = client.get_or_create_collection(
        name="lialya_knowledge",
        embedding_function=gemini_ef
    )

    # Подготовка данных
    documents = [c["content"] for c in chunks]
    metadatas = [{"header": c["header"]} for c in chunks]
    ids = [f"chunk_{i}" for i in range(len(chunks))]

    # Загрузка
    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )
    print(f"Успешно проиндексировано {len(chunks)} разделов.")

if __name__ == "__main__":
    KNOWLEDGE_FILE = r"d:\Research_and_Analytics\База знаний для реабилитации эстонской гончей.md"
    DB_PATH = "./data/chroma_db"
    
    ingest_knowledge(KNOWLEDGE_FILE, DB_PATH)
