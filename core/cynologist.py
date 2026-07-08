import os
import time
import json
import logging
import chromadb
import google.generativeai as genai
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from typing import List, Dict
import PIL.Image

load_dotenv()

logger = logging.getLogger("app")

class CynologistAI:
    def __init__(self):
        # Настройка Gemini (БЕЗ transport='rest', так как он ломает SDK)
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        self.primary_model = genai.GenerativeModel(
            os.getenv("PRIMARY_MODEL", "gemini-2.5-pro")
        )
        self.fast_model = genai.GenerativeModel(
            os.getenv("FAST_MODEL", "gemini-2.5-flash")
        )

        # Настройка ChromaDB
        self.chroma_client = chromadb.PersistentClient(path="./data/chroma_db")

        # ВОЗВРАЩАЕМ ИСХОДНУЮ ФУНКЦИЮ ЭМБЕДДИНГОВ (которая работала вчера)
        self.gemini_ef = embedding_functions.GoogleGenerativeAiEmbeddingFunction(
            api_key=os.getenv("GOOGLE_API_KEY"), model_name="models/gemini-embedding-2"
        )

        self.collection = self.chroma_client.get_collection(
            name="lialya_knowledge", embedding_function=self.gemini_ef
        )

    def _retrieve_context(self, query: str, n_results: int = 5) -> str:
        """Извлекает релевантные куски знаний из векторной базы."""
        logger.info(f"Запрос к ChromaDB для контекста (длина запроса: {len(query)})")
        try:
            results = self.collection.query(query_texts=[query], n_results=n_results)
            docs_count = len(results.get("documents", [[]])[0])
            logger.info(f"Успешный поиск в ChromaDB. Найдено документов: {docs_count}")
            context = "\n\n".join(results["documents"][0])
            return context
        except Exception as e:
            logger.error(f"Ошибка при обращении к ChromaDB: {e}")
            raise e

    async def chat_with_ai(self, query: str) -> str:
        """Отвечает на произвольные вопросы пользователя, используя базу знаний."""
        # 1. Поиск контекста
        context = self._retrieve_context(query, n_results=3)

        # 2. Промпт для чата
        system_prompt = f"""
        Ты — Платон, эксперт-кинолог. Отвечай на вопросы владельца Ляли, опираясь ТОЛЬКО на предоставленные знания.
        Если в знаниях нет ответа, отвечай на основе своего опыта реабилитации гончих, но делай пометку, что это общая рекомендация.
        Будь краток, профессионален и поддерживай владельца.

        БАЗА ЗНАНИЙ:
        {context}

        ВОПРОС ПОЛЬЗОВАТЕЛЯ: {query}
        """

        logger.info("Отправка запроса в Gemini для свободного чата")
        t0 = time.perf_counter()
        try:
            response = await self.primary_model.generate_content_async(system_prompt)
            duration = time.perf_counter() - t0
            logger.info(f"Успешный ответ от Gemini (свободный чат) получен за {duration:.2f} сек.")
            return response.text
        except Exception as e:
            duration = time.perf_counter() - t0
            logger.error(f"Ошибка генерации Gemini (свободный чат) через {duration:.2f} сек: {e}")
            raise e

    async def analyze_report(
        self,
        user_text: str,
        checklist_data: Dict,
        history: List[Dict] = None,
        video_analysis: str = None,
        image_analysis: str = None,
    ) -> Dict:
        """Анализирует отчет пользователя, используя RAG, Gemini и результаты мультимедиа-анализа."""
        # 1. Поиск релевантного контекста в базе знаний
        search_query = f"{user_text} {str(checklist_data)}"
        knowledge_context = self._retrieve_context(search_query)

        # Подготовка истории для промпта
        history_context = ""
        if history:
            history_context = (
                "\nИСТОРИЯ ПРЕДЫДУЩИХ НАБЛЮДЕНИЙ (для понимания динамики):\n"
            )
            for h in history:
                try:
                    h_meta = json.loads(h["checklist_data"])
                    history_context += f"- {h['report_date']}: Страх {h_meta.get('fear_level')}/10. Анализ: {h.get('insight_text', '')[:150]}...\n"
                except:
                    continue

        # 2. Формирование системного промпта
        system_prompt = f"""
        Ты — элитный кинолог-реабилитолог, специализирующийся на пугливых собаках и породе эстонская гончая.
        Твоя задача: проанализировать дневной отчет о собаке по имени Ляля, сопоставить его с историей и составить план на завтра.

        ИСПОЛЬЗУЙ СЛЕДУЮЩИЕ ЭКСПЕРТНЫЕ ЗНАНИЯ (RAG):
        {knowledge_context}
        {history_context}

        ТЕКУЩИЕ ДАННЫЕ ОТЧЕТА:
        Текст владельца: {user_text}
        Расширенные метрики дня (включая детализированное пищевое поведение, фокус и мотивацию): {json.dumps(checklist_data, indent=2, ensure_ascii=False)}
        
        ДАННЫЕ ВИДЕО-АНАЛИЗА (если есть):
        {video_analysis if video_analysis else "Видео не загружалось"}
        
        ДАННЫЕ ФОТО-АНАЛИЗА (если есть):
        {image_analysis if image_analysis else "Фото не загружалось"}

        ФОРМАТ ОТВЕТА (строгий JSON):
        {{
            "insight": "Твой глубокий анализ состояния собаки сегодня (нейробиология, уровень стресса)",
            "detected_triggers": ["список триггеров, которые проявились"],
            "next_day_plan": "Четкий пошаговый план упражнений на завтра на основе прогресса или регресса",
            "safety_warnings": "Предупреждения по безопасности, если есть риск побега или срыва"
        }}
        """

        # 3. Запрос к Gemini
        logger.info("Отправка запроса в Gemini для анализа отчета")
        t0 = time.perf_counter()
        try:
            response = await self.primary_model.generate_content_async(
                system_prompt,
                generation_config=genai.types.GenerationConfig(temperature=0.7),
                # Отключаем фильтры безопасности, чтобы не ловить Finish Reason 2
                safety_settings={
                    genai.types.HarmCategory.HARM_CATEGORY_HARASSMENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
                    genai.types.HarmCategory.HARM_CATEGORY_HATE_SPEECH: genai.types.HarmBlockThreshold.BLOCK_NONE,
                    genai.types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: genai.types.HarmBlockThreshold.BLOCK_NONE,
                    genai.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
                },
            )
            duration = time.perf_counter() - t0
            logger.info(f"Успешный ответ от Gemini (анализ отчета) получен за {duration:.2f} сек.")
        except Exception as e:
            duration = time.perf_counter() - t0
            logger.error(f"Ошибка генерации Gemini (анализ отчета) через {duration:.2f} сек: {e}")
            raise e

        try:
            # Сначала пытаемся получить текст ответа
            try:
                raw_text = response.text
                print(
                    "\n=== RAW GEMINI RESPONSE ===\n",
                    raw_text,
                    "\n===========================\n",
                )
            except Exception as e:
                finish_reason = (
                    getattr(response.candidates[0], "finish_reason", "unknown")
                    if response.candidates
                    else "unknown"
                )
                err_msg = f"Ответ заблокирован ИИ. Причина (finish_reason): {finish_reason}. Ошибка: {str(e)}"
                logger.error(err_msg)
                return {
                    "error": err_msg
                }

            # Очищаем текст от markdown-блоков, если они есть
            clean_text = raw_text.replace("```json", "").replace("```", "").strip()
            return json.loads(clean_text)
        except Exception as e:
            err_msg = f"Ошибка парсинга JSON: {str(e)}. Сырой ответ в консоли."
            logger.error(err_msg)
            return {"error": err_msg}

    async def analyze_video(self, video_path: str) -> str:
        """Загружает видео в Gemini, ждет обработки и анализирует поведение собаки."""
        logger.info(f"Начало анализа видео: {video_path}")
        t0 = time.perf_counter()
        
        # 1. Загрузка файла в Gemini
        try:
            video_file = genai.upload_file(path=video_path)
            logger.info(f"Видео {video_path} успешно загружено в Gemini API.")
        except Exception as e:
            logger.error(f"Ошибка загрузки видео {video_path} в Gemini: {e}")
            raise e

        # 2. Ожидание обработки (видео в Gemini обрабатывается асинхронно)
        try:
            while video_file.state.name == "PROCESSING":
                time.sleep(2)
                video_file = genai.get_file(video_file.name)

            if video_file.state.name == "FAILED":
                raise Exception("Video processing failed in Gemini")
            logger.info(f"Обработка видео в Gemini завершена. Статус: {video_file.state.name}")
        except Exception as e:
            logger.error(f"Ошибка при обработке видео {video_path} в Gemini: {e}")
            raise e

        # 3. Промпт для анализа видео
        prompt = """
        Ты — эксперт по языку тела собак. Проанализируй это видео с собакой (эстонская гончая).
        Твоя задача — найти сигналы стресса или сигналы примирения:
        - Облизывание носа (Lip licking)
        - Зевание (Yawning)
        - Отворачивание головы или взгляда (Head turn)
        - Замирание (Freezing)
        - Частые встряхивания (Shake off)
        - "Китовый глаз" (Whale eye)

        Опиши, что именно ты видишь, на каких секундах и что это значит для процесса реабилитации.
        Дай рекомендации, нужно ли было прекратить занятие или собака справлялась.
        Будь максимально детален. Отвечай на русском языке.
        """

        model_name = os.getenv("MEDIA_MODEL", "gemini-1.5-pro")
        media_model = genai.GenerativeModel(model_name)

        logger.info(f"Отправка запроса в Gemini ({model_name}) для анализа видео")
        try:
            response = await media_model.generate_content_async([video_file, prompt])
            duration = time.perf_counter() - t0
            logger.info(f"Успешный анализ видео получен от Gemini за {duration:.2f} сек.")
        except Exception as e:
            duration = time.perf_counter() - t0
            logger.error(f"Ошибка анализа видео через Gemini после {duration:.2f} сек: {e}")
            raise e

        # Удаляем файл из облака после анализа
        try:
            genai.delete_file(video_file.name)
            logger.info(f"Временный файл видео {video_file.name} удален из облака Gemini.")
        except Exception as e:
            logger.warning(f"Не удалось удалить временный файл видео {video_file.name} из облака: {e}")

        return response.text

    async def analyze_image(self, image_path: str) -> str:
        """Анализирует фотографию собаки на предмет эмоций и поз."""
        logger.info(f"Начало анализа изображения: {image_path}")
        t0 = time.perf_counter()
        
        # 1. Загрузка изображения (через контекстный менеджер для разблокировки файла)
        try:
            with PIL.Image.open(image_path) as img:
                # 2. Промпт для анализа фото
                prompt = """
                Ты — эксперт по языка тела собак. Проанализируй это фото собаки (эстонская гончая).
                Опиши:
                - Положение ушей (прижаты, направлены вперед, расслаблены)
                - Положение хвоста (если видно)
                - Мимику (напряжение пасти, взгляд)
                - Общую позу (зажатость, расслабленность, готовность к бегству)
                
                Сделай вывод об эмоциональном состоянии собаки на этом снимке.
                Отвечай на русском языке.
                """

                model_name = os.getenv("MEDIA_MODEL", "gemini-1.5-pro")
                media_model = genai.GenerativeModel(model_name)

                logger.info(f"Отправка запроса в Gemini ({model_name}) для анализа изображения")
                response = await media_model.generate_content_async([img, prompt])
                duration = time.perf_counter() - t0
                logger.info(f"Успешный анализ изображения получен от Gemini за {duration:.2f} сек.")
                return response.text
        except Exception as e:
            duration = time.perf_counter() - t0
            logger.error(f"Ошибка при анализе изображения {image_path}: {e} (время выполнения: {duration:.2f} сек.)")
            raise e
