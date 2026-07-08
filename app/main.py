import os
import shutil
import sys
import traceback
import logging
from pathlib import Path
from typing import Dict, List, Optional

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv
# pyrefly: ignore [missing-import]
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
# pyrefly: ignore [missing-import]
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
# pyrefly: ignore [missing-import]
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from core.cynologist import CynologistAI
from db.manager import DBManager

load_dotenv()

# Настройка логирования в соответствии с требованиями: Время | Уровень (INFO/ERROR) | Что произошло
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler("app.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("app")

app = FastAPI(title="Lialya AI-Cynologist API")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Инициализация компонентов
db_path = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./data/lialya.db").replace("sqlite+aiosqlite:///", "")
db = DBManager(db_path)
ai = CynologistAI()

class ReportRequest(BaseModel):
    user_text: str
    checklist: Dict
    video_analysis: Optional[str] = None
    image_analysis: Optional[str] = None

class ChatRequest(BaseModel):
    message: str

@app.on_event("startup")
async def startup():
    logger.info("Запуск приложения Lialya AI-Cynologist...")
    await db.init_db()
    if not os.path.exists("data/uploads"):
        os.makedirs("data/uploads")
    logger.info("Инициализация базы данных и каталогов успешно завершена.")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

@app.post("/report")
async def create_report(request: ReportRequest):
    """Принимает отчет, анализирует его через ИИ и сохраняет в БД."""
    logger.info(f"Получен входящий запрос на генерацию отчета. Длина текста отзыва: {len(request.user_text)} символов.")
    try:
        # Получаем историю последних отчетов для контекста динамики
        history = await db.get_last_reports(limit=3)
        
        # 1. Анализ через ИИ (RAG + History + Video + Photo)
        analysis = await ai.analyze_report(
            request.user_text, 
            request.checklist, 
            history=history,
            video_analysis=request.video_analysis,
            image_analysis=request.image_analysis
        )
        
        if "error" in analysis:
            logger.error(f"ИИ вернул ошибку анализа: {analysis['error']}")
            raise HTTPException(status_code=500, detail=analysis["error"])

        # 2. Сохранение отчета в БД
        report_id = await db.save_daily_report(
            user_text=request.user_text,
            checklist_data=request.checklist,
            media_summary=analysis.get("insight", "")[:200] # Краткое превью
        )

        # 3. Сохранение инсайтов ИИ
        await db.save_ai_insight(
            report_id=report_id,
            insight=analysis.get("insight", ""),
            plan=analysis.get("next_day_plan", ""),
            triggers=", ".join(analysis.get("detected_triggers", [])),
            safety=analysis.get("safety_warnings", "")
        )

        logger.info(f"Отчет успешно создан и сохранен. ID отчета: {report_id}")
        return {
            "status": "success",
            "report_id": report_id,
            "analysis": analysis
        }
    except Exception as e:
        logger.error(f"Ошибка при обработке и создании отчета: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/report/{report_id}")
async def delete_report(report_id: int):
    logger.info(f"Получен запрос на удаление отчета с ID: {report_id}")
    try:
        await db.delete_report(report_id)
        logger.info(f"Отчет с ID {report_id} успешно удален.")
        return {"status": "success"}
    except Exception as e:
        logger.error(f"Ошибка при удалении отчета с ID {report_id}: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/history")
async def get_history(limit: int = 10):
    """Возвращает историю отчетов и анализов."""
    logger.info(f"Получен запрос истории отчетов. Лимит: {limit}")
    try:
        reports = await db.get_last_reports(limit=limit)
        logger.info(f"Успешно извлечено {len(reports)} отчетов из базы данных.")
        return reports
    except Exception as e:
        logger.error(f"Ошибка при получении истории: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def chat(request: ChatRequest):
    """Свободный чат с Платоном по базе знаний."""
    logger.info(f"Получен запрос в чат. Длина сообщения: {len(request.message)} символов.")
    try:
        response = await ai.chat_with_ai(request.message)
        logger.info("Ответ чата успешно сгенерирован.")
        return {"response": response}
    except Exception as e:
        logger.error(f"Ошибка чата: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload_video")
async def upload_video(file: UploadFile = File(...)):
    """Принимает видеофайл, сохраняет его локально и отправляет на анализ в AI."""
    logger.info(f"Получен запрос на загрузку видео. Файл: {file.filename}")
    try:
        # Валидация расширения
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in ['.mp4', '.mov', '.avi', '.webm']:
            logger.error(f"Попытка загрузить неподдерживаемый формат видео: {file.filename}")
            raise HTTPException(status_code=400, detail="Неподдерживаемый формат видео")
            
        file_path = f"data/uploads/{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        logger.info(f"Видео успешно сохранено локально: {file_path}. Отправка на анализ...")
        analysis = await ai.analyze_video(file_path)
        
        # Удаляем локальный файл после анализа
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                logger.info(f"Локальный файл видео {file_path} успешно удален после анализа.")
            except Exception as remove_err:
                logger.warning(f"Не удалось удалить локальный файл видео {file_path}: {remove_err}")
            
        return {"analysis": analysis}
    except Exception as e:
        logger.error(f"Ошибка при загрузке или анализе видео: {str(e)}")
        if 'file_path' in locals() and os.path.exists(file_path):
            try:
                os.remove(file_path)
                logger.info(f"Локальный файл видео {file_path} удален в блоке обработки ошибок.")
            except:
                pass
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload_image")
async def upload_image(file: UploadFile = File(...)):
    """Принимает фотографию, сохраняет её локально и отправляет на анализ в AI."""
    logger.info(f"Получен запрос на загрузку изображения. Файл: {file.filename}")
    try:
        # Валидация расширения
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in ['.jpg', '.jpeg', '.png', '.webp']:
            logger.error(f"Попытка загрузить неподдерживаемый формат изображения: {file.filename}")
            raise HTTPException(status_code=400, detail="Неподдерживаемый формат изображения")
            
        file_path = f"data/uploads/{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        logger.info(f"Изображение успешно сохранено локально: {file_path}. Отправка на анализ...")
        analysis = await ai.analyze_image(file_path)
        
        # Удаляем локальный файл после анализа
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                logger.info(f"Локальный файл изображения {file_path} успешно удален после анализа.")
            except Exception as remove_err:
                logger.warning(f"Не удалось удалить локальный файл изображения {file_path}: {remove_err}")
            
        return {"analysis": analysis}
    except Exception as e:
        logger.error(f"Ошибка при загрузке или анализе изображения: {str(e)}")
        if 'file_path' in locals() and os.path.exists(file_path):
            try:
                os.remove(file_path)
                logger.info(f"Локальный файл изображения {file_path} удален в блоке обработки ошибок.")
            except:
                pass
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
