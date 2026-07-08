import aiosqlite
import os
from datetime import date
from typing import Optional, List, Dict
import json

class DBManager:
    def __init__(self, db_path: str):
        self.db_path = db_path

    async def _execute_script(self, script_path: str):
        """Выполняет SQL-скрипт из файла."""
        if not os.path.exists(script_path):
            return
        
        async with aiosqlite.connect(self.db_path) as db:
            with open(script_path, 'r', encoding='utf-8') as f:
                await db.executescript(f.read())
            await db.commit()

    async def init_db(self):
        """Инициализирует базу данных, создавая таблицы."""
        # Убедимся, что папка для БД существует
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        schema_path = os.path.join(os.path.dirname(__file__), 'schema.sql')
        await self._execute_script(schema_path)
        
        # Миграция: добавим safety_warnings если её нет
        async with aiosqlite.connect(self.db_path) as db:
            try:
                await db.execute("ALTER TABLE ai_insights ADD COLUMN safety_warnings TEXT")
                await db.commit()
            except:
                pass # Колонка уже есть

    async def save_daily_report(self, user_text: str, checklist_data: Dict, media_summary: Optional[str] = None) -> int:
        """Сохраняет ежедневный отчет и возвращает его ID."""
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                """
                INSERT INTO daily_reports (user_text, checklist_data, media_analysis_summary)
                VALUES (?, ?, ?)
                """,
                (user_text, json.dumps(checklist_data, ensure_ascii=False), media_summary)
            )
            report_id = cursor.lastrowid
            await db.commit()
            return report_id

    async def save_ai_insight(self, report_id: int, insight: str, plan: str, triggers: str, safety: str):
        """Сохраняет анализ ИИ для конкретного отчета."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                INSERT INTO ai_insights (report_id, insight_text, next_day_plan, detected_triggers, safety_warnings)
                VALUES (?, ?, ?, ?, ?)
                """,
                (report_id, insight, plan, triggers, safety)
            )
            await db.commit()

    async def get_last_reports(self, limit: int = 7) -> List[Dict]:
        """Возвращает список последних отчетов с анализом."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                """
                SELECT r.*, i.insight_text, i.next_day_plan, i.detected_triggers, i.safety_warnings
                FROM daily_reports r
                LEFT JOIN ai_insights i ON r.id = i.report_id
                ORDER BY r.report_date DESC
                LIMIT ?
                """,
                (limit,)
            ) as cursor:
                rows = await cursor.fetchall()
                return [dict(row) for row in rows]

    async def delete_report(self, report_id: int):
        """Удаляет отчет и связанные с ним инсайты."""
        async with aiosqlite.connect(self.db_path) as db:
            # Сначала удаляем инсайты (внешний ключ)
            await db.execute("DELETE FROM ai_insights WHERE report_id = ?", (report_id,))
            # Затем сам отчет
            await db.execute("DELETE FROM daily_reports WHERE id = ?", (report_id,))
            await db.commit()
