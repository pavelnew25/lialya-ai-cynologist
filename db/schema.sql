-- Профиль собаки
CREATE TABLE IF NOT EXISTS dog_profile (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    breed TEXT NOT NULL,
    start_date DATE DEFAULT CURRENT_DATE
);

-- Ежедневные отчеты
CREATE TABLE IF NOT EXISTS daily_reports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    report_date DATE DEFAULT CURRENT_DATE,
    user_text TEXT,              -- Свободное описание дня
    checklist_data TEXT,         -- JSON с ответами (команды, уровень страха и т.д.)
    media_analysis_summary TEXT  -- Краткий итог анализа видео/фото
);

-- Результаты работы ИИ
CREATE TABLE IF NOT EXISTS ai_insights (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    report_id INTEGER,
    insight_text TEXT,           -- Что ИИ понял сегодня
    next_day_plan TEXT,          -- План на завтра (Реактивное планирование)
    detected_triggers TEXT,      -- Найденные триггеры страха
    safety_warnings TEXT,        -- Предупреждения по безопасности
    FOREIGN KEY (report_id) REFERENCES daily_reports(id)
);
