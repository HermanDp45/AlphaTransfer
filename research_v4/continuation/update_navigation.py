"""Link the completed continuation; preserve prior report versions verbatim."""
from pathlib import Path
HERE=Path(__file__).resolve().parent
for name in ('README.md','REPORT.md'):
    original=(HERE/'prior_reports'/name).read_text()
    banner='> **Продолжение V4:** OXR 2018–2026, десять лет обучения NOW поверх Chronos, бюджет 300/900 шагов и стресс-тест задержки источников. [Новый отчёт](continuation/REPORT.md), [новая таблица сравнений](continuation/COMPARISON.csv). Ниже — исходный основной этап V4.\n\n'
    if name=='REPORT.md':
        banner+='> **Уточнение задержки Halyk, KZT 2023–2025:** указанные ниже 0.186250 относятся к отдельному обучению с lag2. Для исходной неизменённой модели при задержке входа на день измерено 0.181404; детали и контролируемые сравнения — в продолжении.\n\n'
    (HERE.parent/name).write_text(banner+original)
