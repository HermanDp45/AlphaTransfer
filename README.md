# AlphaTransfer

Хакатонное решение для сигналов выгодного момента трансграничного перевода.

## С чего начать

- [План ML, алгоритма и MVP](Researches/ml_hackathon/HACKATHON_ML_PLAN.md) — приоритеты, задачи, критерии приёмки и текущие baseline-выводы.
- [Результаты baseline-прогона](Researches/ml_hackathon/README.md) — как воспроизвести аудит, бэктест и таблицу сигналов.
- [Исследование чата Казахстана](Researches/chat_research_kz/report.md) — Jobs to be Done, сегменты и продуктовые ограничения.
- [Постановка кейса](global_task.md) и [ответы кейсодателя](Вопросы%20и%20ответы.md).

Быстрый запуск из этой директории:

```bash
python3 scripts/hackathon_baseline.py
python3 -m unittest -v tests/test_hackathon_baseline.py
```
