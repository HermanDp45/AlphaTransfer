# Дополнительные исследования

Эти скрипты полезны для аудита, но не вызываются центральным `main.py`:

- `cadence_quality_frontier.py` — post-hoc frontier частоты и качества сигналов;
- `cbr_cross_rate_identity.py` — арифметическая проверка механизма CBR cross-rate;
- `legacy_evaluator_audit.py` — воспроизведение ошибок исходного KZT evaluator.

Готовые результаты cadence/mechanism лежат в `artifacts/`. Legacy-аудит требует
исходные `data/kzt_v0/features.csv` и `backtest.json` из исследуемой ветки; это
исторический инструмент диагностики, а не часть воспроизводимой final-модели.
