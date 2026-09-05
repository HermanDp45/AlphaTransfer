# Продолжение проверки Chronos

Основной результат: [REPORT.md](REPORT.md). Машинный статус научных проверок: [final_verification.json](final_verification.json). Все локальные артефакты перечислены с SHA-256 в [MANIFEST.json](MANIFEST.json).

| Каталог | Содержание |
|---|---|
| `output/`, `summary.csv`, `paired_intervals.csv` | 40 годовых голов: сопоставимые окна 2 и 10 лет, сохранён прогрев исходных признаков V4 |
| `extended_contract/` | 20 десятилетних голов на точной расширенной панели V3; базовый HGB воспроизводит V3 long |
| `march/` | Январская и мартовская заморозки для исходного сопоставимого набора признаков |
| `extended_contract/march/` | Основное сравнение заморозок на точной панели V3, одинаковые даты 3 марта — 25 августа 2026 года |
| `budget900/` | Четыре полных дообучения Small по 900 шагов, прогнозы, головы и сравнение с исходными 300 шагами |
| `budget900/cpu300/` | Четыре головы на неизменённых 300-шаговых весах с полностью CPU-прогнозами; нейросеть повторно не обучалась |
| `budget900/cpu_matched_*.csv*` | Основное сравнение 900 и 300 шагов с совпадающим устройством инференса |
| `forecasts/` | Дополненные ранние прогнозы; все пересекающиеся старые значения сохранены побитово |
| `training_smoke*/` | Только измерение скорости 20 шагов; эти веса не используются в итоговых метриках |

Воспроизведение из каталога `AlphaTransfer` на Python 3.11, установленном в `/private/tmp/alphatransfer-tabm-venv/bin/python`. Используются локальные torch 2.14.0 и chronos-forecasting 2.3.1. Подробные зависимости и источники весов сохранены в исходном [foundation/README.md](../../foundation/README.md). Сеть при выполнении отключена через `HF_HUB_OFFLINE=1`; нужны неизменённые исходные веса и данные V3/V4.

```bash
python research_v4/continuation/foundation/run_heads.py --smoke
python research_v4/continuation/foundation/run_heads.py --device mps
python research_v4/continuation/foundation/assess_heads.py
python research_v4/continuation/foundation/run_extended.py
python research_v4/continuation/foundation/run_march.py
python research_v4/continuation/foundation/run_extended_march.py
python research_v4/continuation/foundation/run_budget900.py
python research_v4/continuation/foundation/run_cpu300.py
python research_v4/continuation/foundation/verify_heads.py
python research_v4/continuation/foundation/verify_additions.py
python research_v4/continuation/foundation/write_report.py
python research_v4/continuation/foundation/finalize.py
```

MPS доступен в полноценном локальном процессе, но может быть недоступен внутри sandbox. Для основного прогноза можно указать `--device cpu`; наличие готового кэша сохраняет его исходные значения и устройство происхождения. Два варианта обучения головы используют одинаковые прогнозы. Прежде чем менять код, конфигурацию или входы, создайте новое имя эксперимента: сохранённые квитанции проверяют их идентичность.

Внешние годы 2023–2025 и 2026 уже просматривались ранее. Ранние обучающие признаки головы не являются независимыми прогнозами нейросети, поскольку фиксированная нейросеть дообучалась на этой ранней истории. Эти ограничения сохраняются независимо от прохождения технических проверок.
