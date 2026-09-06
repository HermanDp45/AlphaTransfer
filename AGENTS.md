# AlphaTransfer: карта актуальных артефактов

Этот файл — рабочая навигация для агентов и разработчиков. Перед изменением моделей прочитайте корневой `README.md`, `docs/MODEL_METRICS.md` и соответствующий selection/receipt. Не используйте старый V3/V4 артефакт как финальный только потому, что у него выше одна метрика.

## Финальные профили

| Профиль | Конфигурация | Политика | Канонический результат |
| --- | --- | --- | --- |
| H3 | `tabm_kzt_fullhistory`, NOW_H3, история с 2010 года | `rank80`, окно 63, cooldown 2, максимум 2/нед. | `final_solution/tabm_h3/evaluation/` |
| H5 | `tabm_kzt_h5_fullhistory`, NOW_H5, расширяющаяся история с 2010 года | `rare65`, цель 0,6–0,7 сигнала/нед., cooldown 2, максимум 2/нед. | `research_v4/h5_fullhistory/` и `final_solution/tabm_h5/` |

H3 и H5 имеют разные targets и base rate. Не сравнивайте их hit rate как метрику одной и той же задачи и не объединяйте их строки в один «общий lift».

## Быстрый запуск

Рабочая директория для всех команд — корень `AlphaTransfer`.

```bash
python3.11 -m pip install \
  -r final_solution/tabm_h3/requirements.txt \
  -r docs/requirements.txt

# Train / rebuild
python3.11 final_solution/main.py --model h3 --action train
python3.11 final_solution/main.py --model h5 --action train

# Infer; при отсутствии валидных артефактов train запускается автоматически
python3.11 final_solution/main.py --model h3 --action infer \
  --as-of 2026-09-05 --mode historical_smoke
python3.11 final_solution/main.py --model h5 --action infer \
  --corridor-filter KZT

# Metrics; также автоматически вызывает train при отсутствии артефактов
python3.11 final_solution/main.py --model h3 --action metrics --corridor-filter KZT
python3.11 final_solution/main.py --model h5 --action metrics --corridor-filter KZT
```

`--force` принудительно переобучает/пересобирает артефакты. `--skip-data-prefetch` отключает автоматическую проверку и подкачивание данных. Прямой запуск без `--model/--action` сохраняет прежний H3 inference; `--legacy` и `--research-v3` оставлены для воспроизводимости старых потоков.

## Production / current

- `final_solution/main.py` — публичный entrypoint и совместимость со старыми режимами.
- `final_solution/core.py` — общие `ModelRunner`, H3/H5 routing, auto-train, data preflight и единый metrics CSV.
- `final_solution/tabm_h3/bundle.json` — активная H3-конфигурация, features, calibration, policy и пути источников.
- `final_solution/tabm_h3/{features.py,model.py,policy.py,predict.py}` — самостоятельный H3 runtime.
- `final_solution/tabm_h3/{training_receipt.json,source_receipt.json,model_card.json}` — происхождение финального H3 checkpoint.
- `final_solution/tabm_h3/evaluation/{summary.csv,by_year.csv,selection.json,paired_intervals.csv}` — публичные H3 recipe-метрики и выбор полной истории.
- `final_solution/tabm_h3/evaluation/{closing_by_year.csv,closing_head_metrics.csv}` — метрики слоя «окно может закрыться» поверх H3; слой не создаёт новых контактов.
- `final_solution/tabm_h5/{bundle.json,feature_contract.json,policy.json}` — выбранный H5 recipe, воспроизводимый контракт признаков и редкая политика контактов.
- `final_solution/tabm_h5/{training_receipt.json,source_receipt.json,selection_receipt.json}` — происхождение финального H5 refit и ссылка на запечатанный rolling OOT выбор.
- `final_solution/tabm_h5/output/` — runtime-выводы H5; не использовать как единственный источник отчётных метрик.

Веса `*.pt` и сериализованные preprocessing/closing-файлы не должны возвращаться в git. CLI обязан восстановить их обучением, если локальных файлов нет.

## Research / evidence

- `research_v4/h5_fullhistory/{summary.csv,by_year.csv,predictions.csv.gz}` — каноническое сравнение H5 full-history и 120m с политикой `rare65` на одинаковых rolling OOT когортах 2024–2026.
- `research_v4/h5_fullhistory/selection.json` — запечатанное правило и фактический выбор длины истории H5; это source of truth для финального H5 bundle.
- `research_v4/h5_fullhistory/{paired_intervals.csv,verification.json}` — post-selection paired monthly bootstrap, проверки зрелости меток и равенства validation/test когорт.
- `research_v4/robust_selection/` — предшествующий сравнительный эксперимент H3/H5/V3 и прежний H5 rank90; использовать как evidence, а не как финальный H5 source.
- `research_v4/robust_selection/{paired_intervals.csv,retrospective_paired_intervals.csv,selected_utility_intervals.csv}` — сохранённые uncertainty-расчёты.
- `research_v4/robust_selection/REPORTING_ALL_METRICS.csv` — полный объединённый отчёт H3/H5/V3.
- `research_v4/h3_finalization/` — эксперимент полной H3-истории, ежегодные checkpoints, аудит и финальный refit.
- `research_v4/oxr2010_bank/` — эксперименты с OXR 2010+, Halyk и Treasury.
- `analysis_notes/v3_lift_reconciliation/` — объяснение расхождения старых формулировок lift V3.
- `review_artifacts/METRIC_CONTRACT_V2.md` — определения основных метрик и ограничения интерпретации.

Запуск research-скриптов требует чтения их локального `REPORT.md`/`protocol.json`: многие из них дорогие, используют просмотренные периоды и не являются production entrypoints.

## Данные

- `final_solution/tabm_h3/data/` — упакованные источники H3: CBR, MOEX CNY close/fixing, OXR, Halyk, Treasury.
- `data/open_exchange_rates/rub_cis_daily.csv` — полная OXR-история с 2010 года.
- `final_solution/data_pipeline/fetch_open_data.py` — общий fetch/normalization открытых данных и manifest.
- `final_solution/data/data_manifest.json` — происхождение и статус общих public sources.

Новые joins должны быть point-in-time. Не заполняйте ранние пропуски будущими значениями. Для каждого источника сохраняйте доступную на дату решения версию, задержку и receipt.

## Документы и визуализации

- `README.md` — короткая публичная история и быстрый запуск.
- `docs/MODEL_METRICS.md` — подробные модельные метрики, uncertainty и ограничения.
- `docs/assets/model-metrics/metrics_snapshot.csv` — ровно восемь публичных строк: aggregate и 2024–2026 для H3/H5.
- `scripts/build_readme_assets.py` — проверка selections и детерминированная генерация SVG/PNG/snapshot, включая реальный timeline сигналов.
- `Presentation Artifacts/` — презентация и визуальный язык; старые численные слайды не являются финальными H3/H5 метриками.
- `product_artifacts/CLIENT_JOURNEY.md`, `V3_DECISION_CONTRACT.md`, `Библиотека_пушей.md` — продуктовый путь, контракт решения и коммуникации.

После изменения selection, модели или отчётных CSV выполните:

```bash
python3.11 scripts/build_readme_assets.py --build
python3.11 scripts/build_readme_assets.py --check
```

Если публичные агрегаты изменились, generator намеренно завершится ошибкой. Сначала пересмотрите выводы и тексты README, затем осознанно обновите expected values в скрипте.

## Legacy / archive

- `research_v3/` — старый V3 research и preview; нужен для сравнений, не является выбранным финальным профилем.
- `final_solution/alphatransfer_final/`, `config.legacy.toml`, `README.legacy.md` — legacy runtime.
- `final_solution/output-*` и `review_artifacts/generated*` — исторические/smoke outputs.
- `Researches/ml_hackathon/` — ранний baseline и постановка работ.

Не удаляйте legacy-артефакты без отдельной задачи: они нужны для аудита происхождения результатов.

## Правила аналитической целостности

1. Публичные H3/H5 цифры брать из канонических `summary.csv`/`by_year.csv` через `build_readme_assets.py`, а не переписывать из сообщений или старых слайдов.
2. `historical_smoke` показывает воспроизводимость runtime на известной истории; это не новый OOT и не live alpha quote.
3. Финальный refit после 2026-09-05 пока не имеет независимого будущего теста.
4. Не заявлять статистическое превосходство, если сохранённый CI включает ноль. Не восстанавливать CI из одних агрегированных point estimates.
5. Forward delta использует официальный reference-rate, а не фактический клиентский курс с комиссиями и банковским спредом.
6. Финальный H5 recipe — `full_history`, но код не должен хардкодить этот результат: читать `research_v4/h5_fullhistory/selection.json`, чтобы новый запечатанный отбор мог безопасно выбрать `full_history` или `120m`.
7. Слой CLOSING_H3 должен менять формулировку только существующего NOW-сообщения и не создавать дополнительный контакт; неоднородность 2025 отражать в техническом отчёте.
8. Не выбирать разные модели задним числом для разных тестовых лет.
9. Для H5 `rare65` означает calibration-only цель частоты 0,65 сигнала/неделю, а не вероятность 65%; политика не обязана создавать сигнал в молчащую неделю.

## Минимальная проверка перед передачей

```bash
# Документальные артефакты и selection guards
python3.11 scripts/build_readme_assets.py --check

# H3 runtime tests
python3.11 -m unittest discover -s final_solution/tabm_h3/tests -v

# Общие final_solution tests
python3.11 -m unittest discover -s final_solution/tests -v

# CLI metrics smoke
python3.11 final_solution/main.py --model h3 --action metrics --corridor-filter KZT
python3.11 final_solution/main.py --model h5 --action metrics --corridor-filter KZT

# H5 selection, cohort parity, maturity and cadence/gate contract
python3.11 -m unittest tests.test_h5_fullhistory_contract -v
```

Проверьте, что H3/H5 metrics CSV имеют одинаковую публичную схему, повторный infer не переобучает готовую модель и все ссылки из `README.md`/`docs/MODEL_METRICS.md` существуют.
