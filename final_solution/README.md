# AlphaTransfer: финальные TabM KZT H3 и H5

Доступны два профиля на истории с 2010 года: регулярный **H3** и более редкий **H5** для другого горизонта и пользовательской группы. Все 33 признака, preprocessing, калибровка, источники и состояние триггера упакованы отдельно в `tabm_h3/` и `tabm_h5/`.

## Запуск

```bash
python3.11 -m pip install -r final_solution/requirements-tabm-h3.txt
python3.11 final_solution/main.py --as-of 2026-09-05 --mode historical_smoke
```

Команда строит признаки из семи упакованных источников и пишет `tabm_h3/output/signal_decision.json`, `predictions.csv`, `next_state.json`, `run_receipt.json`. Это явно помеченная историческая демонстрация: модель имеет as-of5сентября, последний доступный справочный курс —3сентября. Демо не является новым out-of-sample тестом и не отправляет сообщения клиентам.

Для обновлённых источников:

```bash
python3.11 final_solution/main.py --as-of YYYY-MM-DD --mode operational \
  --sources /absolute/path/source_paths.json \
  --state-in final_solution/tabm_h3/operational_state.json \
  --state-out /absolute/path/next_state.json
```

JSON источников использует те же ключи, что `source_paths` в `tabm_h3/bundle.json`; абсолютные пути допустимы. Данные должны продолжать полный прежний префикс без пропуска CBR-сессий. Operational mode проверяет доступность даты и не выдаёт предшествующие cutoff строки за новые прогнозы. После успешного запуска передавайте сохранённое состояние следующему запуску.

## Выбранный рецепт

| Год теста | Lift | Hit rate | Недель с1–2 сигналами | Выгода к базе, б.п. |
| --- | --- | --- | --- | --- |
| 2024 | 1,609 | 53,8% | 80,4% | +57,5 |
| 2025 | 1,452 | 38,1% | 92,2% | +40,3 |
| 2026, доступная часть | 1,441 | 61,1% | 84,8% | +73,8 |

Длинная история закрыла провал прежнего10-летнего H3 в2026, но ухудшила2025. Во всех трёх годах выполнены точечные цели lift≥1,3, coverage≥80%, положительная разность к базовому дню. Доверительные интервалы общего улучшения относительно10-летнего варианта включаютноль; гарантированное превосходство не заявляется.

Финальные веса используют3869наблюдений:2010-01-01–2025-08-30. Последние243зрелых наблюдения до2026-08-29 использованы отдельно для калибровки и триггера; их исходы известны к2026-09-03. Это использование всей истории с разделением train/calibration, без обучения калибратора на ответах, уже виденных весами. Исторические показатели относятся к ежегодному recipe; новый final checkpoint ещё не имеет будущего теста.

## Сигналы

NOW_H3 означает прогноз, что текущий RUB/KZT не выше минимума следующих3эффективных CBR-сессий. Rank80 сравнивает текущий score с63предшествующими; квантиль выбирается по прошлой калибровке. Максимум2кандидата за календарную неделю, cooldown2сессии. Состояние защищено привязкой к модели, горизонту, калибровке и правилу.

Отдельная CLOSING_H3-голова усиливает существующее NOW-сообщение формулировкой «окно может закрыться» и не добавляет контактов. На rolling OOT она уточнила 61 из 151 NOW-сигнала; направление подтвердилось в 67% случаев. Годовой эффект неоднороден и подробно раскрыт в корневом техническом отчёте.

## Альтернативный редкий H5

H5 предсказывает выгодный момент на горизонте пяти эффективных сессий. Политика `rare65` калибруется на прошлых score для частоты около 0,65 сигнала в неделю, использует cooldown 2 сессии и максимум 2 контакта в неделю. Rolling OOT 2024–2026: lift 1,986×, hit rate 52,8%, +95,6 б.п. и 89 сигналов. Full-history вариант выбран против точного 120m-контроля по заранее заданным gates устойчивости и частоты.

## Где смотреть

- `tabm_h3/bundle.json` — активная конфигурация и выбранная история.
- `tabm_h3/features.py`, `model.py`, `policy.py`, `predict.py` — самостоятельный исполняемый путь.
- `tabm_h3/model/`, `data/`, `source_receipt.json`, `training_receipt.json` — веса, данные, происхождение.
- `../research_v4/h3_finalization/REPORT.md` — подробное сравнение длин истории.
- `../research_v4/h3_finalization/REPORTING_ALL_METRICS.csv` — единый CSV всех сравнений.

Прежний путь доступен через `python3.11 final_solution/main.py --legacy`; его документация сохранена в `README.legacy.md`. Корневой `config.toml` относится кlegacy-контру; активная H3-конфигурация находится в `tabm_h3/bundle.json`.


## Единый CLI (новый)

Новый единый интерфейс запускается через флаги `--action` и/или `--model`:

```bash
python3.11 final_solution/main.py --model h3 --action infer --as-of 2026-09-05 --mode historical_smoke
python3.11 final_solution/main.py --model h3 --action train --force
python3.11 final_solution/main.py --model h3 --action metrics --corridor-filter KZT --as-of-from 2024-01-01 --as-of-to 2024-12-31

python3.11 final_solution/main.py --model h5 --action infer --corridor-filter KZT
python3.11 final_solution/main.py --model h5 --action train --force
python3.11 final_solution/main.py --model h5 --action metrics --corridor-filter KZT
```

`metrics` и `infer` автоматически запускают обучение, если артефакты отсутствуют (или сломаны),
если не указан `--force` и артефакты валидны, повторная выдача идёт без переобучения.

Результаты метрик всегда пишутся в CSV через общий формат в `--output-dir`:
- `model`
- `action`
- `dataset`
- `period`
- `corridor_filter`
- `horizon`
- `threshold_policy`
- `threshold`
- `hit_rate`
- `lift`
- `forward_delta_bps`
- `regret_bps`
- `signals_per_corridor_week`
- `mean_cell_week_coverage`
- и диагностические столбцы строк/дней/сигналов.

Устаревшие режимы сохранены:
- `python3.11 final_solution/main.py --legacy`
- `python3.11 final_solution/main.py --research-v3`
- а также прямой запуск H3 без новых флагов как раньше:
  `python3.11 final_solution/main.py --as-of 2026-09-05 --mode historical_smoke`
