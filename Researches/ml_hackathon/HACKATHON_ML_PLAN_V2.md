# AlphaTransfer: актуальный план после Q&A и экспериментов `edelkin_test`

Версия: 03.09.2026. Этот документ заменяет первоначальный общий ML-roadmap. Основания: последняя [сводка Q&A](<../../Q&A для команд — сводка 20260903.md>), полный [банк ответов](<../../Вопросы и ответы.md>), диагностический baseline в `main` и коммиты `09cc060…745fdc3` ветки `origin/edelkin_test`.

## Решение

Следующая задача команды — **не придумывать новые модели, а исправить контур оценки и объединить уже сделанные части**.

В `edelkin_test` уже есть сильный инженерный V0 для RUB→KZT: ЦБ РФ + НБК + MOEX, нормализация, freshness, 49 point-in-time признаков, expanding walk-forward, logistic regression, shallow boosting, policy, CLI `--as-of`, HTML-отчёт и stale-state прототип. Параллельно в `main` уже есть простой evaluator пяти коридоров и всех пяти горизонтов.

Но headline-метрики V0 пока нельзя защищать:

- truth `favorable_now` не совпадает с Q&A;
- truth `window_closing` является частью самого trigger и не использует будущее;
- validation не очищена от последних `h` наблюдений перед test;
- Brier score считается не по вероятностям модели на всех днях;
- частота скрывает пять последовательных folds без единого сигнала.

Поэтому правильная стратегия — **repair → rerun → decide**, а не новый model zoo.

## 1. Что именно говорит актуальная Q&A

### Зафиксировано кейсодателем

| Вопрос | Решение для команды |
|---|---|
| Цель кейса | Триггерная коммуникация без скидки/надбавки; это не trigger pricing |
| Price prediction | Не является целью; гипотеза кейса должна решаться проще |
| `favorable_now` hit | Курс в день сигнала остался не хуже следующие `h` дней |
| Горизонты | Обязательны все `1/3/5/10/20`; главного нет |
| Random baseline | Случайный день того же коридора и периода |
| Lift | Отношение hit rates; bps — отдельная метрика |
| Устойчивость | Несколько коридоров и out-of-time окна; независимость коридоров не требуется |
| Выгода | Основная метрика из задания — относительно среднего в `±h`; forward-only допустима дополнительно с обоснованием |
| Частота | 1–2 сигнала на коридор в неделю — самопроверка модели; клиентский CRM-лимит отдельный и вне scope |
| Данные | Только открытые и воспроизводимые; источник и гранулярность выбирает команда |
| Cutoff | Обязательна воспроизводимая функция «сигнал как на дату T» |
| Клиенты/A-B | Инкрементальный объём, каннибализация и CTR на хакатоне не измеряются |
| Сдача | Нужны инструкция запуска, ограничения/компромиссы и краткий план пилота |

### Оставлено на решение команды

- tolerance `δ` и продуктовый порог выгоды;
- конкретная формула truth для `window_closing`;
- исключение перенесённых дней из signal/random eligibility;
- трактовка 2022 года;
- поведение в тихий месяц;
- шкалы силы/скорости;
- сценарии, которые превращаются в push, и сценарии молчания;
- UX, deeplink и предзаполнение.

Каждое такое решение должно иметь sensitivity-проверку или явное обоснование. Нельзя выдавать его за требование кейсодателя.

## 2. Реестр уже проведённых экспериментов

### E0 — старое поколение: ЦБ РФ, пять коридоров

Источник: `reports/2026-09-01_backtest_summary.md` в `edelkin_test`; код сохранён только в коммите `bca8ace`.

- 01.01.2020–01.09.2026, по 1 641 публикации на AMD/KGS/KZT/TJS/UZS;
- expanding walk-forward 252 train / 63 test, `h=5`;
- ни rules, ни logistic regression не прошли одновременно lift ≥1,3 и частоту;
- ближайший результат — KZT `window_closing`, lift 1,27;
- ML KZT — lift 1,26 и 2,60 сигнала/неделю.

Статус: **полезный отрицательный результат, но не финальный benchmark**. Причины: только `h=5`, устаревший код удалён из актуального дерева, definitions надо сверить с текущей Q&A.

### E1 — текущее поколение V0: KZT, три источника

Источник: `data/kzt_v0/backtest.json` в `edelkin_test`.

- 1 295 модельных MOEX-сессий, 11.05.2021–03.09.2026;
- 7 folds: 36 месяцев train → 6 validation → 3 untouched test;
- 49 признаков; logistic regression и самописный shallow boosting;
- 58 raw candidates, 17 policy-eligible сигналов;
- первые 5 из 7 test folds — без единого кандидата;
- фактическая частота на полном test-span 638 дней — **0,187 сигнала/неделю**, а не целевые 1–2;
- `favorable_now`, `h=5`: 7 сигналов, lift 10,57 по CBR, но lift 0 по NBK и MOEX;
- `window_closing`, `h=5`: 10 сигналов, hit rate 100%, random 76,7%, lift 1,30.

Статус: **инженерный каркас сохранить, численные выводы пересчитать**.

### E2 — независимый диагностический baseline в `main`

Источник: `scripts/hackathon_baseline.py` и `Researches/ml_hackathon/results/`.

- OXR cross-rate, пять коридоров, 2018–2026;
- четыре rules, все пять горизонтов, yearly OOT, block-bootstrap CI;
- ни одна из 100 комбинаций не прошла все гейты;
- 55 — negative control, 45 — research only;
- результаты подтверждают нестабильность по годам и смещение `±h` для momentum.

Статус: **независимый evaluator/sanity check**, но не финальный источник истины: OXR не равен официальному ЦБ или execution rate.

## 3. Аудит `edelkin_test`: что сохранить, что исправить

### Сохранить и перенести

- нормализацию CBR/NBK/MOEX и явные единицы `RUB_per_KZT`;
- разделение `effective_date` и `available_at`;
- календарь, freshness и запрет создавать momentum на fill-днях;
- point-in-time feature builder;
- CLI `fetch/backtest/signal/report/all` и `--as-of`;
- logistic regression как прозрачный challenger;
- cooldown, cap и причины suppression;
- deterministic HTML report;
- stale-state UX и автономный prototype;
- существующие тесты normalization, future-invariance, stale fill, policy и reproducibility.

### Исправить до любого нового эксперимента

1. **`favorable_now` truth.** Сейчас `local_minimum_label` требует уникальный минимум в `±h`. Q&A требует только `S_t ≤ min(S_{t+1…t+h})`. Прошлое может быть признаком и участвовать в bps, но не должно дисквалифицировать hit.
2. **`window_closing` truth.** Сейчас `window_closing_label` проверяет уже случившийся rebound от прошлого минимума и игнорирует `h`. Это trigger condition, не future truth; 100% hit rate тавтологичен.
3. **Validation purge.** Последние `h` строк validation получают labels из будущего test. Их надо исключить при выборе модели, threshold и rebound threshold.
4. **Калибровка.** Brier сейчас получает `confidence` только policy-eligible events, а всем остальным дням приписывает `0`; для `window_closing` это вообще эвристическая confidence. Нужны raw OOT probabilities на каждом eligible day и отдельная calibration для каждого target.
5. **Агрегация baseline.** Hit rate signal взвешивается по числу сигналов, random rate — простым средним folds. Оба числителя и знаменателя надо агрегировать по исходным counts либо считать единую OOT confusion table.
6. **Частота.** Считать по полному eligible test-time, включая folds без кандидатов; показывать `signals/week`, долю silent weeks, max gap и распределение по folds.
7. **Source robustness.** Три источника полезны, но gate «2 из 3» — решение команды, не требование Q&A. Его показываем отдельно от обязательной переносимости между коридорами.
8. **Размерность.** 49 признаков на 1 295 строк и 7 folds — высокий риск нестабильности. До новых моделей сделать feature ablation и сократить core set примерно до 10–15 признаков.

## 4. Рабочие определения V2

| Сущность | Primary definition | Статус |
|---|---|---|
| Rate | `RUB за 1 LCY`, меньше лучше отправителю | Зафиксировано |
| Decision time | Используются только observations с `available_at ≤ T` | Зафиксировано |
| `NOW-hit(h)` | `S_t ≤ min(S_{t+1…t+h})` | Зафиксировано Q&A |
| `NOW-hit(h,δ)` | `S_t − min(future) ≤ δ` | Sensitivity команды |
| `CLOSING-hit(h,δ)` | `max(S_{t+1…t+h}) − S_t ≥ δ` | Primary-допущение команды |
| Closing endpoint | `S_{t+h} − S_t ≥ δ` | Robustness sensitivity |
| Advantage bps | `(mean(S_{t-h…t+h}) / S_t − 1) × 10 000` | Обязательная метрика |
| Forward bps | `(mean(S_{t+1…t+h}) / S_t − 1) × 10 000` | Дополнительная диагностика |
| Random eligibility | Те же source/calendar/freshness/test folds, что у модели | Зафиксировано + допущение календаря |
| Model frequency | Сигналы / все eligible недели коридора | Зафиксировано как self-check |

Primary run должен использовать `δ=0`; `25/50/75/100` б.п. — sensitivity. Барьерная трёхклассовая разметка не заменяет официальные hit-метрики.

## 5. Актуальный backlog

### P0 — ремонт доказательной базы

| ID | Статус | Задача | Критерий приёмки |
|---|---|---|---|
| EVAL-01 | Переделать | Две future-truth функции по формулам выше | Toy-tests различают past trigger и future outcome; результат меняется с `h` |
| EVAL-02 | Переделать | Purge/embargo на train и validation границах | Ни один label, участвующий в selection, не касается test |
| EVAL-03 | Переделать | Counts-based aggregate и random baseline | Aggregate воспроизводится суммой TP/N; zero-signal folds не исчезают |
| EVAL-04 | Переделать | Реальная frequency/clustering | Полный test-span, silent-week share, max gap, fold distribution |
| CAL-01 | Переделать | OOT probabilities и calibration | Brier/reliability на всех eligible days отдельно для NOW/CLOSING |
| TEST-01 | Дополнить | Contract tests из Q&A | Exact NOW, closing future truth, validation-boundary leakage, aggregation, frequency denominator |
| RUN-01 | Повторить | KZT V0 после исправлений | Старые headline 10,57/1,30 помечены invalidated; новый report generated |

### P0 — консолидация покрытия кейса

| ID | Статус | Задача | Критерий приёмки |
|---|---|---|---|
| DATA-ALL | Частично есть | Вернуть официальный CBR ingestion пяти коридоров в архитектуру V0 | AMD/KGS/KZT/TJS/UZS, номиналы, timestamps, одна схема |
| RULES-ALL | Повторить | Down streak, level, rebound, seasonality на новом evaluator | Полная матрица `indicator × corridor × h` |
| WF-ALL | Новое | Единые временные folds для всех коридоров | Параметры проверяются переносом, один период не оказывается train и test одновременно |
| SOURCE-KZT | Уже есть | CBR/NBK/MOEX как robustness KZT | Отдельная таблица, не подмена multi-corridor gate |
| CUTOFF | Уже есть | Усилить `signal --as-of` массовой проверкой | ≥100 случайных T: full-prefix = physically truncated run |
| OUTPUT | Частично есть | Требуемая signal schema | date, corridor, indicator, direction, strength, speed, scenario, suppression reason |

### P1 — только после зелёного evaluator

| ID | Эксперимент | Решение |
|---|---|---|
| M1 | Rules-only policy | Обязательный baseline и возможный финальный ответ |
| M2 | Cost-sensitive logistic regression на 10–15 признаках | Перезапустить существующую реализацию, не писать заново |
| M3 | Существующий shallow boosting | Оставить только если улучшает worst-fold и не ломает frequency |
| M4 | Ablation: price-only / +calendar / +cross-source | Выяснить, какие 49 признаков действительно добавляют OOT value |
| M5 | Rolling 2–3 года против expanding | Проверить structural drift, раз сигналы появились только в 2026 |
| M6 | Common RUB factor + corridor residual | Пробовать после multi-corridor baseline как компактный feature block |

ML считается полезной только если обгоняет rules по counts-based OOT lift, положительной `±h` выгоде, worst-fold и частоте. Победа по одному pooled числу не засчитывается.

### P2 — только если M1–M6 дали устойчивый сигнал

- barrier `NOW/WAIT/INDIFFERENT` как дополнительная decision sensitivity;
- простой regime gate (`vol_5/vol_60`, trend/reversal), не HMM;
- quantile/hazard estimate цены ожидания;
- внутридневной survival сигнала;
- новости только как veto.

EBM, CatBoost, HMM, сложный ансамбль и новая макродата сейчас не являются следующим шагом: они не исправят неверный target и малое число OOT events.

## 6. MVP: не строить заново

В `edelkin_test` уже есть:

- deterministic HTML report;
- self-contained `star/prototype.html`;
- stale signal suppression;
- current no-candidate state;
- объяснения факторов и freshness.

Нужно заменить данные на output исправленного evaluator и показать четыре кейса:

1. `NOW` с последующим hit;
2. дорогой false positive `NOW`;
3. `WINDOW_CLOSING` с future confirmation;
4. `SILENCE` из-за stale/low confidence/conflict.

Отдельный новый frontend, push-service и интеграция с приложением не нужны. В интерфейсе future overlay скрыт до режима «проверить, что произошло потом».

## 7. Очерёдность работ

### Шаг 1 — evaluator, 2–3 часа

- EVAL-01…04 и пять новых contract tests;
- пометить старые KZT headline-метрики как invalidated;
- повторить KZT run без изменения features/models.

### Шаг 2 — честное решение по V0, 1–2 часа

- если сигнал исчез — зафиксировать отрицательный результат;
- если остался — проверить source robustness и fold concentration;
- выбрать rules или logistic только по frozen criteria.

### Шаг 3 — пять коридоров, 3–5 часов

- перенести CBR ingestion из `bca8ace`, но не старый evaluator;
- построить полную матрицу пяти горизонтов;
- прогнать единые folds и перенос параметров.

### Шаг 4 — защита и MVP, 2–3 часа

- подключить новые JSON/CSV к существующему отчёту и prototype;
- сделать один слайд «что сломали собственным аудитом и как исправили»;
- добавить инструкцию запуска, ограничения и компактный план пилота.

## 8. Stop/go criteria

### GO как хакатонное доказательство сигнала

- exact Q&A truth и zero leakage подтверждены тестами;
- хотя бы один сценарий имеет lift ≥1,3 на нескольких коридорах и нескольких OOT folds;
- `±h` bps статистически >0;
- frequency находится около 1–2/неделю на полном test-span, а не только в активных folds;
- результат не держится на единственном годе или источнике;
- rules/logistic decision объясним и воспроизводим на `--as-of T`.

### NO-GO для коммуникации, но хороший результат хакатона

- evaluator чистый, однако критерии выше не выполнены;
- команда показывает отрицательные эксперименты, цену ожидания и работающий no-signal MVP;
- следующий тест формулируется из конкретного failure mode, а не как «добавим более сложную модель».

## 9. Что остаётся про пилот

Q&A требует **план пилота как сопровождающий артефакт**, но не проведение пилота. Достаточно одной страницы: необходимые данные банка, random holdout, net-volume/маржа и каннибализация, срок набора событий при фактической частоте, stale-rate/complaint guardrails и ключевые риски. Это не должно конкурировать по времени с исправлением evaluator.
