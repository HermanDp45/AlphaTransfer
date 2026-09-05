# Проверка объёма работы

Цель пользователя сохранена целиком. Ни наличие красивого отчёта, ни улучшение одной метрики не считаются доказательством всех требований.

| Требование | Проверяемое выполнение | Ограничение результата |
|---|---|---|
| Изучить кейс, Q&A, существующее решение и review/product artifacts | METHODOLOGY_REVIEW, REPORT, V3_DECISION_CONTRACT; численный аудит baseline и текста | Старые материалы сохранены, неправильные семантические утверждения исправлены |
| Улучшить модели и сравнить метрики | 39 конфигураций core harness, отдельные TabM/data ветки; COMPARISON.csv, paired CIs, code и receipts | Нет модели, доминирующей по всем критериям; это явно отражено |
| Исследовать ранее исключённые Bloomberg/FRED/другие | Реальные FRED/CBOE/ICE fits; source_access_decisions и REPORT; Bloomberg access investigation | Bloomberg export отсутствует: эффект неизвестен, эксперимент не выдуман |
| Отдельно отметить гипотетически свободное использование | External-data tier и условия в REPORT; FRED/Treasury numerical parity | Restricted raw inputs не объявлены свободным distribution bundle |
| Увеличивать train intervals | CBR 2010–2026; 3/6/12/24/36/60/120/expanding, monthly refits, early CNY ablation | Старые exogenous features не имеют полного покрытия с 2010 |
| Пользовательская синтетика и частотные группы | Simulator, evidence map, 12 scenarios/3 seeds, segment and equal-budget results, checks | Без банковских данных нельзя доказать реалистичность банка или causal uplift; это прямо указано |
| Развить сами задачи/метрики | Forecast/factual/behavior decomposition, risk–coverage frontier, temporal/sparse-support audit | Новые proxy не переименованы в реальную банковскую выгоду |
| Найти новые сильные данные | Treasury real yields, CBOE, GPR, ICE, longer MOEX/CBR; prioritized executable-quote/flow acquisition plan | Большой эффект общей macro-покупки не установлен |
| Реализовать результат в решении | final_solution/main.py --research-v3; facts.py, behavior.py, product.py; example decision | Исторический preview, реальные сообщения не отправляются |
| Описать все подходы и вклад | REPORT + FINAL_COMPARISON +166-row COMPARISON +branch reports | Частота, proper scoring и synthetic эффекты разделены |
| Проверить воспроизводимость | verify.py, per-model receipts, full raw/code/output hashes, baseline parity, unit/temporal/CLI checks | Новые независимые будущие данные остаются предметом реального пилота |

Свидетельства запуска тестов — `validation.json`; независимая hash/coverage/numeric проверка — `_SUCCESS.json`. Финальный отчёт не заявляет возможности получить недоступные данные или подтвердить реальный эффект без измерений.
