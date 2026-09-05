# Единый CSV для отчётности

Файл `REPORTING_ALL_METRICS.csv` содержит все числовые строки новых сравнений и включённых предыдущих отчётов. Значения не округлены. Пустое поле означает «не применимо или отсутствовало в исходном отчёте», а не ноль.

Для новых годовых сравнений фильтруйте `experiment_protocol=robust_selection`, `row_type=annual`. Затем выберите `config_id`, `train_horizon`, `evaluation_horizon`, `policy`, `evaluation_scope`, `year`. Для всех коридоров `evaluation_scope=all5`; KZT-only TabM не имеет all5 строки. `row_type=year_corridor` содержит отдельные коридоры в `corridor`.

Для финальной KZT-рекомендации: `config_id=tabm_kzt`, `train_horizon=5`, `policy=rank90`, `evaluation_scope=KZT`. Для строгой V3: `config_id=v3`, `policy=strict05`.

`row_type=cross_horizon` — фиксированные контакты на общих H5-датах, где train_horizon и evaluation_horizon могут различаться. Вероятностные метрики чужого горизонта намеренно пустые.

`row_type=selection` — ранжирование правил, не новая оценка на другом датасете. `period=development_2024_2025` означает выбор по первым двум годам; `retrospective_2024_2026` использует все три года.

`paired_interval` — кандидат минус baseline: отрицательный delta_brier лучше, положительный lift_delta/forward_delta_bps_delta лучше. `selected_utility_interval` — интервал выгоды относительно базового дня, после отбора. `post_selection_paired_interval` — описательное сравнение выбранных ретроспективно правил.

Предыдущие эксперименты имеют другие значения `experiment_protocol`; определения/популяции могут различаться. Всегда сохраняйте `source_report` и `row_type` при построении сводной таблицы. Не суммируйте aggregate+annual+year_corridor, это повторные представления одних прогнозов. Отдельный reporting_sources.json содержит список источников, числа строк и SHA256.

Доли (`hit_rate`, `week_coverage`, `mean_cell_week_coverage`) — 0–1. Денежные относительные показатели — базисные пункты, 1bp=0,01%. Lift — безразмерное отношение. Coverage не является вероятностью истинности сигнала. Число сигналов и недель — целые счётчики; signals_per_corridor_week — их отношение по целым наблюдаемым календарным неделям в новых экспериментах. Исторические поля оставлены в исходных определениях и требуют фильтрации по источнику.
