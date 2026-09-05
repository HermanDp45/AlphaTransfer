# Длинные HGB-модели: OXR2010 и Halyk

[Итоги](../REPORT.md) · [Все значения](summary.csv) · [Парные интервалы](paired_intervals.csv) · [Одновременные интервалы27 моделей](simultaneous_intervals.csv) · [Проверка](verification.json)

27 конфигураций, по пять временных cutoff: 135 fits. Базовый HGB: 120 деревьев, depth2, learning rate0.05, min leaf40, L2=2, early stopping выключен. Параметры прежние; увеличение истории не сопровождается скрытым увеличением capacity. Полная история доступна с2010, поэтому вариант180m фактически имеет12–15 лет train в зависимости от cutoff. Перед test выделен год calibration, обе границы очищены по фактическому времени появления пятой будущей котировки.

После аудита добавлены ещё35 численных контролей, итого170 fits. В [canonical_control.py](canonical_control.py) OXR-признаки на общем периоде после01.09.2018 сделаны точно одинаковыми с2018-представлением; настоящая ранняя2010-история сохранена. Полное переобучение дало ровно те же50 250 прогнозов и все сигналы: [retraining_parity.json](canonical/retraining_parity.json). Эти проверки не выбирали новую модель. В исходной HGB-матрице27 конфигураций, дополнительные7 являются численными контролями.

Основная группа: [protocol.json](protocol.json), [experiment.py](experiment.py), [выбор до нового2026](selection.json), [checkpoints и построчные прогнозы](output/).

Treasury-группа: [фиксированный addendum](treasury/protocol.json), [код](treasury_addendum.py), [результаты](treasury/development_summary.csv). Она добавляет признаки из прежнего frozen Treasury-panel с лагом7, покрытие начинается в2020. Три [контроля вклада банка/адаптации](bank_controls/protocol.json) добавлены после первого прочтения результатов; параметры не подбирались, все результаты сохранены.

Новый кандидат Treasury+Halyk+KZT shrink найден среди этих вариантов по development. Он не заменяет автоматически V3 и прежний Halyk: выигрывает по Brier, но уступает прежнему Halyk по reference-policy-bps и чувствителен к задержке банка.

## Корректный смысл лагов

OXR доступен не раньше max(конец наблюдаемого дня, published_at)+24h; стресс добавляет ещё24h. Halyk —effective chart date+1 календарный день, стресс+2. Это допущение исторической доступности банковского архива, не доказанный timestamp публикации.

В normal/stress используются одни веса, калибратор, порог и состояние выбора до cutoff. Внезапная задержка начинается после cutoff. Derived rolling features пересчитываются со сшитой историей: normal до cutoff, delayed после. При аудите было исправлено первоначальное использование delayed history до cutoff. [Исходный код fit](fit_engine_snapshot.py) соответствует hash протокола; [исправление inference](repair_delay_onset.py) не переобучает модель. [Журнал изменений](onset_repair_changes.csv) и [receipt](onset_repair.json) сохраняют происхождение результата.

До исправления aggregate-файлы сохранены в [before_onset_repair](before_onset_repair/); в Treasury аналогичный каталог. Старый selection SHA относится именно к этому immutable development-файлу. Normal probabilities/candidate decisions совпадают после исправления, выбор модели не изменился. Исправленные checkpoint receipts содержат старый и новый prediction SHA. Все итоговые summary/CI построены по исправленным прогнозам.

## Воспроизводимость

Python3.11, numpy/pandas/sklearn и точные версии зафиксированы в [verification.json](verification.json). Рабочее окружение этого запуска: `/private/tmp/alphatransfer-tabm-venv/bin/python`. Данные локальные, сетевых запросов и новых нейросетевых вычислений нет.

На копии каталога, чтобы не переписать sealed artifacts: `experiment.py`, затем `treasury_addendum.py`, затем `bank_controls.py` выполняют все fits. Текущий engine уже содержит исправленный onset, поэтому новый запуск не нуждается в repair. `assess.py` строит таблицы и парные интервалы, `simultaneous.py` —дополнительные bands. `verify.py` предназначен для проверки именно сохранённого исторического запуска с его исходным snapshot/repair chain; он воспроизводит135 моделей и285 model/view комбинаций и проверяет точное совпадение основных прежних V3/V4 контролей.

Bootstrap:10 000 повторов, seed20260905; месяц,20/60 дат. Основные CI условны на фиксированные модели. Одновременные max-statistic bands охватывают27 текущих normal HGB-вариантов KZT против одного V3long; прошлый поиск V3/V4, другие targets и перебор задержек этим не покрываются.

`forward_delta_bps` —условное преимущество относительно official-reference baseline год×валюта на отобранных сигналах. Это не банковская цена и не измеренная экономия пользователей.
