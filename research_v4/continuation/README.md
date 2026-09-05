# Продолжение V4

[Итоговый отчёт](REPORT.md) и [таблица сравнений](COMPARISON.csv).

- [OXR](oxr/REPORT.md): 95 обучений, временные отсечки и глубина источника; [код](oxr/experiment.py), [независимый аудит](oxr/audit/REPORT.md).
- [Chronos и NOW](foundation/REPORT.md): длинные головы, точная база V3, MPS/CPU, контроль300/900 шагов. Инструкции запуска внутри отчёта.
- [Задержка Halyk](robustness/REPORT.md): неизменённая модель при задержке, регуляризация, возраст источника и два frozen cutoffs.

Из каталога AlphaTransfer, окружениеPython3.11 с requirements-market.txt и foundation/requirements.txt исходногоV4:

```sh
python research_v4/continuation/oxr/experiment.py --phase development
python research_v4/continuation/oxr/experiment.py --phase test
python research_v4/continuation/oxr/history_sensitivity.py
OPENBLAS_NUM_THREADS=1 python research_v4/continuation/oxr/assess.py
python research_v4/continuation/oxr/build_report.py
python research_v4/continuation/build_report.py
python3 research_v4/continuation/verify.py
python3 research_v4/verify_package.py --seal
```

Модельные проверки в каждой ветке восстанавливают прогнозы и проверяют причинность; verify.py проверяет их результаты, исходныйV3 и предыдущийV4. После научных перезапусков пакет нужно запечатать заново, поскольку outputs/receipts изменяются. Сохранять старый протокол выбора как будто он создан до новых данных нельзя: новый запуск должен иметь собственный протокол/каталог.

`previous_v4_manifest.json` — точная копия старой печатиV4. `prior_reports` сохраняет исходные версии REPORT/README. Навигация родительскогоV4 обновляется отдельным шагом после завершения всех проверок; исходные модели и научные результаты не меняются.
