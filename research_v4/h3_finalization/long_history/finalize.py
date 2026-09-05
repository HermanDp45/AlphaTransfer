"""Summarize annual history control and final H3 artifacts; no model refits."""
from pathlib import Path
import sys,json
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import pandas as pd
from research_v4.h3_finalization.long_history import experiment as e
def main():
    final=e.HERE.parent/'final_fit';closing=e.HERE.parent/'closing'
    annual=json.loads((e.HERE/'verification.json').read_text());fv=json.loads((final/'verification.json').read_text());cv=json.loads((closing/'verification.json').read_text());iv=json.loads((closing/'isolated_runtime_verification.json').read_text())
    assert all(v['status']=='PASS' for v in (annual,fv,cv,iv))
    receipt=json.loads((final/'receipt.json').read_text())
    yearly=pd.read_csv(e.HERE.parent/'by_year.csv');selected=yearly[yearly.config_id.eq('tabm_kzt_fullhistory')&yearly.policy.eq('rank80')]
    rows=['| Год | Lift | Недель с 1–2 сигналами | Эффект к среднему дню, bps |','| --- | --- | --- | --- |']
    for q in selected.itertuples():rows.append(f'| {q.year} | {q.lift:.6f} | {q.week_coverage:.2%} | {q.forward_delta_bps:.2f} |')
    table='\n'.join(rows)
    report=f'''# Итоговая KZT H3: история с 2010 года и финальное обучение

Обучены три ежегодные модели TabM с расширяющейся историей и финальная модель на актуальном проверенном срезе. Основной эксперимент выбрал **историю с 2010 года и политику rank80**: она выполняет заданные условия во всех трёх годах. Минимум покрытия —80,39% недель, минимум lift —1,4411, минимальный эффект к среднему дню —40,25bps. Это ретроспективно выбранная конфигурация; превосходство всех моделей данного семейства из неё не следует.

{table}

Ежегодный контроль изменяет только длину обучения. Вместо120месяцев используются все доступные наблюдения начиная с1января2010года; FULL33, seed20261105, модель KZT и горизонт H3 сохранены. Получены3210/3457/3705обучающих строк для2024/2025/2026: на747/993/1240больше десятилетнего контроля. Три новых нейросетевых обучения, включая внутренний выбор эпох, заняли около30секунд CPU. Даты, исходы и признаки калибровки, проверки и начального контекста побитово совпадают с соответствующим120m-контролем.

В панели действительно есть данные KZT с2010-01-01:249наблюдений за2010год. OXR начинает фактически использоваться с2010-01-12; полный набор его скользящих признаков появляется с2010-01-23. Halyk представлен с2020-01-10, Treasury —с2020-01-01. Признаки ранних лет не выдуманы и не заполнены будущими значениями: пропуски обрабатываются train-only импутацией и индикаторами отсутствия. Подробная доступность каждого поля находится в `source_availability.csv`.

Метка H3 сохранена: текущая рублёвая цена не выше минимума следующих трёх наблюдений CBR. Дата зрелости —третье следующее наблюдение. Обучение заканчивается до12месячного периода калибровки, включая строгую проверку зрелости меток; число эпох выбирается на последней части самого обучения. Сырые вероятности до калибровки не улучшились во всех годах: Brier полной истории —0,214406/0,166059/0,184031, а120m —0,211875/0,160223/0,192791. Выбор сделан по одинаково оценённой устойчивости политики, а не по утверждению о безусловном улучшении Brier. Общие интервалы и калиброванные метрики представлены в родительском отчёте.

## Финальная модель

Использован проверенный `audit/latest_panel.pkl`:4118наблюдений KZT до2026-09-03, точное совпадение33признаков на4116старых строках подтверждено независимым аудитом. SHA256 панели: `bf061d7bc487c55e16e00515587300680e24c0fd0bbbf04326dbc7d4b7152ada`.

Дата финального среза —2026-09-05. Обучение покрывает2010-01-01..2025-08-30:3869строк; последняя обучающая метка созревает2025-09-04, строго до начала калибровки2025-09-05. Калибровочная часть содержит243созревшие строки до2026-08-29; последняя её метка созревает2026-09-03. История содержит246строк. Модель обучена заново,40эпох выбраны только внутри обучающей части. Калибровочные примеры в обучение не входят.

Экспортированы validation/history/warmup/tail в `final_fit/raw_predictions.csv.gz`. Warmup содержит63даты панели перед началом калибровки, включая исключённый обучающий хвост, с полностью скрытыми исходами. Незрелый хвост —2026-09-01/02/03. Его прогнозы являются текущими оценками финального среза; они не выдаются за исторические вневыборочные прогнозы этих дат. Калибратор, пороги rank80 и состояние политики формирует родительский пакет.

Веса: `final_fit/model/weights.pt`, препроцессор: `preprocess.joblib`, описание: `model.json`. Сохранён обычный state_dict TabM и sklearn-препроцессор; пользовательские исследовательские классы не сериализованы в модель.

## Дополнительная аннотация CLOSING

Отдельно обучены четыре HGB: ежегодные2024–2026 и финальный. Собственная метка —строгое условие `R[t+3] > R[t]`. Входы, история и временные границы соответствуют выбранной H3-конфигурации. `closing/final/model.joblib` —полностью стандартный sklearn Pipeline, включающий препроцессор и HGB. Загрузка в изолированном Python без пути проекта проверена; прогнозы хвоста совпадают точно.

Предусмотренное правило аннотации: вероятность не ниже0,5, текущий `ret1 > 0` и уже выбранный NOW. Дополнительных контактов оно не создаёт. Важно различать метки: истинный NOW почти логически влечёт положительный конечный курс, кроме равенств; в рассматриваемых годах все213истинных NOW имеют CLOSING=1. Поэтому дополнительную аннотацию нельзя представлять как независимое подтверждение другого реализованного финансового события. Её оценка выполняется отдельно на предсказанных сигналах.

## Проверки

Все проверки имеют статус PASS: независимый пересчёт H3 и дат зрелости, train-only препроцессинг, три ежегодных и финальный checkpoints,63-дневный контекст с исключённым хвостом, точное совпадение контрольных данных и повторные прогнозы. Четыре CLOSING-модели также воспроизведены из sklearn checkpoints. `final_verification.json` связывает проверки ежегодной, финальной и дополнительной моделей. Старые sealed-артефакты не изменены; дальнейшая упаковка и итоговые метрики находятся в родительском эксперименте.
'''
    import re
    report=re.sub(r'(?<=[А-Яа-яёЁ])(?=\d)|(?<=\d)(?=[А-Яа-яёЁ])',' ',report)
    report=report.replace('—80','— 80').replace('—1','— 1').replace('—40','— 40').replace('—с','— с').replace('—третье','— третье').replace('—0','— 0').replace('—2026','— 2026').replace('—полностью','— полностью').replace('seed20261105','seed 20261105').replace('FULL33','FULL33').replace('40,25bps','40,25 bps').replace('последней части самого обучения','последних 63 датах самого обучения')
    (e.HERE/'REPORT.md').write_text(report)
    (e.HERE/'README.md').write_text('# KZT H3: расширяющаяся история\n\n[Отчёт](REPORT.md) · [Связанные проверки](final_verification.json)\n\nТри annual fit и финальная H3-модель готовы. Финальные веса и raw exports: `../final_fit/`. Дополнительный sklearn CLOSING: `../closing/`.\n')
    external={}
    for p in [final/'verification.json',final/'receipt.json',final/'raw_predictions.csv.gz',final/'model/weights.pt',final/'model/preprocess.joblib',closing/'verification.json',closing/'isolated_runtime_verification.json',closing/'final/model.joblib',closing/'raw_predictions.csv.gz']:
        external[str(p.relative_to(ROOT))]=e.sha(p)
    e.save(e.HERE/'final_verification.json',dict(status='PASS',annual_tabm_fits=3,final_tabm_fits=1,inner_neural_fits=4,closing_hgb_fits=4,horizon=3,scope='KZT only',final_history='from2010',final_cutoff='2026-09-05',annual_verification_sha256=e.sha(e.HERE/'verification.json'),final_weights_sha256=receipt['weights_sha256'],latest_panel_sha256=receipt['panel_sha256'],external_artifacts=external,report_sha256=e.sha(e.HERE/'REPORT.md')))
    files=[p for p in e.HERE.rglob('*') if p.is_file() and p.name!='MANIFEST.json' and '__pycache__' not in p.parts]
    e.save(e.HERE/'MANIFEST.json',dict(status='complete',files={str(p.relative_to(e.HERE)):dict(sha256=e.sha(p),bytes=p.stat().st_size) for p in sorted(files)}))
    print('H3 FINALIZATION PASS',len(files),'local files;',len(report.split()),'report words',flush=True)
if __name__=='__main__':main()
