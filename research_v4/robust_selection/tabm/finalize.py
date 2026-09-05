"""Finalize raw TabM deliverables for the parent-owned common evaluation."""
from pathlib import Path
import sys,json,importlib.metadata
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import pandas as pd
from research_v4.robust_selection.tabm import experiment as e
def table(df):
    rows=['| '+' | '.join(map(str,df.columns))+' |','| '+' | '.join(['---']*len(df.columns))+' |']
    for row in df.itertuples(index=False,name=None):rows.append('| '+' | '.join(f'{v:.6f}' if isinstance(v,float) else str(v) for v in row)+' |')
    return '\n'.join(rows)
def main():
    v=json.loads((e.HERE/'verification.json').read_text());assert v['status']=='PASS'
    completion=json.loads((e.HERE/'completion.json').read_text());receipts=json.loads((e.HERE/'receipts.json').read_text())
    score=pd.read_csv(e.HERE/'raw_score_diagnostic.csv');score=score[score.split.eq('test')]
    diagnostic=score.pivot(index=['year','train_horizon'],columns='config_id',values='raw_brier').reset_index()
    diagnostic.columns=['Год','Горизонт','KZT-only, сырой Brier','Пять коридоров, сырой Brier']
    report=f'''# TabM: отдельные H3/H5 и ежегодное скользящее обучение

Подготовлены **12 моделей по комбинациям горизонта, года и состава коридоров**. Реально выполнены семь новых нейросетевых обучений и семь внутренних обучений для выбора числа эпох. Пять H5-checkpoint переиспользованы после строгой проверки совпадения данных. Экспортировано **25 392 строки** сырых вероятностей для калибровки, истории и оценки, а также **2268 строк** начального контекста ранжирования. Все проверки — **PASS**.

Горизонты H3 и H5 означают следующие три или пять фактических наблюдений CBR, а не календарные дни. Для каждого горизонта заново созданы `target`, `forward_bps`, `symmetric_bps`, `regret_bps` и `label_available_date`. NOW по-прежнему означает, что текущая рублёвая стоимость не выше минимума будущих наблюдений. H3 не получен обрезкой сигнала модели H5: все шесть H3-моделей обучены отдельно на собственных метках. Проверка независимо пересчитала оба вида меток, все будущие показатели и даты их созревания.

Состав моделей фиксирован: FULL33, TabM с числовыми PeriodicEmbeddings, seed 20261105, итоговый seed 20261106. Используется неизменный обучающий класс из предыдущего эксперимента: два блока по128 нейронов,16 участников ансамбля, train-only медианная импутация и квантильное преобразование, индикаторы пропусков. Отдельные модели обучены только на KZT и на пяти коридорах вместе. Последние предназначены для оценки всех пяти коридоров; KZT-модель не распространяется на остальные валюты.

Для дат запуска 1 января2024,2025 и2026 года обучение занимает120 месяцев **до** предшествующего года калибровки. Граница зрелости меток проверяется по фактическому третьему/пятому следующему наблюдению. Сам `temporal_split` получает соответствующий горизонт3/5. Внутренняя проверка для выбора числа эпох также использует новую фактическую дату созревания. Ни seed, ни архитектура, ни состав признаков не выбирались по результатам этой серии;2026 год ранее просматривался и не считается нетронутой контрольной выборкой.

Повторно использованы KZT H5 для2024–2026 и pooled H5 для2024–2025. Проверка охватывает все обучающие строки, признаки, метки, будущие показатели и maturity, а также веса, seed и независимо переобученный препроцессор. В новую папку скопированы собственные checkpoints; старые артефакты не изменены. Pooled H5 для2026 обучен заново, поскольку точного прежнего seed2-checkpoint не было.

Перед калибровочным годом сохранены вероятности за последние **63 PANEL-даты**, включая исключённый при обучении хвост из H наблюдений. Используется тот же checkpoint, все исходы полностью замаскированы. Этот контекст нужен только для последовательного ранжирования вероятностей, а не для оценки качества: большая его часть находится внутри обучающей выборки. Канонический файл — `warmup.csv.gz`. Первоначальный контекст без исключённого хвоста сохранён под именами `warmup_purged_train_*` как отклонённый вариант; в оценщике его использовать нельзя.

Ниже диагностические Brier **до** общей калибровки. Значения H3 и H5 относятся к разным событиям и не доказывают преимущество одного горизонта. Окончательное сравнение с настоящим V3 strict0.5 и политики с80–90% недель выполняет родительский эксперимент на одинаковых правилах.

{table(diagnostic)}

В H3 допустимых тестовых дат на две больше в каждом году: для KZT2024/2025/2026 —245/244/158; в H5 —243/242/156. При прямом межгоризонтном сравнении нужны общие даты или явное указание разных панелей. Внутри одного горизонта KZT имеет тот же набор дат и исходов, что KZT-часть pooled-выборки.

`verification.json` подтверждает независимую семантику обоих горизонтов, отсутствие дубликатов, фактическую зрелость на границах и48 повторных расчётов из checkpoints: validation, history, test и warmup для12 моделей. Незрелые исходы обычной истории замаскированы. Канонический warmup содержит ровно63 даты и последний доступный день перед калибровкой, включая весь исключённый из обучения хвост.

Основные входы оценщика: `raw_predictions.csv.gz` и `warmup.csv.gz`. Поля: `date`, `corridor`, все KEEP-метрики, `raw_probability`, `config_id` (`tabm_kzt`/`tabm_pooled`), `train_horizon`, `fold_test_year`, `cutoff`, `split`. Основной split — `validation`, `history`, `test`; дополнительный — `warmup`. Уникальность проверена также без года по сочетанию `train_horizon/config_id/split/date/corridor`.

Воспроизводимость: `protocol.json`, `warmup_protocol.json`, `receipts.json`, собственные `checkpoints/`, `checkpoint_replay.csv`, `target_semantic_checks.csv`, `verification.json` и `MANIFEST.json`. Запуск: experiment.py → warmup.py → verify.py → finalize.py. Для переиспользования H5 сохраняются ссылки и хеши исходных receipts. Весь набор признаков и входная панель заморожены; новые данные из сети не загружались.
'''
    import re
    report=re.sub(r'(?<=[А-Яа-яёЁ])(?=\d)|(?<=\d)(?=[А-Яа-яёЁ])',' ',report)
    report=report.replace('по128','по 128').replace(',16',', 16').replace('2024,2025','2024, 2025').replace('strict0.5','strict 0.5').replace('с80','с 80').replace('для12','для 12')
    (e.HERE/'REPORT.md').write_text(report)
    (e.HERE/'README.md').write_text('# TabM: ежегодные H3/H5\n\n[Отчёт](REPORT.md) · [Проверки](final_verification.json)\n\nКанонические входы родительского оценщика: `raw_predictions.csv.gz` и `warmup.csv.gz`. Файлы `warmup_purged_train_*` — отклонённая первоначальная версия контекста; не использовать.\n')
    e.save(e.HERE/'environment.json',dict(python=sys.version,packages={p:importlib.metadata.version(p) for p in ['torch','tabm','rtdl-num-embeddings','numpy','pandas','scikit-learn']},cpu_threads=2))
    e.save(e.HERE/'final_verification.json',dict(status='PASS',model_cells=12,new_neural_fits=completion['new_neural_fits'],new_inner_fits=completion['new_neural_fits'],reused_h5_checkpoints=completion['reused_h5_checkpoints'],raw_rows=completion['raw_prediction_rows'],warmup_rows=2268,raw_predictions_sha256=e.sha(e.HERE/'raw_predictions.csv.gz'),warmup_sha256=e.sha(e.HERE/'warmup.csv.gz'),verification_sha256=e.sha(e.HERE/'verification.json'),report_sha256=e.sha(e.HERE/'REPORT.md'),parent_scope='Common calibration, V3 strict0.5 and80–90%coverage policies are parent-owned'))
    files=[p for p in e.HERE.rglob('*') if p.is_file() and p.name!='MANIFEST.json' and '__pycache__' not in p.parts]
    e.save(e.HERE/'MANIFEST.json',dict(status='complete',files={str(p.relative_to(e.HERE)):dict(sha256=e.sha(p),bytes=p.stat().st_size) for p in sorted(files)}))
    print('FINAL PASS',len(files),'files;',len(report.split()),'report words',flush=True)
if __name__=='__main__':main()
