"""Finalize local pooled artifacts; parent owns common calibration and policies."""
from pathlib import Path
import sys,json,importlib.metadata
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import numpy as np,pandas as pd
from research_v4.architecture_2023_2025.pooled import experiment as e
from research_v4.architecture_2023_2025.pooled.resume_verified import audit
def md(frame):
    def fmt(v):return f'{v:.6f}' if isinstance(v,float) else str(v)
    return '\n'.join(['| '+' | '.join(map(str,frame.columns))+' |','| '+' | '.join(['---']*len(frame.columns))+' |']+['| '+' | '.join(fmt(v) for v in row)+' |' for row in frame.itertuples(index=False,name=None)])
def main():
    guarded=audit();ind=json.loads((e.HERE/'independent_review.json').read_text());assert ind['status']=='PASS' and ind['completed_model_pairs_reviewed']==6
    assert ind['reviewed_code_sha256']==e.sha(e.__file__)
    raw=pd.read_csv(e.HERE/'raw_predictions.csv.gz',parse_dates=['date','label_available_date'])
    assert raw.fold_test_year.isin([2023,2024,2025]).all()
    assert len(raw)==43920
    keys=['date','corridor','split','fold_test_year']
    reference=None
    for cid,q in raw.groupby('config_id'):
        assert not q.duplicated(keys).any()
        q=q.sort_values(keys).reset_index(drop=True)
        values=q[keys+e.KEEP[2:]]
        if reference is None:reference=values
        else:pd.testing.assert_frame_equal(reference,values,check_exact=True)
    scores=pd.read_csv(e.HERE/'raw_score_diagnostic.csv')
    scores=scores[scores.split.eq('test')].copy()
    scores['Архитектура']=scores.config_id.str.extract('pooled_(tabm|hgb)_')[0]
    scores['Признаки']=scores.config_id.str.extract('(base15|full33)')[0]
    summary=scores.pivot(index=['year','Признаки'],columns='Архитектура',values='raw_brier').reset_index()
    summary.columns=['Год','Признаки','HGB, сырой Brier','TabM, сырой Brier']
    receipts=[json.loads(p.read_text()) for p in e.OUT.glob('*_receipt.json')]
    runtime_nn=sum(r['neural']['fit_seconds'] for r in receipts);runtime_hgb=sum(r['hgb_seconds'] for r in receipts)
    report=f'''# Pooled TabM / HGB: сопоставимые конфигурации, 2023–2025

Реально обучены **шесть TabM и шесть HGB**: три календарных периода, два набора признаков, обе модели объединяют пять валютных коридоров. Подготовлены 43 920 строк сырых вероятностей для калибровки, восстановления истории и оценки. Независимая проверка всех моделей — **PASS**.

Сравнение сохраняет одинаковые внешние входы. BASE15 — базовые признаки расширенной панели V3, включая волатильность и биржевой базис CNY/RUB. FULL33 добавляет OXR с 2010 года, Halyk и Treasury. Список полей находится в `protocol.json`. Длинное окно не означает полного десятилетнего покрытия новых источников: ранние отсутствующие значения остаются пропусками.

Для каждой пары моделей медианная импутация и квантильное преобразование обучаются только на её обучающем окне. Добавляются индикаторы пропусков всех числовых признаков. HGB получает буквально тот же массив float32, что и TabM, плюс one-hot пяти коридоров. Внутри TabM коридор тоже преобразуется в тот же one-hot без обучаемых параметров. Числовые PeriodicEmbeddings остаются частью архитектуры TabM.

TabM использует seed 20261105, при итоговом переобучении — 20261106; два блока по 128 нейронов, 16 участников ансамбля и числовые эмбеддинги размерности 16. Число эпох выбирается на последних 63 датах самого обучающего окна, после чего модель переобучается на всём разрешённом обучении. HGB зафиксирован на 120 деревьях глубины 2, шаге 0,05, минимуме 40 строк в листе и L2=2; автоматическая остановка отключена. Это сравнение фиксированных рецептов, а не универсальный рейтинг архитектур или одинаковый бюджет поиска гиперпараметров.

Для запусков 1 января 2023, 2024 и 2025 года используются 120 месяцев обучения и предшествующие запуску 12 месяцев калибровки. Метка NOW сохраняет исходный смысл: текущая рублёвая стоимость не выше минимума следующих пяти наблюдений CBR. Реальное пятое следующее наблюдение должно появиться до следующей границы. Проверена также внутренняя граница выбора эпох. В незрелом хвосте истории замаскированы target и все три будущих показателя эффекта; сами вероятности истории сохранены.

В каждой архитектуре и наборе признаков оцениваются одни и те же 1210, 1215 и 1210 строк соответственно, всего 3635 пар дата–коридор. Пять коридоров одной даты зависимы, поэтому доверительные интервалы следует строить общими временными блоками. В этой ветке нет обучения, оценки или выбора по 2026 году. Рецепты пришли из предыдущего исследования, поэтому 2023–2025 также не объявляются нетронутой проверочной выборкой.

Ниже только **сырые вероятности до общей калибровки**. Это диагностическая таблица, а не окончательная оценка архитектур: основной эксперимент одинаково калибрует вероятности по коридорам и применяет общие политики.

{md(summary)}

Результаты неоднородны. В частности, расширенные признаки резко ухудшили некалиброванный TabM в 2023 году; в 2024 году TabM с FULL33 оказался лучше HGB с тем же набором, а в 2025 году обе модели получили лучшие сырые оценки с FULL33. Обобщать этот результат в утверждение о превосходстве нейросетей нельзя.

Обучение TabM, включая внутренний выбор эпох и итоговое переобучение, заняло суммарно {runtime_nn:.2f} с CPU; HGB — {runtime_hgb:.2f} с. Подготовка данных и независимые проверки сюда не входят. Подобранные внутри обучения числа эпох: BASE15 — 6/4/20, FULL33 — 15/4/18.

Независимый проверяющий заново обучил препроцессоры только на исходных обучающих строках, получил побитово совпадающие статистики, преобразования, индикаторы пропусков и кодирование коридоров. Все сохранённые вероятности восстановлены из 12 checkpoints; максимальная разность после CSV-представления — 1,11×10⁻¹⁶. Проверены хеши исходных данных, даты созревания, общность строк и маскирование незрелой истории. Подробности: `independent_review.json` и `independent_review.md`.

Для основного оценщика: `raw_predictions.csv.gz`, ключи `config_id, cutoff, split, date, corridor`; поле `raw_probability`. Идентификаторы: `arch_pooled_tabm_base15`, `arch_pooled_tabm_full33`, `arch_pooled_hgb_base15`, `arch_pooled_hgb_full33`. Split принимает `validation`, `history`, `test`. Итоговые калиброванные оценки и политики будут представлены на уровне родительского эксперимента.

Для проверенного повторного запуска используйте `resume_verified.py`: он сначала сверяет хеши входов, кода, препроцессоров и весов, seed, признаки и обучающие строки, затем вызывает неизменный движок. Режим `--check-only` выполняет только проверки. Исходные и прежние sealed-артефакты не изменены. `MANIFEST.json` фиксирует файлы этой ветки.
'''
    (e.HERE/'REPORT.md').write_text(report)
    (e.HERE/'README.md').write_text('# Pooled architecture factorial\n\n[Отчёт](REPORT.md) · [Протокол](PROTOCOL.md) · [Проверки](final_verification.json)\n\nОбщий raw export: `raw_predictions.csv.gz`. Калибровку и политики выполняет родительский эксперимент. Безопасный повторный запуск: `resume_verified.py`; только сверка: `resume_verified.py --check-only`.\n')
    e.save(e.HERE/'environment.json',dict(python=sys.version,packages={x:importlib.metadata.version(x) for x in ['torch','tabm','rtdl-num-embeddings','numpy','pandas','scikit-learn']},torch_threads=2))
    e.save(e.HERE/'final_verification.json',dict(status='PASS',neural_fits=6,temporary_inner_neural_fits=6,hgb_fits=6,raw_rows=len(raw),model_pairs=6,guarded_pairs=guarded,independent_check_blocks=len(ind['checks']),independent_review_sha256=e.sha(e.HERE/'independent_review.json'),raw_predictions_sha256=e.sha(e.HERE/'raw_predictions.csv.gz'),protocol_sha256=e.sha(e.HERE/'protocol.json'),experiment_code_sha256=e.sha(e.__file__),report_sha256=e.sha(e.HERE/'REPORT.md'),years=[2023,2024,2025],scope='Raw matched fixed configurations; identical parent-owned per-corridor calibration/policy pending'))
    files=[p for p in e.HERE.rglob('*') if p.is_file() and p.name!='MANIFEST.json' and '__pycache__' not in p.parts]
    e.save(e.HERE/'MANIFEST.json',dict(status='complete',files={str(p.relative_to(e.HERE)):dict(sha256=e.sha(p),bytes=p.stat().st_size) for p in sorted(files)}))
    print('POOLED FINAL PASS',len(files),'files;',len(raw),'raw rows;',len(report.split()),'report words',flush=True)
if __name__=='__main__':main()
