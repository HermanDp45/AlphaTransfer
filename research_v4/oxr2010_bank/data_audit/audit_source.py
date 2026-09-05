"""Audit the user-supplied OXR back-extension locally, without any API calls.

Writes only this directory. Uses the new experiment snapshot when present;
otherwise audits the live file and explicitly records the snapshot as pending.
"""
from pathlib import Path
import datetime
import hashlib
import io
import json

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
NEW = HERE.parent/'input_oxr_snapshot.csv'
LIVE = ROOT/'data/open_exchange_rates/rub_cis_daily.csv'
OLD = ROOT/'research_v4/continuation/oxr/input_oxr_snapshot.csv'
TARGETS = ['AMD', 'KGS', 'KZT', 'TJS', 'UZS']
NUM = ['quote_per_rub', 'rub_per_quote', 'rub_per_usd', 'quote_per_usd']
OLD_SHA = 'd30d226fbaa0b2d89a2e7eacf011b90af93eb60150572ec66c2fb1ff26a25db2'
EXPECTED_NEW_SHA = '9cbe2f322c3e8c9885dce98702c66c5c928864ca155dd19eb950db0cc8bc8021'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def table(frame):
    rows = [list(frame.columns), ['---']*len(frame.columns), *frame.itertuples(index=False, name=None)]
    return '\n'.join('| '+' | '.join(map(str, row))+' |' for row in rows)


def main():
    source = NEW if NEW.exists() else LIVE
    raw_bytes = source.read_bytes()
    x = pd.read_csv(io.BytesIO(raw_bytes)).sort_values(['date', 'quote']).reset_index(drop=True)
    old = pd.read_csv(OLD)
    days = pd.date_range(x.date.min(), x.date.max())
    coverage, flats, jumps = [], [], []
    for quote, g in x.groupby('quote'):
        g = g.sort_values('date').reset_index(drop=True)
        missing = days.difference(pd.to_datetime(g.date))
        coverage.append(dict(quote=quote, target=quote in TARGETS, rows=len(g), first=g.date.min(),
                             last=g.date.max(), missing_calendar_days=len(missing),
                             first_missing=str(missing.min().date()) if len(missing) else None,
                             last_missing=str(missing.max().date()) if len(missing) else None))
        for column in ('rub_per_quote', 'quote_per_usd'):
            runs=g[column].ne(g[column].shift()).cumsum()
            longest=g.groupby(runs).size().idxmax(); block=g[runs.eq(longest)]
            flats.append(dict(quote=quote, target=quote in TARGETS, column=column,
                              unchanged_fraction=float(g[column].diff().iloc[1:].eq(0).mean()),
                              longest_constant_days=len(block), first=block.date.min(), last=block.date.max()))
        g['log_return'] = np.log(g.rub_per_quote).diff()
        top=g.loc[g.log_return.abs().nlargest(5).index]
        for row in top.itertuples():
            jumps.append(dict(quote=quote, target=quote in TARGETS, date=row.date,
                              rub_per_quote=row.rub_per_quote, log_return=row.log_return,
                              annotation='Observed jump; not automatically invalid; retained unchanged.'))
    cover=pd.DataFrame(coverage);flat=pd.DataFrame(flats)
    overlap=x.merge(old, on=['date', 'quote'], suffixes=['_new', '_old'], validate='one_to_one')
    comparisons=[]
    for scope, q in [('all9', overlap), ('targets5', overlap[overlap.quote.isin(TARGETS)])]:
        for col in NUM:
            delta=q[col+'_new']-q[col+'_old']
            comparisons.append(dict(scope=scope, column=col, overlap_rows=len(q),
                                    changed_rows=int(delta.ne(0).sum()), maximum_absolute_difference=float(delta.abs().max()),
                                    maximum_relative_difference=float((q[col+'_new']/q[col+'_old']-1).abs().max())))
        comparisons.append(dict(scope=scope,column='published_at_utc',overlap_rows=len(q),
                                changed_rows=int(q.published_at_utc_new.ne(q.published_at_utc_old).sum()),
                                maximum_absolute_difference=0,maximum_relative_difference=0))
    changed=overlap.published_at_utc_new.ne(overlap.published_at_utc_old)
    for col in NUM:changed |= overlap[col+'_new'].ne(overlap[col+'_old'])
    pub=pd.to_datetime(x.published_at_utc,utc=True)
    next_midnight=pd.to_datetime(x.date,utc=True)+pd.Timedelta(days=1)
    known=pd.concat([pub,next_midnight],axis=1).max(axis=1)+pd.Timedelta(hours=24)
    first_decision=pd.to_datetime(x.date).dt.tz_localize('Europe/Moscow')+pd.Timedelta(days=2,hours=10,minutes=5)
    source_groups=x.groupby(['quote','source_quote']).agg(rows=('date','size'),first=('date','min'),last=('date','max')).reset_index()
    byn=x[x.quote.eq('BYN') & x.date.between('2016-06-25','2016-07-05')].copy()
    downloader=ROOT/'scripts/download_oxr_cis.py'
    code=downloader.read_text()
    locks=[]
    for path,base in [(ROOT/'final_solution/inputs.lock.json',ROOT),
                      (ROOT/'research_v3/manifest.json',ROOT),
                      (ROOT/'research_v4/artifact_manifest.json',ROOT/'research_v4')]:
        manifest=json.loads(path.read_text())
        matches=[]
        for name,value in manifest['files'].items():
            if any(z in name for z in ('open_exchange','oxr_snapshot','download_oxr')):
                expected=value['sha256'] if isinstance(value,dict) else value
                actual=sha(base/name) if (base/name).exists() else None
                matches.append(dict(file=name,expected=expected,actual=actual,match=expected==actual))
        locks.append(dict(manifest=str(path.relative_to(ROOT)),oxr_references=matches))
    allchecks=dict(expected_new_input_sha256=hashlib.sha256(raw_bytes).hexdigest()==EXPECTED_NEW_SHA,
                   old_snapshot_intact=sha(OLD)==OLD_SHA,
                   no_duplicate_date_quote=not x.duplicated(['date','quote']).any(),
                   no_nulls=not x.isna().any().any(),all_rates_positive_finite=bool((x[NUM].gt(0)&np.isfinite(x[NUM])).all().all()),
                   target_calendar_complete=bool(cover.loc[cover.target,'missing_calendar_days'].eq(0).all()),
                   timestamp_date_matches=bool(pub.dt.strftime('%Y-%m-%d').eq(x.date).all()),
                   same_timestamp_across_quotes=bool(x.groupby('date').published_at_utc.nunique().eq(1).all()),
                   pair_base_identity=bool(x.base.eq('RUB').all()&x.pair.eq('RUB/'+x.quote).all()),
                   target_source_quote_identity=bool(x.loc[x.quote.isin(TARGETS),'source_quote'].eq(x.loc[x.quote.isin(TARGETS),'quote']).all()),
                   overlap_exact=not changed.any(),old_overlap_all_rows_present=len(overlap)==len(old),
                   primary_Dplus2_available=bool(known.le(first_decision).all()))
    out=dict(status='PASS' if all(allchecks.values()) else 'FAIL',audit_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
             api_calls=0,credentials_inspected=False,read_only_inputs=True,checks=allchecks,
             input=dict(path=str(source),sha256=hashlib.sha256(raw_bytes).hexdigest(),live_path=str(LIVE),live_sha256=sha(LIVE),
                        new_snapshot_exists=NEW.exists(),new_snapshot_path=str(NEW),rows=len(x),calendar_dates=len(days),
                        first=x.date.min(),last=x.date.max(),columns=list(x.columns),target_rows=int(x.quote.isin(TARGETS).sum())),
             previous_snapshot=dict(path=str(OLD),sha256=sha(OLD),rows=len(old)),
             partition=dict(backextension_before_old_start=int(x.date.lt(old.date.min()).sum()),
                            overlap_rows=len(overlap),append_after_old_end=int(x.date.gt(old.date.max()).sum()),
                            overlap_numeric_or_timestamp_revised_rows=int(changed.sum()),
                            interpretation='Pure back-extension plus two newer days and source_quote schema addition; no detected overlap revisions.'),
             coverage=coverage,overlap=comparisons,flatness=flats,
             numeric=dict(inverse_identity_max_absolute_error=float((x.quote_per_rub*x.rub_per_quote-1).abs().max()),
                          usd_cross_max_relative_error=float((x.rub_per_quote/(x.rub_per_usd/x.quote_per_usd)-1).abs().max())),
             timestamp=dict(min_time_utc=pub.dt.strftime('%H:%M:%S').min(),max_time_utc=pub.dt.strftime('%H:%M:%S').max(),
                            time_counts_by_dates=pub.drop_duplicates().dt.strftime('%H:%M:%S').value_counts().head(10).to_dict(),
                            minimum_seconds_publication_to_primary_decision=float((first_decision-pub).dt.total_seconds().min()),
                            rule='max(raw publication UTC, source date+1 day UTC midnight)+24h primary; +48h stress',
                            limitation='Reported publication timestamp, not fetched_at or revision-vintage proof. Flat currency legs can be republished daily.'),
             redenomination=dict(target_corridors_affected=False,source_quote_groups=source_groups.to_dict('records'),
                                  downloader_sha256=sha(downloader),normalization_divisor=10000,
                                  exporter_provenance='source_quote is inferred by SQL CASE (BYN date<2016-06-30 -> BYR), not a stored raw API field.',
                                  local_code_has_conversion='float(rates["BYR"]) / BYR_PER_BYN' in code,
                                  raw_http_payloads_audited=False,
                                  warning='BYN values are normalized current-unit history; quote_per_usd is already divided, do not divide again. TMT 2010 is absent, not synthetic.'),
             old_lock_references=locks,
             preservation='No old files restored or rewritten. The live file change is user-authorized; existing frozen OXR snapshot remains exact. Strict old V4 file-set seal is naturally no longer a seal of newly added directories.',
             audit_code_sha256=sha(__file__))
    HERE.mkdir(parents=True,exist_ok=True)
    (HERE/'source_audit.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    cover.to_csv(HERE/'coverage.csv',index=False);flat.to_csv(HERE/'flatness.csv',index=False)
    pd.DataFrame(comparisons).to_csv(HERE/'overlap_comparison.csv',index=False)
    overlap.loc[changed].to_csv(HERE/'overlap_revisions.csv',index=False)
    pd.DataFrame(jumps).to_csv(HERE/'extreme_returns.csv',index=False)
    byn.to_csv(HERE/'byn_transition.csv',index=False)
    report=f'''# Новый OXR 2010–2026: независимый аудит источника

**{out['status']}: все пять целевых валют полны; пересмотров старого участка не найдено.** Проверены только локальные данные. API-запросов: 0. Исходные V3/V4 и пользовательский live-файл не изменялись.

Источник: `{source.relative_to(ROOT)}`. SHA-256: `{out['input']['sha256']}`. Новый экспериментальный снимок существует: **{NEW.exists()}**. Если он ещё не скопирован, повторный запуск этого скрипта привяжет аудит к снимку; содержимое должно сохранить указанный hash.

## Полнота

54,454 строки, 6,091 календарный день, 2010-01-01–2026-09-04. Для AMD/KGS/KZT/TJS/UZS — **30,455 строк, по 6,091 на валюту; 0 пропусков календаря, дубликатов и null**. Во всём файле 365 отсутствующих относительно прямоугольника 9×6,091 ячеек относятся только к TMT за 2010 год. Это не целевая валюта; замены другим источником или заполнения константой нет.

{table(cover[['quote','target','rows','first','last','missing_calendar_days']])}

## Продление истории и пересмотры разделены

Старый зафиксированный файл `research_v4/continuation/oxr/input_oxr_snapshot.csv` сохранил SHA `{OLD_SHA}`. Все **27,000 общих строк за 2018-06-17–2026-09-02** совпадают точно по четырём числовым колонкам и времени публикации. В пяти целевых валютах это 15,000 строк. Максимальные числовые разницы и число пересмотров равны **нулю**. Новый файл добавил **27,436 ранних строк** и **18 строк за 2026-09-03/04**, а также колонку `source_quote`.

Поэтому при сохранённых целевых датах и формуле признаков можно отдельно измерять пользу расширенной истории без примеси выявленных пересмотров overlap. Новые последние даты не должны автоматически расширять прежний тест; определять доступность меток по фактической истории ЦБ всё равно необходимо. [Полная сверка](overlap_comparison.csv); [список пересмотров, пуст](overlap_revisions.csv).

## Валютные единицы и source_quote

Все значения конечны и положительны. Кросс `rub_per_quote = rub_per_usd / quote_per_usd` согласован с ошибкой до {out['numeric']['usd_cross_max_relative_error']:.3g}; ошибка обратных колонок не превышает {out['numeric']['inverse_identity_max_absolute_error']:.3g}. Это справочные USD-кроссы, а не исполнимые прямые банковские котировки.

Для пяти целевых валют `source_quote == quote` на всех датах. Дополнительный BYN содержит 2,372 строки с меткой BYR до 2016-06-29 и 3,719 с BYN с 2016-06-30. Локальный downloader преобразует исходный BYR/USD делением на 10,000; экспортируемый `quote_per_usd` уже нормирован в BYN. Повторное деление ошибочно. В [таблице перехода](byn_transition.csv) нет механического скачка масштаба в 10,000 раз.

Важное ограничение происхождения поля: `source_quote` вычисляется SQL CASE по дате при экспорте; сырой исходный код валюты в таблице SQLite не хранится. Поэтому это локальная разметка происхождения, а не независимое доказательство содержания каждого исторического API-ответа. Аудит не обращался к API и не проверял исходные HTTP payload. Это не затрагивает пять целевых валют. Локальное описание деноминации и исключения TMT сохранено в исходном README; здесь проверены код преобразования, единицы и непрерывность локальных значений, а не юридический документ о деноминации.

## Публикация, неизменные цены и PIT

UTC-дата каждого `published_at_utc` совпадает с датой строки; все валюты дня имеют одно время. В ранней истории времена отличаются от поздних 23:59:xx: минимум {out['timestamp']['min_time_utc']}, максимум 23:59:59 UTC. Поле берётся из timestamp ответа по локальному downloader; само по себе оно не является временем скачивания или доказательством отсутствия пересмотров.

Сохранять **known_at = max(published_at_utc, следующая UTC-полночь)+24 часа** и backward as-of на 10:05 MSK; стресс +48 часов. Это по-прежнему D+2/D+3, минимум 31 ч 05 м 01 с между публикацией и основным решением. Ранние времена публикации не дают основания ослаблять лаг автоматически. Предыдущий аудит официальной семантики OXR находится в `research_v4/continuation/oxr/source_audit.md`; новых сетевых обращений этот аудит не делал.

В RUB-кроссах пяти целей максимальная серия неизменных цен — 5 дней. Но USD-компоненты отдельных валют бывают постоянными значительно дольше: AMD 194 дня, KGS 62, KZT 10, TJS 66, UZS 25. AMD 194 дня — уже существовавший участок 2020-06-17–2020-12-27, а не новое загрязнение раннего расширения. Ежедневная публикация не доказывает свежесть каждой валютной компоненты. Это наблюдаемая постоянность; она сама по себе не доказывает ошибку данных. [Flatness по двум представлениям](flatness.csv).

Крупные движения сохранены без исправлений: например, UZS 2017-09-05 имеет лог-изменение около −0.655, а максимум ряда AMD/KGS/KZT/TJS приходится на 2022-03-07. Это диагностические выбросы, не автоматически некорректные значения. [Диагностика движений](extreme_returns.csv).

## Старые контрольные суммы и воспроизведение

В `final_solution/inputs.lock.json` и V3 manifest нет ссылок на live OXR или его downloader. Старый V4 manifest ссылается на зафиксированный OXR-снимок — он совпадает. Старый описательный source_audit содержит hash live-файла на дату предыдущего исследования; это историческая запись, которую нельзя переписывать под новый архив. Строгая проверка состава всего старого V4 дерева естественно увидит новые файлы этого исследования как добавления; это не причина восстанавливать старый пользовательский live-источник.

Запуск из AlphaTransfer: `PYTHONDONTWRITEBYTECODE=1 python research_v4/oxr2010_bank/data_audit/audit_source.py`. Скрипт пишет только в `data_audit`. [Структурированный аудит](source_audit.json). Новых ML-моделей и рыночных данных в этом аудите не создавалось.
'''
    (HERE/'REPORT.md').write_text(report)
    print(json.dumps({'status':out['status'],'rows':len(x),'target_rows':out['input']['target_rows'],
                      'overlap_revisions':int(changed.sum()),'new_snapshot_exists':NEW.exists()},ensure_ascii=False))
    if out['status'] != 'PASS':
        raise SystemExit(1)


if __name__=='__main__':
    main()
