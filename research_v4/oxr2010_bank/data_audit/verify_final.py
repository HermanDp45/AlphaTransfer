"""Independent read-only canonical-attribution and simultaneous-band audit.

No fitting, network requests, or writes outside this directory.
"""
from pathlib import Path
import hashlib
import json
import os
import sys
import warnings

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from research_v4.oxr2010_bank.long_models import experiment as engine
from research_v3.external_data.benchmark import augment_panel


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    checks = []
    evidence = {}
    def check(name, passed, **details):
        checks.append(dict(check=name, passed=bool(passed), **details))
        print(name, 'PASS' if passed else 'FAIL', flush=True)

    folder = engine.HERE / 'canonical'
    protocol = json.loads((folder / 'protocol.json').read_text())
    check('canonical_code_and_engine_protocol_hashes',
          sha(engine.HERE / 'canonical_control.py') == protocol['code_sha256']
          and sha(engine.__file__) == protocol['engine_sha256'])
    views, _ = engine.build_views()
    features = engine.oxr.RET + engine.oxr.BASIS + engine.oxr.COVER
    warmup = []
    for delay in (24, 48):
        for lag in (1, 2):
            short = views['2018-06-17', delay, lag]
            long = views['2010-01-01', delay, lag]
            common = short.date.ge(protocol['common_start'])
            before = long.loc[~common].copy()
            for corridor, q in short.groupby('corridor'):
                first = q.loc[q.date.ge(protocol['common_start'])].iloc[0]
                available_past = q[q.date.le(first.date) & q.oxr_log_basis.notna()]
                native_days = (first.observed_date - pd.Timestamp('2018-06-17')).days + 1
                warmup.append(dict(delay=delay, bank_lag=lag, corridor=corridor,
                                   first_common_date=str(first.date.date()),
                                   observed_source_date=str(first.observed_date.date()),
                                   source_calendar_observations=int(native_days),
                                   prior_inclusive_CBR_basis_observations=len(available_past)))
            check(f'canonical_common_missingness_D{delay}_B{lag}',
                  short.loc[common, features].isna().equals(long.loc[common, features].isna()))
            long.loc[common, features] = short.loc[common, features]
            check(f'canonical_only_common_features_changed_D{delay}_B{lag}',
                  long.loc[~common].equals(before)
                  and long.loc[common, features].equals(short.loc[common, features]))
    check('canonical_boundary_exceeds_both_full_warmups',
          min(x['source_calendar_observations'] for x in warmup) >= 21
          and min(x['prior_inclusive_CBR_basis_observations'] for x in warmup) >= 20,
          minimum_source_observations=min(x['source_calendar_observations'] for x in warmup),
          minimum_CBR_basis_observations=min(x['prior_inclusive_CBR_basis_observations'] for x in warmup))
    evidence['warmup'] = warmup
    views = {k: augment_panel(v) for k, v in views.items()}
    receipts = list((folder / 'output').glob('*.json'))
    errors = []
    for path in receipts:
        receipt = json.loads(path.read_text())
        spec = receipt['spec']; cutoff = pd.Timestamp(receipt['cutoff'])
        train, val, test = engine.old.temporal_split(
            views[spec['since'], 24, 1], 5, cutoff, pd.Timestamp(cutoff.year + 1, 1, 1),
            engine.old.Spec(spec['name'], months=spec['months'], extended=True))
        fields = ['date', 'corridor', *receipt['features']]
        if not (engine.fp(train[fields]) == receipt['train_feature_fingerprint']
                and sha(path.with_suffix('.pkl')) == receipt['checkpoint_sha256']
                and sha(path.with_suffix('.csv.gz')) == receipt['predictions_sha256']
                and sha(engine.SNAPSHOT) == receipt['source_sha256']
                and train.label_available_date.max() < cutoff - pd.DateOffset(years=1)
                and val.label_available_date.max() < cutoff):
            errors.append(path.name)
    check('all_35_canonical_training_fingerprints_hashes_and_label_purges',
          len(receipts) == 35 and not errors, checkpoints=len(receipts), errors=errors)
    canonical = pd.read_csv(folder / 'all_predictions.csv.gz')
    original = pd.concat([pd.read_csv(p) for p in
                          (engine.HERE / 'all_predictions.csv.gz', engine.HERE / 'treasury/all_predictions.csv.gz')])
    canonical.config_id = canonical.config_id.str.removeprefix('canonical_')
    keys = ['config_id', 'cutoff', 'mode', 'date', 'corridor']
    matched = canonical.merge(original, on=keys, suffixes=['_canonical', '_original'], validate='one_to_one')
    errors = {c: float((matched[c + '_canonical'].astype(float) - matched[c + '_original'].astype(float)).abs().max())
              for c in ['raw_probability', 'probability', 'target', 'forward_bps', 'candidate_signal', 'signal']}
    check('canonical_retraining_all_saved_views_exact',
          len(matched) == len(canonical) == 50250 and max(errors.values()) == 0,
          matched_rows=len(matched), maximum_differences=errors,
          note='Repeated model/view row evaluations, not independent observations.')

    # Direct sampled-index block sums independently reconstruct the root weight-matrix bootstrap.
    allp = pd.concat([pd.read_csv(p, parse_dates=['date']) for p in
                     (engine.HERE / 'all_predictions.csv.gz', engine.HERE / 'treasury/all_predictions.csv.gz',
                      engine.HERE / 'bank_controls/all_predictions.csv.gz')])
    allp = allp[allp['mode'].eq('normal') & allp.corridor.eq('KZT')]
    check('simultaneous_family_common_targets',
          allp.groupby(['cutoff', 'date']).target.nunique().eq(1).all()
          and allp.config_id.nunique() == 27)
    jan = allp[allp.cutoff.eq('2026-01-01')]
    march = allp[allp.cutoff.eq('2026-03-01')]
    common = set(jan.date) & set(march.date)
    tracks = {'development_2023_2025': allp[allp.fold_test_year.lt(2026)], '2026_january': jan,
              '2026_common_january': jan[jan.date.isin(common)], '2026_common_march': march[march.date.isin(common)]}
    saved = pd.read_csv(engine.HERE / 'simultaneous_intervals.csv', dtype={'block': str})
    for track, p in tracks.items():
        losses = p.assign(loss=(p.probability - p.target)**2).pivot(
            index=['fold_test_year', 'date'], columns='config_id', values='loss')
        assert losses.notna().all().all()
        names = [c for c in losses if c != 'v3_120m']
        delta = losses[names].subtract(losses.v3_120m, axis=0)
        values = delta.to_numpy(); dates = delta.index.get_level_values('date')
        years = delta.index.get_level_values('fold_test_year').to_numpy()
        for block in ('month', '20', '60'):
            labels = dates.to_period('M').astype(str).to_numpy() if block == 'month' else np.arange(len(dates)) // int(block)
            rng = np.random.default_rng(20260905)
            sampled_sums = np.zeros((10000, len(names))); sampled_counts = np.zeros(10000)
            for year in sorted(set(years)):
                group_labels = sorted(set(labels[years == year]))
                positions = [np.flatnonzero((years == year) & (labels == label)) for label in group_labels]
                sums = np.array([values[ix].sum(axis=0) for ix in positions])
                counts = np.array([len(ix) for ix in positions])
                indices = rng.integers(0, len(positions), size=(10000, len(positions)))
                sampled_sums += sums[indices].sum(axis=1)
                sampled_counts += counts[indices].sum(axis=1)
            draws = sampled_sums / sampled_counts[:, None]
            observed = values.mean(axis=0); se = draws.std(axis=0, ddof=1)
            critical = np.quantile(np.max(abs((draws - observed) / se), axis=1), .95)
            ref = saved[saved.track.eq(track) & saved.block.eq(block)].set_index('config_id').loc[names]
            max_error = float(max(np.max(abs(ref.delta_brier.to_numpy() - observed)),
                                  np.max(abs(ref.bootstrap_se.to_numpy() - se)),
                                  np.max(abs(ref.simultaneous_ci_low.to_numpy() - (observed - critical * se))),
                                  np.max(abs(ref.simultaneous_ci_high.to_numpy() - (observed + critical * se))),
                                  np.max(abs(ref.critical_max_abs_t.to_numpy() - critical))))
            check(f'independent_maxT_{track}_{block}', max_error < 1e-12 and len(ref) == 26,
                  maximum_numeric_error=max_error, dates=len(dates), comparisons=len(ref))

    result = dict(status='PASS' if all(c['passed'] for c in checks) else 'FAIL',
                  passed=sum(c['passed'] for c in checks), failed=sum(not c['passed'] for c in checks),
                  checks=checks, evidence=evidence, model_fits=0, api_calls=0,
                  source_sha256=sha(engine.SNAPSHOT), engine_sha256=sha(engine.__file__),
                  canonical_code_sha256=sha(engine.HERE / 'canonical_control.py'),
                  simultaneous_code_sha256=sha(engine.HERE / 'simultaneous.py'), verifier_sha256=sha(__file__),
                  scope='Independent canonical feature fingerprints, preserved early observations, label purge, '
                        'all35 canonical output parity and 12 direct-index max-statistic bootstrap reconstructions; '
                        'earlier design and source audits are separate receipts.')
    (HERE / 'final_verification.json').write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({k: result[k] for k in ('status', 'passed', 'failed')}), flush=True)
    if result['failed']:
        raise SystemExit(1)


if __name__ == '__main__':
    with threadpool_limits(limits=1), warnings.catch_warnings():
        warnings.simplefilter('ignore', pd.errors.PerformanceWarning)
        main()
