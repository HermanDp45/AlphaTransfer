"""Independent OXR audit: checkpoint replay and adversarial inputs, no tree fitting.

Run from AlphaTransfer with the ML environment used by experiment.py.
Writes only this audit directory. A fresh Platt calibration is fitted solely to
verify the saved calibration on the original purged validation rows.
"""
from pathlib import Path
import hashlib
import importlib
import json
import pickle
import sys

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def maxdiff(a, b):
    return float(np.max(np.abs(np.asarray(a, float) - np.asarray(b, float))))


def main():
    checks = []

    def check(name, passed, **details):
        checks.append(dict(check=name, passed=bool(passed), **details))
        print(name, 'PASS' if passed else 'FAIL', flush=True)

    before = {str(p): (p.stat().st_size, p.stat().st_mtime_ns)
              for p in HERE.parent.rglob('*') if p.is_file() and HERE not in p.parents}
    exp = importlib.import_module('research_v4.continuation.oxr.experiment')
    assess = importlib.import_module('research_v4.continuation.oxr.assess')
    after = {str(p): (p.stat().st_size, p.stat().st_mtime_ns)
             for p in HERE.parent.rglob('*') if p.is_file() and HERE not in p.parents}
    check('imports_do_not_mutate_experiment_files', before == after)
    core, old = exp.core, exp.old
    dev = pd.read_csv(exp.HERE/'development_predictions.csv.gz', parse_dates=['date'])
    test = pd.read_csv(exp.HERE/'test_predictions.csv.gz', parse_dates=['date'])
    specifications = {s['name']: s for s in exp.specs()}
    sensitivity_protocol = exp.HERE/'history_sensitivity_protocol.json'
    extra_pred = []
    if sensitivity_protocol.exists():
        protocol = json.loads(sensitivity_protocol.read_text())
        specifications.update({s['name']: s for s in protocol['specs']})
        extra_pred = [pd.read_csv(exp.HERE/filename, parse_dates=['date']) for filename in
                      ['sensitivity_development_predictions.csv.gz', 'sensitivity_test_predictions.csv.gz']]
        check('extra_history_trials_labelled_post_readout',
              protocol['status'] == 'exploratory_followup_after_initial_2026_readout'
              and protocol['test_selection'] == 'none')
    allpred = pd.concat([dev, test, *extra_pred], ignore_index=True)
    check('source_snapshot_matches_audited_csv', sha(exp.SNAPSHOT) ==
          'd30d226fbaa0b2d89a2e7eacf011b90af93eb60150572ec66c2fb1ff26a25db2')
    for months, filename in [(24, 'baseline_reproduction_h5_predictions.csv.gz'),
                              (120, 'basis_train_120m_h5_predictions.csv.gz')]:
        v3 = pd.read_csv(ROOT/'research_v3/models'/filename, parse_dates=['date'])
        v3 = v3[v3.fold_test_year.le(2025)]
        q = dev[dev.config_id.eq(f'v3_{months}m')]
        m = q.merge(v3, on=['date', 'corridor'], suffixes=['_new', '_old'], validate='one_to_one')
        errors = {c: maxdiff(m[c+'_new'], m[c+'_old']) for c in
                  ['probability', 'target', 'candidate_signal', 'signal', 'forward_bps']}
        check(f'v3_{months}m_exact_development_parity',
              len(m) == len(v3) == len(q) == 3635 and max(errors.values()) == 0,
              rows=len(m), maximum_errors=errors)
        v3_test = pd.read_csv(ROOT/'research_v3/models'/filename, parse_dates=['date'])
        v3_test = v3_test[v3_test.fold_test_year.eq(2026)]
        q_test = test[test.config_id.eq(f'v3_{months}m') & test.cutoff.eq('2026-01-01')]
        mt = q_test.merge(v3_test, on=['date', 'corridor'], suffixes=['_new', '_old'], validate='one_to_one')
        errors = {c: maxdiff(mt[c+'_new'], mt[c+'_old']) for c in
                  ['probability', 'target', 'candidate_signal', 'signal', 'forward_bps']}
        check(f'v3_{months}m_exact_january_2026_parity',
              len(mt) == len(v3_test) == len(q_test) == 780 and max(errors.values()) == 0,
              rows=len(mt), maximum_errors=errors)

    panels = {}

    def panel(spec):
        key = (spec['delay'], spec['since'], spec['months'] == 120)
        if key not in panels:
            panels[key] = core.add_target(exp.build_panel(key[0], key[1], extended=key[2]), 5)
        return panels[key]

    receipt_errors = []
    matched = []
    for receipt_file in sorted((exp.HERE/'output').glob('*.json')):
        receipt = json.loads(receipt_file.read_text())
        spec = receipt['spec']; p = panel(spec)
        cutoff = pd.Timestamp(receipt['cutoff'])
        end = pd.Timestamp(cutoff.year+1, 1, 1)
        tr, va, te = old.temporal_split(p, 5, cutoff, end, old.Spec(spec['name'], months=spec['months']))
        future = p.groupby('corridor').date.shift(-5)
        matured = bool((future.loc[tr.index] < cutoff-pd.DateOffset(years=1)).all()
                       and (future.loc[va.index] < cutoff).all())
        f = exp.features(spec)
        framehash = hashlib.sha256(pd.util.hash_pandas_object(
            p[['date', 'corridor', *f]], index=False).to_numpy().tobytes()).hexdigest()
        stem = receipt_file.with_suffix('')
        saved = pd.read_csv(str(stem)+'.csv.gz', parse_dates=['date'])
        aligned = saved[['date', 'corridor']].equals(te[['date', 'corridor']].reset_index(drop=True))
        checks_ok = [matured, aligned, framehash == receipt['feature_frame_sha256'],
                     sha(str(stem)+'.pkl') == receipt['checkpoint_sha256'],
                     sha(str(stem)+'.csv.gz') == receipt['predictions_sha256'],
                     sha(exp.SNAPSHOT) == receipt['source_sha256'],
                     len(tr) == receipt['train_rows'], len(te) == receipt['test_rows'],
                     not {'target', 'forward_bps', 'regret_bps', 'symmetric_bps'} & set(f)]
        if not all(checks_ok):
            receipt_errors.append(dict(file=receipt_file.name, checks=checks_ok))
        matched.append((spec['name'], str(cutoff.date()), len(te)))
    check('all_receipts_hashes_labels_and_target_rows',
          len(matched) == len(specifications)*5 and not receipt_errors,
          checkpoints=len(matched), expected_checkpoints=len(specifications)*5, errors=receipt_errors)
    for cutoff, q in allpred.groupby('cutoff'):
        reference = q[q.config_id.eq('v3_24m')].sort_values(['date', 'corridor'])
        for name, z in q.groupby('config_id'):
            z = z.sort_values(['date', 'corridor'])
            check(f'matched_scope_{cutoff}_{name}',
                  z[['date', 'corridor', 'target', 'forward_bps']].reset_index(drop=True).equals(
                      reference[['date', 'corridor', 'target', 'forward_bps']].reset_index(drop=True)),
                  rows=len(z))

    samples = [('v3_24m', '2023-01-01'), ('v3_120m', '2023-01-01'),
               ('oxr_basis_120m_delay24h', '2023-01-01'),
               ('oxr_basis_120m_delay24h', '2025-01-01'),
               ('oxr_basis_120m_delay24h', '2026-01-01'),
               ('oxr_basis_120m_delay24h', '2026-03-01'),
               ('oxr_full_120m_delay48h', '2024-01-01'),
               ('oxr_full_120m_since2020', '2024-01-01'),
               ('oxr_full_120m_since2022', '2024-01-01')]
    if sensitivity_protocol.exists():
        samples += [('oxr_basis_120m_since2020', '2026-01-01'),
                    ('oxr_basis_120m_since2022', '2026-03-01')]
    for name, day in samples:
        spec = specifications[name]; p = panel(spec)
        cutoff = pd.Timestamp(day)
        tr, va, te = old.temporal_split(p, 5, cutoff, pd.Timestamp(cutoff.year+1, 1, 1),
                                       old.Spec(name, months=spec['months']))
        stem = exp.HERE/'output'/f'{name}_{day}'
        checkpoint = pickle.loads(Path(str(stem)+'.pkl').read_bytes())
        saved = pd.read_csv(str(stem)+'.csv.gz', parse_dates=['date'])
        model, calibrator, f = checkpoint['model'], checkpoint['calibrator'], checkpoint['features']
        raw = model.predict_proba(te[f+['corridor']])[:, 1]
        prob = core.apply_platt(calibrator, raw)
        vraw = model.predict_proba(va[f+['corridor']])[:, 1]
        replay_cal = core.fit_platt_calibrator(vraw, va.target)
        vp = core.apply_platt(replay_cal, vraw)
        threshold, _, _ = core.choose_frequency_threshold(va, vp)
        hist = p[p.date.ge(cutoff-pd.DateOffset(years=1)) & p.date.lt(cutoff)].copy()
        hp = core.apply_platt(replay_cal, model.predict_proba(hist[f+['corridor']])[:, 1])
        hi = core.select_per_corridor_with_cooldown(hist, hp, threshold)
        state = core.corridor_selection_state(hist, hi)
        ps = core.selection_state(hist, core.select_portfolio_from_candidates(hist, hp, hi))
        selected = core.select_per_corridor_with_cooldown(te, prob, threshold, state)
        port = core.select_portfolio_from_candidates(te, prob, selected, ps)
        poisoned_history = hist.copy()
        poisoned_history[['target', 'forward_bps', 'regret_bps', 'symmetric_bps']] = 1e99
        poisoned_hi = core.select_per_corridor_with_cooldown(poisoned_history, hp, threshold)
        errors = dict(raw=maxdiff(raw, saved.raw_probability),
                      probability=maxdiff(prob, saved.probability),
                      recalibration=maxdiff(prob, core.apply_platt(replay_cal, raw)))
        check(f'checkpoint_replay_{name}_{day}', max(errors.values()) < 1e-12
              and threshold == checkpoint['threshold']
              and np.array_equal(te.index.isin(selected), saved.candidate_signal)
              and np.array_equal(te.index.isin(port), saved.signal)
              and np.array_equal(hi, poisoned_hi), rows=len(te), errors=errors,
              calibrator_status=replay_cal.status, threshold_exact=True)

    raw = pd.read_csv(exp.SNAPSHOT)
    for delay in (24, 48):
        for boundary in ('2023-01-01', '2026-03-01'):
            cutoff = pd.Timestamp(boundary)
            decision = cutoff.tz_localize('Europe/Moscow') + pd.Timedelta(hours=10, minutes=5)
            availability = pd.concat([pd.to_datetime(raw.published_at_utc, utc=True),
                pd.to_datetime(raw.date, utc=True)+pd.Timedelta(days=1)], axis=1).max(axis=1)
            availability += pd.Timedelta(hours=delay)
            poison = raw.copy()
            future_mask = availability.gt(decision)
            poison.loc[future_mask, 'rub_per_quote'] *= 7.123
            original = exp.build_panel(delay, raw=raw)
            altered = exp.build_panel(delay, raw=poison)
            cols = exp.RET+exp.BASIS+exp.COVER
            prior = original.date.le(cutoff)
            equal = original.loc[prior, cols].equals(altered.loc[prior, cols])
            check(f'future_source_poison_delay{delay}_{boundary}', equal,
                  source_rows_poisoned=int(future_mask.sum()), past_target_rows=int(prior.sum()))

    for boundary in ('2023-01-01', '2026-01-01', '2026-03-01'):
        spec = specifications['oxr_basis_120m_delay24h']; p = panel(spec)
        cutoff = pd.Timestamp(boundary)
        tr, va, _ = old.temporal_split(p, 5, cutoff, pd.Timestamp(cutoff.year+1, 1, 1),
                                      old.Spec(spec['name'], months=120))
        poisoned = p.copy()
        poisoned.loc[poisoned.date.ge(cutoff), 'rub_per_unit'] *= 9.123
        altered = core.add_target(poisoned, 5)
        idx = tr.index.union(va.index)
        label_cols = ['target', 'forward_bps', 'regret_bps', 'symmetric_bps']
        check(f'future_target_poison_{boundary}', p.loc[idx, label_cols].equals(altered.loc[idx, label_cols]),
              train_rows=len(tr), validation_rows=len(va))

    selection = json.loads((exp.HERE/'selection.json').read_text())
    eligible = [s['name'] for s in exp.specs() if s['months'] == 120 and s['family'] in
                ('returns', 'basis', 'full') and s['delay'] == 24 and s['since'] == '2018-06-17']
    scores = {name: float(np.mean((g.probability-g.target)**2)) for name, g in
              dev[dev.config_id.isin(eligible)].groupby('config_id')}
    check('selection_uses_development_only_and_matches_snapshot',
          selection['primary_candidate'] == min(scores, key=scores.get)
          and sha(exp.HERE/'development_predictions.csv.gz') == selection['development_predictions_sha256']
          and max(abs(scores[k]-selection['scores'][k]) for k in scores) < 1e-14,
          primary_candidate=selection['primary_candidate'])

    a = dev[dev.config_id.eq('v3_120m')].sort_values(['date', 'corridor']).reset_index(drop=True)
    b = dev[dev.config_id.eq(selection['primary_candidate'])].sort_values(['date', 'corridor']).reset_index(drop=True)
    # Independent direct row-replication bootstrap for a bounded verification.
    # Reuse the documented seed/draw layout, but rebuild all policy baselines
    # from each resampled dataframe rather than the production tensor formula.
    reps = 128
    previous_reps = assess.REPS
    assess.REPS = reps
    production = assess.paired(a, b, 'month')
    assess.REPS = previous_reps
    period = a.date.dt.to_period('M').astype(str)
    groups = a.groupby([a.fold_test_year, period]).indices
    keys = list(groups)
    rng = np.random.default_rng(20260905)
    picked = [[] for _ in range(reps)]
    for year in sorted(a.fold_test_year.unique()):
        year_keys = [k for k in keys if k[0] == year]
        draws = rng.integers(0, len(year_keys), (reps, len(year_keys)))
        for rep in range(reps):
            for j in draws[rep]:
                picked[rep].extend(groups[year_keys[j]])

    def independent_point(frame):
        n, success, expected, forward_sum = 0, 0., 0., 0.
        for _, group in frame.groupby(['fold_test_year', 'corridor']):
            chosen = group[group.candidate_signal]
            n += len(chosen)
            success += chosen.target.sum()
            expected += len(chosen)*group.target.mean()
            forward_sum += chosen.forward_bps.sum()-len(chosen)*group.forward_bps.mean()
        return dict(brier=float(np.mean((frame.probability-frame.target)**2)),
                    lift=success/expected, forward_delta_bps=forward_sum/n)

    # Independent point-score calculation, including the complete newly added
    # source-history families, without looking up any winner in 2026 outcomes.
    metric_errors = []
    for (name, cutoff), group in allpred.groupby(['config_id', 'cutoff']):
        for scope in ('all', 'KZT'):
            part = group if scope == 'all' else group[group.corridor.eq('KZT')]
            expected, actual = independent_point(part), exp.metrics(part).query('scope == @scope').iloc[0]
            error = max(abs(expected[k]-actual[k]) for k in expected)
            if error > 1e-10:
                metric_errors.append(dict(name=name, cutoff=cutoff, scope=scope, error=error))
    check('independent_brier_and_policy_points_all_checkpoints', not metric_errors,
          checkpoints=allpred.groupby(['config_id', 'cutoff']).ngroups, errors=metric_errors)

    draws = []
    for idx in picked:
        pa, pb = independent_point(a.iloc[idx]), independent_point(b.iloc[idx])
        draws.append({k: pb[k]-pa[k] for k in pa})
    dist = pd.DataFrame(draws)
    errors = {}
    for metric in dist:
        key = '' if metric == 'brier' else metric+'_'
        for quantile, suffix in [(.025, 'ci_low'), (.975, 'ci_high')]:
            errors[metric+'_'+suffix] = abs(float(dist[metric].quantile(quantile))-production[key+suffix])
    check('bootstrap_direct_dataframe_resampling_matches_tensor', max(errors.values()) < 1e-10,
          verification_draws=reps, errors=errors)

    result = dict(checks=checks, passed=sum(c['passed'] for c in checks),
                  failed=sum(not c['passed'] for c in checks), tree_fits=0,
                  platt_recalibrations_for_replay=len(samples),
                  source_sha256=sha(exp.SNAPSHOT), experiment_sha256=sha(exp.HERE/'experiment.py'),
                  assess_sha256=sha(exp.HERE/'assess.py'), verifier_sha256=sha(__file__),
                  pandas_version=pd.__version__)
    (HERE/'verification.json').write_text(json.dumps(result, indent=2, default=str)+'\n')
    print(json.dumps({k: result[k] for k in ('passed', 'failed', 'tree_fits')}, indent=2))
    if result['failed']:
        raise SystemExit(1)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
