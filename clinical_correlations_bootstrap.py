from release_config import resource
from release_config import analysis_root
'Recalculate verified NSP clinical associations without changing upstream measures.'
from pathlib import Path
import sys, csv, json, hashlib, shutil
import numpy as np
from scipy import stats
from scipy.io import loadmat
ROOT = analysis_root()
OUT = ROOT / resource('resource_001')
OUT.mkdir(exist_ok=True)
source = ROOT / resource('resource_003') / resource('resource_002')
d = loadmat(source, simplify_cells=True)
group = np.asarray(d['group'], int)
age = np.asarray(d['age'], float)
sex = (np.asarray(d['sex'], int) == 2).astype(float)
edu = np.asarray(d['education'], float)
Y = np.asarray(d['NSP_outcomes'], float)
C = np.asarray(d['clinical'], float)
names = list(d['clinical_names'])
features = list(d['feature_names'])
assert Y.shape == (145, 164) and sum(group == 2) == 103 and (sum(group == 1) == 42)
assert len(np.unique(d['original_index'])) == 145

def bh(p):
    p = np.asarray(p, float)
    ix = np.argsort(p)
    v = p[ix] * len(p) / np.arange(1, len(p) + 1)
    q = np.empty(len(p))
    q[ix] = np.minimum(np.minimum.accumulate(v[::-1])[::-1], 1)
    return q

def partial(x, y, z):
    ranks = np.column_stack([stats.rankdata(a) for a in np.column_stack([x, y, z]).T])
    design = np.column_stack([np.ones(len(x)), ranks[:, 2:]])
    rank = np.linalg.matrix_rank(design)
    residual = ranks[:, :2] - design @ np.linalg.lstsq(design, ranks[:, :2], rcond=None)[0]
    if np.min(np.std(residual, axis=0)) < 1e-12:
        return (np.nan, np.nan, len(x) - rank - 1)
    r = float(np.corrcoef(residual.T)[0, 1])
    df = len(x) - rank - 1
    t = r * np.sqrt(df / max(1 - r * r, 1e-16))
    return (r, float(2 * stats.t.sf(abs(t), df)), df)

def write(name, rows):
    keys = list(dict.fromkeys((k for x in rows for k in x)))
    with (OUT / name).open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(rows)
results = []
inputs = {}
for g in [1, 2]:
    for (j, feature) in enumerate(features):
        for (k, outcome) in enumerate(names[:19]):
            if g == 1 and k >= 13:
                continue
            cog = k < 13
            z = np.column_stack([age, sex, edu]) if cog else np.column_stack([age, sex])
            valid = (group == g) & np.isfinite(Y[:, j]) & np.isfinite(C[:, k]) & np.isfinite(z).all(axis=1)
            (x, y, zz) = (Y[valid, j], C[valid, k], z[valid])
            if len(x) < len(zz[0]) + 7 or np.ptp(y) == 0:
                continue
            (r, p, df) = partial(x, y, zz)
            if not np.isfinite(p):
                continue
            row = {'group': 'HC' if g == 1 else 'ALS', 'feature': feature, 'outcome': outcome, 'family': 'Cognitive_Psychological_13' if cog else 'Disease_6', 'covariates': 'age, sex, education' if cog else 'age, sex', 'n': len(x), 'df': df, 'r_partial_spearman': r, 'p': p}
            results.append(row)
            inputs[row['group'], feature, outcome] = (x, y, zz)
for g in ['HC', 'ALS']:
    for feature in features:
        for family in ['Cognitive_Psychological_13', 'Disease_6']:
            rows = [x for x in results if x['group'] == g and x['feature'] == feature and (x['family'] == family)]
            for (x, q) in zip(rows, bh([x['p'] for x in rows])):
                x['q_within_feature_family'] = float(q)
                x['family_n'] = len(rows)
    rows = [x for x in results if x['group'] == g]
    for (x, q) in zip(rows, bh([x['p'] for x in rows])):
        x['q_broad_NSP_search_supplementary'] = float(q)
        x['broad_family_n'] = len(rows)
prior = list(csv.DictReader((ROOT / resource('resource_006') / resource('resource_005') / resource('resource_004')).open(encoding='utf-8-sig')))
selected_keys = {(x['feature'], x['outcome']) for x in prior}
assert len(selected_keys) == 12
selected = [x.copy() for x in results if x['group'] == 'ALS' and (x['feature'], x['outcome']) in selected_keys]
assert len(selected) == 12
rng = np.random.default_rng(20261006)
for row in selected:
    (x, y, z) = inputs['ALS', row['feature'], row['outcome']]
    rr = []
    for _ in range(2000):
        ix = rng.integers(0, len(x), len(x))
        (r, _, _) = partial(x[ix], y[ix], z[ix])
        if np.isfinite(r):
            rr.append(r)
    assert len(rr) >= 1900
    (lo, hi) = np.percentile(rr, [2.5, 97.5])
    row.update(bootstrap_ci_low=float(lo), bootstrap_ci_high=float(hi), bootstrap_valid=len(rr), bootstrap_seed=20261006, bootstrap_replicates=2000)
    print(row['feature'], row['outcome'], 'r', round(row['r_partial_spearman'], 4), 'q', round(row['q_within_feature_family'], 5), flush=True)
write(resource('resource_007'), results)
write(resource('resource_008'), selected)
write(resource('resource_009'), [x for x in results if x['q_within_feature_family'] < 0.05])
lookup = {(x['feature'], x['outcome']): x for x in selected}
comparison = []
for old in prior:
    x = lookup[old['feature'], old['outcome']]
    comparison.append({'feature': x['feature'], 'outcome': x['outcome'], 'previous_edu_FD_r': float(old['r']), 'new_split_covariates_r': x['r_partial_spearman'], 'new_p': x['p'], 'new_within_feature_q': x['q_within_feature_family'], 'new_broad_search_q': x['q_broad_NSP_search_supplementary'], 'new_covariates': x['covariates'], 'n': x['n']})
write(resource('resource_010'), comparison)
errors = []
for row in selected:
    (x, y, z) = inputs['ALS', row['feature'], row['outcome']]
    a = np.column_stack([stats.rankdata(v) for v in np.column_stack([x, y, z]).T])
    design = np.column_stack([np.ones(len(x)), a[:, 2:]])
    (q, _) = np.linalg.qr(design, mode='reduced')
    res = a[:, :2] - q @ (q.T @ a[:, :2])
    errors.append(abs(float(np.corrcoef(res.T)[0, 1]) - row['r_partial_spearman']))
assert max(errors) < 1e-10
summary = {'input_sha256': hashlib.sha256(source.read_bytes()).hexdigest(), 'n_HC': 42, 'n_ALS': 103, 'cognitive_covariates': ['age', 'sex', 'education'], 'disease_covariates': ['age', 'sex'], 'FD_included': False, 'polynomial_or_interaction_terms': False, 'missing_values_zero_filled': False, 'method': 'Partial Spearman by rank transformation and covariate residualization; two-sided t approximation', 'FDR': 'BH within each imaging feature and cognitive/psychological (13) or disease (6) family, separately by cohort; broad NSP search retained separately', 'all_tests': len(results), 'original_12_significant_within_family': sum((x['q_within_feature_family'] < 0.05 for x in selected)), 'original_12_significant_broad_search': sum((x['q_broad_NSP_search_supplementary'] < 0.05 for x in selected)), 'independent_QR_max_r_error': max(errors), 'notes': ['Outcome-specific complete cases; n can differ from submitted figure.', 'NSP outcomes and calibration unchanged.', 'Original clinical code contained education for disease scores; the new disease model removes it.', 'Pointwise bootstrap intervals are not multiple-comparison-adjusted intervals.', 'Full-search output covers NSP only, not historical NCT/SFC features.']}
(OUT / resource('resource_011')).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
lines = ['# Figure 8 recalculation with separate covariates', '', 'Cognition/psychological scores: age, sex, education. Disease scores: age, sex. No FD, age-square or interaction terms. Verified raw NSP participant values retained; no missing scale is replaced by zero.', '', 'Primary FDR reproduces the original family structure: each imaging feature has a separate 13-score cognitive/psychological family and six-score disease family, separately by cohort. This does not control across all imaging features. The broad NSP-search correction is retained as a supplementary column. Original 12 panels are fixed before this rerun; no additional significant panels were substituted.', '', '| Feature | Outcome | n | r | p | Within-family q | 95% bootstrap CI |', '|---|---|---:|---:|---:|---:|---|']
for x in selected:
    lines.append(f"| {x['feature']} | {x['outcome']} | {x['n']} | {x['r_partial_spearman']:.4f} | {x['p']:.5f} | {x['q_within_feature_family']:.5f} | [{x['bootstrap_ci_low']:.4f}, {x['bootstrap_ci_high']:.4f}] |")
lines += ['', f"Original panels passing requested within-family FDR: {summary['original_12_significant_within_family']}/12. Passing broad search: {summary['original_12_significant_broad_search']}/12.", '', 'The original source code used polynomial age terms and filled missing cognitive scores with zero; the verified recalculation intentionally does neither. It therefore is not expected to reproduce the submitted figure exactly. Differences from previous revised results also involve removing FD and changing correction families; the comparison must not be described as an FD-only contrast. Existing manuscript and figure files were not overwritten.']
(OUT / resource('resource_012')).write_text('\n'.join(lines) + '\n', encoding='utf-8')
print(json.dumps(summary, ensure_ascii=False), flush=True)
