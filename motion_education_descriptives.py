from release_config import resource
from release_config import analysis_root
from pathlib import Path
import sys, csv, json
import numpy as np
from scipy import stats
from scipy.io import loadmat
R = analysis_root()
O = R / resource('resource_013') / resource('resource_029')
O.mkdir(exist_ok=True)
d = loadmat(R / resource('resource_003') / resource('resource_002'))
ids = d['original_index'].ravel().astype(int)
g = d['group'].ravel().astype(int)

def records(p):
    with p.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))
qc = {int(x['original_index']): x for x in records(R / resource('resource_033') / resource('resource_032') / resource('resource_031') / resource('resource_030'))}
cov = {int(x['original_index']): x for x in records(R / resource('resource_003') / resource('resource_034'))}
assert len(ids) == 145 and (g == 2).sum() == 103 and ((g == 1).sum() == 42)
for (i, grp) in zip(ids, g):
    assert int(qc[i]['group_code']) == grp and int(cov[i]['group_code']) == grp
fd = d['meanFD'].ravel()
assert np.max(abs(fd - np.array([float(cov[i]['meanFD_pre_excl_first_mm']) for i in ids]))) < 1e-12
metrics = {'fMRI_mean_FD_mm': fd, 'DTI_mean_absolute_RMS_mm': np.array([float(qc[i]['absolute_RMS_mean_mm']) for i in ids]), 'DTI_mean_relative_RMS_mm': np.array([float(qc[i]['relative_RMS_excl_first_mm']) for i in ids]), 'education_years': d['education'].ravel(), 'fMRI_censored_frames': np.array([158 - int(cov[i]['retained_frames']) for i in ids])}
out = []
for (metric, y) in metrics.items():
    assert np.isfinite(y).all()
    a = y[g == 2]
    b = y[g == 1]
    va = a.var(ddof=1) / len(a)
    vb = b.var(ddof=1) / len(b)
    se = np.sqrt(va + vb)
    df = (va + vb) ** 2 / (va * va / (len(a) - 1) + vb * vb / (len(b) - 1))
    diff = a.mean() - b.mean()
    test = stats.ttest_ind(a, b, equal_var=False)
    crit = stats.t.ppf(0.975, df)
    out.append(dict(metric=metric, n_ALS=len(a), n_HC=len(b), mean_ALS=float(a.mean()), sd_ALS=float(a.std(ddof=1)), mean_HC=float(b.mean()), sd_HC=float(b.std(ddof=1)), difference_ALS_minus_HC=float(diff), ci_low=float(diff - crit * se), ci_high=float(diff + crit * se), t=float(test.statistic), df=float(df), p=float(test.pvalue), rank_p=float(stats.mannwhitneyu(a, b, alternative='two-sided').pvalue), min_ALS=float(a.min()), max_ALS=float(a.max()), min_HC=float(b.min()), max_HC=float(b.max())))
with (O / resource('resource_035')).open('w', encoding='utf-8-sig', newline='') as f:
    w = csv.DictWriter(f, out[0].keys())
    w.writeheader()
    w.writerows(out)
(O / resource('resource_036')).write_text(json.dumps({'cohort': {'ALS': 103, 'HC': 42}, 'test': 'two-sided Welch descriptive comparisons; unadjusted p; rank-test sensitivity separately recorded', 'DTI_RMS_definition': 'eddy absolute RMS mean across 36 volumes; relative RMS mean excludes artificial first zero', 'results': out}, indent=2), encoding='utf-8')
print(json.dumps(out, indent=2))
