from release_config import resource
from release_config import analysis_root
from pathlib import Path
import sys, csv, json, hashlib
import numpy as np
from scipy.io import loadmat, savemat
from scipy import stats
R = analysis_root()
ROOT = R / resource('resource_013')
A = ROOT / resource('resource_040')
O = ROOT / resource('resource_072')
O.mkdir(exist_ok=True)

def read(p):
    with p.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))
d = loadmat(R / resource('resource_003') / resource('resource_002'))
m = loadmat(A / resource('resource_039'))
ids = d['original_index'].ravel().astype(int)
grp = d['group'].ravel()
yeo = d['roi2yeo'].ravel()
assert np.array_equal(ids, m['original_index'].ravel())
qpath = R / resource('resource_033') / resource('resource_032') / resource('resource_031') / resource('resource_030')
qc = {int(x['original_index']): x for x in read(qpath)}
for (i, g) in zip(ids, grp):
    assert int(qc[i]['group_code']) == g and qc[i]['counts_consistent'] == '1' and (qc[i]['log_finished'] == '1')
rms = np.array([float(qc[i]['relative_RMS_excl_first_mm']) for i in ids])
X = np.c_[np.ones(145), grp == 2, d['age'].ravel(), d['sex'].ravel() == 2, d['education'].ravel(), rms]
assert np.isfinite(X).all() and np.linalg.matrix_rank(X) == 6
prior_csv = O / resource('resource_073') / resource('resource_041')
base = read(prior_csv if prior_csv.exists() else A / resource('resource_041'))
arrays = {key: m[key] for key in m if key.startswith('SC_')}
mapped = {'DK68_DM_Pooled': arrays['SC_DM_Pooled'], 'DK68_OriginalSaved_DM': arrays['SC_OriginalSaved_G1'], 'DK68_DM_HC_reference': arrays['SC_DM_HC'], 'DK68_PCA_Pooled': arrays['SC_PCA'], 'Lausanne114_DM_Pooled': arrays['SC_114_commonDK_G1'], 'Lausanne219_DM_Pooled': arrays['SC_219_commonDK_G1']}
B = R / resource('resource_074')
for (folder, file, keys) in [('SC_NOS_Sparsity_Sensitivity', resource('resource_075'), ['NOS_G1_80', 'NOS_G1_70']), ('SC_Distance_Attenuation_Sensitivity', resource('resource_076'), ['nearest_10pct_half_weight', 'nearest_20pct_half_weight']), ('SC_Tracking_Sensitivity', resource('resource_077'), ['DTI_G1'])]:
    z = loadmat(B / folder / file)
    ix = z.get('subject_original_index', z.get('original_index')).ravel().astype(int)
    order = [np.flatnonzero(ix == i)[0] for i in ids]
    for key in keys:
        mapped[key] = z[key][order]
out = []
(qr, rr) = np.linalg.qr(X, mode='reduced')
inv = np.linalg.inv(rr)
crit = stats.t.ppf(0.975, 139)
for setting in dict.fromkeys((x['setting'] for x in base if x['modality'] == 'SC')):
    z = mapped[setting]
    z = z[:, :, 0] if z.ndim == 3 else z
    assert z.shape == (145, 68) and np.isfinite(z).all()
    y = np.column_stack([z[:, yeo == i].mean(1) for i in range(1, 8)])
    co = np.linalg.solve(rr, qr.T @ y)
    res = y - X @ co
    se = np.sqrt((res ** 2).sum(0) / 139 * (inv @ inv.T)[1, 1])
    t = co[1] / se
    p = 2 * stats.t.sf(abs(t), 139)
    q = stats.false_discovery_control(p, method='bh')
    assert np.max(abs(co - np.linalg.lstsq(X, y, rcond=None)[0])) < 1e-09
    prior = [x for x in base if x['modality'] == 'SC' and x['setting'] == setting]
    assert len(prior) == 7
    for (j, x) in enumerate(prior):
        n = dict(x)
        n.update(beta=float(co[1, j]), t=float(t[j]), p=float(p[j]), q=float(q[j]), ci_low=float(co[1, j] - crit * se[j]), ci_high=float(co[1, j] + crit * se[j]))
        out.append(n)
lookup = {(x['setting'], x['network']): x for x in out}
allrows = [lookup[x['setting'], x['network']] if x['modality'] == 'SC' else x for x in base]
for x in allrows:
    x['motion_covariate'] = 'DTI_mean_relative_RMS_mm' if x['modality'] == 'SC' else 'fMRI_mean_FD_mm'
with (O / resource('resource_041')).open('w', encoding='utf-8-sig', newline='') as f:
    w = csv.DictWriter(f, allrows[0].keys())
    w.writeheader()
    w.writerows(allrows)
compare = [dict(setting=x['setting'], network=x['network'], old_FD_beta=x['beta'], old_FD_q=x['q'], new_RMS_beta=lookup[x['setting'], x['network']]['beta'], new_RMS_q=lookup[x['setting'], x['network']]['q']) for x in base if x['modality'] == 'SC']
with (O / resource('resource_078')).open('w', encoding='utf-8-sig', newline='') as f:
    w = csv.DictWriter(f, compare[0].keys())
    w.writeheader()
    w.writerows(compare)
savemat(O / resource('resource_079'), {'original_index': ids, 'group': grp, 'mean_relative_RMS_mm': rms, 'design_SC': X})
summary = {'status': 'passed', 'n': 145, 'ALS': 103, 'HC': 42, 'df': 139, 'SC_model': 'intercept + group + age + sex + education + DTI mean relative RMS (excluding first zero)', 'FC_NSP_model': 'intercept + group + age + sex + education + fMRI pre-censoring mean FD', 'FDR': '7 networks separately per axis, modality and setting', 'SC_tests': len(out), 'SC_families': len(mapped), 'source_QC_sha256': hashlib.sha256(qpath.read_bytes()).hexdigest(), 'primary': [x for x in out if x['setting'] == 'DK68_OriginalSaved_DM'], 'significant_SC': [x for x in out if float(x['q']) < 0.05], 'gradient_maps_unchanged': True, 'FC_NSP_results_unchanged': True}
(O / resource('resource_080')).write_text(json.dumps(summary, indent=2), encoding='utf-8')
print(json.dumps({'primary': summary['primary'], 'significant': [(x['setting'], x['network'], x['q']) for x in summary['significant_SC']]}, indent=2))
