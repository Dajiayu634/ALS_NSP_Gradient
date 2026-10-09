from release_config import resource
from release_config import analysis_root
from pathlib import Path
import sys, csv, json
import numpy as np
from scipy import stats
from scipy.io import loadmat, savemat
R = analysis_root()
ROOT = R / resource('resource_013')
O = ROOT / resource('resource_037')
O.mkdir(exist_ok=True)

def read(p):
    with p.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def write(n, rows):
    with (O / n).open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, list(dict.fromkeys((k for x in rows for k in x))))
        w.writeheader()
        w.writerows(rows)

def bh(p):
    return stats.false_discovery_control(p, method='bh')
d = loadmat(R / resource('resource_003') / resource('resource_002'), simplify_cells=True)
ids = d['original_index'].astype(int)
g = d['group']
age = d['age']
sex = (d['sex'] == 2).astype(float)
edu = d['education']
fd = d['meanFD']
Y = d['NSP_outcomes']
C = d['clinical']
features = list(d['feature_names'])
clinical = list(d['clinical_names'])
yeo = d['roi2yeo']
nets = ['VIS', 'SMN', 'DAN', 'VAN', 'LIM', 'FPN', 'DMN']
rms = loadmat(ROOT / resource('resource_072') / resource('resource_079'), simplify_cells=True)['mean_relative_RMS_mm']
X = np.c_[np.ones(145), g == 2, age, sex, edu, rms]

def fit(y, x):
    co = np.linalg.lstsq(x, y, rcond=None)[0]
    df = len(x) - np.linalg.matrix_rank(x)
    e = y - x @ co
    v = np.linalg.inv(x.T @ x)[1, 1]
    se = np.sqrt((e * e).sum(0) / df * v)
    t = co[1] / se
    p = 2 * stats.t.sf(abs(t), df)
    crit = stats.t.ppf(0.975, df)
    return (co[1], t, p, co[1] - crit * se, co[1] + crit * se, df)
fa = loadmat(R / resource('resource_083') / resource('resource_082') / resource('resource_081'), simplify_cells=True)
print('FAfields', [(k, getattr(v, 'shape', None)) for (k, v) in fa['FA'].items()], flush=True)
assert set(fa['original_index']) == set(ids)
fo = [int(np.flatnonzero(fa['original_index'] == i)[0]) for i in ids]
ni = loadmat(R / resource('resource_085') / resource('resource_084'), simplify_cells=True)
no = [int(np.flatnonzero(ni['original_index'] == i)[0]) for i in ids]
assert np.max(abs(ni['FA_G1'].T[no] - d['FA_G1'])) < 1e-10
z = np.asarray(d['FA_G1'])
assert z.shape == (145, 68) and np.isfinite(z).all()
ym = np.column_stack([z[:, yeo == i].mean(1) for i in range(1, 8)])
(b, t, p, lo, hi, df) = fit(ym, X)
q = bh(p)
faout = [dict(setting='CSD_FA', network=n, n=145, df=df, beta=b[j], ci_low=lo[j], ci_high=hi[j], t=t[j], p=p[j], q=q[j]) for (j, n) in enumerate(nets)]
write(resource('resource_086'), faout)
print('FA significant', [(x['network'], x['q']) for x in faout if x['q'] < 0.05], flush=True)
pairs = read(R / resource('resource_089') / resource('resource_088') / resource('resource_087'))
ih = np.array([int(x['HC_original_index']) - 1 for x in pairs])
ia = np.array([int(x['ALS_original_index']) - 1 for x in pairs])
idx = np.r_[ih, ia]
assert len(set(idx)) == 84 and (g[ih] == 1).all() and (g[ia] == 2).all()
for (key, a) in [('education', edu), ('age', age), ('sex', d['sex']), ('meanFD', fd)]:
    for (row, i, j) in zip(pairs, ih, ia):
        assert abs(float(row['HC_' + key]) - a[i]) < 1e-10 and abs(float(row['ALS_' + key]) - a[j]) < 1e-10
old = read(R / resource('resource_089') / resource('resource_088') / resource('resource_090'))
xm = np.c_[np.ones(84), g[idx] == 2, age[idx], sex[idx], fd[idx]]
(b, t, p, lo, hi, df) = fit(Y[idx], xm)
assert max(abs(b - np.array([float(x['ALS_minus_HC_beta']) for x in old]))) < 1e-10
matched = []
for (model, xx) in [('Supplied_No_Education', xm), ('Unified_With_Education', np.c_[xm, edu[idx]])]:
    (b, t, p, lo, hi, df) = fit(Y[idx], xx)
    q = np.zeros(164)
    for jj in [list(range(2)), list(range(2, 70)), list(range(70, 138)), list(range(138, 145)), list(range(145, 152)), list(range(152, 158)), list(range(158, 164))]:
        q[jj] = bh(p[jj])
    for (j, oldr) in enumerate(old):
        matched.append(dict(model=model, feature=features[j], scope=oldr['scope'], index=oldr['metric'], n=84, n_HC=42, n_ALS=42, df=df, beta=b[j], ci_low=lo[j], ci_high=hi[j], t=t[j], p=p[j], q=q[j], source_q=float(oldr['q_BH_family'])))
write(resource('resource_091'), matched)
balance = []
for (key, a) in [('age', age), ('education', edu), ('meanFD', fd)]:
    h = a[ih]
    al = a[ia]
    pooled = np.sqrt((h.var(ddof=1) + al.var(ddof=1)) / 2)
    balance.append(dict(variable=key, HC_mean=h.mean(), HC_SD=h.std(ddof=1), ALS_mean=al.mean(), ALS_SD=al.std(ddof=1), SMD=(al.mean() - h.mean()) / pooled, Welch_p=stats.ttest_ind(al, h, equal_var=False).pvalue))
balance.append(dict(variable='sex_female_proportion', HC_mean=sex[ih].mean(), HC_SD='', ALS_mean=sex[ia].mean(), ALS_SD='', SMD=(sex[ia].mean() - sex[ih].mean()) / np.sqrt((sex[ia].var() + sex[ih].var()) / 2), Welch_p=''))
write(resource('resource_092'), balance)
write(resource('resource_093'), [dict(pair=j + 1, HC_input_row=int(h + 1), ALS_input_row=int(a + 1), HC_original_study_index=int(ids[h]), ALS_original_study_index=int(ids[a])) for (j, (h, a)) in enumerate(zip(ih, ia))])

def partial(x, y, z):
    a = np.column_stack([stats.rankdata(v) for v in np.column_stack([x, y, z]).T])
    xx = np.c_[np.ones(len(x)), a[:, 2:]]
    res = a[:, :2] - xx @ np.linalg.lstsq(xx, a[:, :2], rcond=None)[0]
    rank = np.linalg.matrix_rank(xx)
    r = np.corrcoef(res.T)[0, 1]
    df = len(x) - rank - 1
    t = r * np.sqrt(df / max(1 - r * r, 1e-16))
    return (float(r), float(2 * stats.t.sf(abs(t), df)), df, res)
allr = read(R / resource('resource_001') / resource('resource_007'))
allr = [x for x in allr if x['group'] == 'ALS']
prior = read(R / resource('resource_001') / resource('resource_008'))
groupstats = [x for x in read(ROOT / resource('resource_021')) if x['input_scheme'] == 'Absolute' and x['model'] == 'Age_Sex_Education_FD']
sig = {features[j] for (j, x) in enumerate(groupstats) if float(x['q']) < 0.05}
original = []
plotdata = {}
sensitivity = []
rng = np.random.default_rng(20261007)
for old in prior:
    row = dict(old)
    f = row['feature']
    name = row['outcome']
    j = features.index(f)
    k = clinical.index(name)
    cog = k < 13
    zz = np.c_[age, sex, edu] if cog else np.c_[age, sex]
    valid = (g == 2) & np.isfinite(C[:, k]) & np.isfinite(zz).all(1)
    x = Y[valid, j]
    y = C[valid, k]
    zv = zz[valid]
    (r, p, df, res) = partial(x, y, zv)
    assert abs(r - float(row['r_partial_spearman'])) < 1e-10
    row['group_FDR_supported'] = f in sig
    row['retained_current_figure8'] = f in sig and float(row['q_within_feature_family']) < 0.05
    row['removed_reason'] = '' if f in sig else 'Imaging feature did not pass the revised group-level FDR'
    original.append(row)
    if row['retained_current_figure8']:
        plotdata[f + '__' + name] = res
    part = []
    for kk in range(13) if cog else range(13, 19):
        v = (g == 2) & np.isfinite(C[:, kk]) & np.isfinite(zz).all(1) & np.isfinite(fd)
        (rr, pp, dd, _) = partial(Y[v, j], C[v, kk], np.c_[zz[v], fd[v]])
        part.append(dict(feature=f, outcome=clinical[kk], n=int(v.sum()), df=dd, r=rr, p=pp, covariates='age, sex, education, meanFD' if cog else 'age, sex, meanFD'))
    for (rr, qq) in zip(part, bh([x['p'] for x in part])):
        rr['q'] = float(qq)
    selected = next((x for x in part if x['outcome'] == name))
    sensitivity.append(selected)
    if row['retained_current_figure8']:
        dur = C[:, 19]
        v = valid & np.isfinite(dur)
        (rr, pp, dd, _) = partial(Y[v, j], C[v, k], np.c_[zz[v], dur[v]])
        sensitivity.append(dict(feature=f, outcome=name, n=int(v.sum()), df=dd, r=rr, p=pp, q='', covariates='age, sex, education, disease duration' if cog else 'age, sex, disease duration'))
write(resource('resource_042'), original)
write(resource('resource_094'), sensitivity)
savemat(O / resource('resource_043'), plotdata)
W = np.load(R / resource('resource_003') / resource('resource_095'), mmap_mode='r')
m = loadmat(ROOT / resource('resource_040') / resource('resource_039'), simplify_cells=True)
maps = {'FC': m['FC_OriginalSaved_G1'][g == 1].mean(0), 'NOS': m['SC_OriginalSaved_G1'][g == 1].mean(0), 'HC_Integration': Y[g == 1, 2:70].mean(0), 'HC_Segregation': Y[g == 1, 70:138].mean(0)}
sp = []
for (outcome, jj) in [('Integration', range(2, 70)), ('Segregation', range(70, 138))]:
    x = np.array([float(groupstats[j]['t']) for j in jj])
    a = np.einsum('pij,j->pi', W, x)
    a -= a.mean(1, keepdims=True)
    for ref in ['HC_' + outcome, 'FC', 'NOS']:
        y = maps[ref]
        r = np.corrcoef(x, y)[0, 1]
        yc = y - y.mean()
        rn = a @ yc / np.sqrt((a * a).sum(1) * (yc @ yc))
        count = int((abs(rn) >= abs(r) - 1e-14).sum())
        sp.append(dict(outcome=outcome, reference=ref, r=r, p_spin=(count + 1) / 5001, exceedances=count, n_spins=5000))
for (row, q) in zip(sp, bh([x['p_spin'] for x in sp])):
    row['q'] = q
write(resource('resource_061'), sp)
counts = read(R / resource('resource_003') / resource('resource_096'))
counts = [float(x['whole_CSD_streamline_records']) for x in counts if x['included'] == '1']
assert len(counts) == 145
summary = {'n': 145, 'df': 139, 'FA_significant': [(x['network'], x['q']) for x in faout if x['q'] < 0.05], 'FA_min_q': min((q['q'] for q in faout)), 'subgroup_n': 84, 'subgroup_matching_is_approximate': True, 'subgroup_row_index_labels_corrected': True, 'matched_global': [x for x in matched if x['scope'] == 'Global'], 'matched_significant': [x for x in matched if x['q'] < 0.05], 'clinical_retained': [x for x in original if x['retained_current_figure8']], 'clinical_removed': [x['feature'] + ' / ' + x['outcome'] for x in original if not x['group_FDR_supported']], 'spin_results': sp, 'streamlines': dict(mean=np.mean(counts), SD=np.std(counts, ddof=1), median=np.median(counts), min=min(counts), max=max(counts))}
(O / resource('resource_028')).write_text(json.dumps(summary, indent=2, default=lambda x: x.item() if isinstance(x, np.generic) else str(x)), encoding='utf-8')
print(json.dumps(summary, default=str, indent=2), flush=True)
