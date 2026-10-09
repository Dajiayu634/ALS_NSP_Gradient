from release_config import resource
from release_config import analysis_root
'Major 1: fixed cohort, simple covariates, existing calibrated NSP, new Yeo comparisons.'
import sys, json, csv, hashlib
from pathlib import Path
import numpy as np
from scipy import stats
from scipy.io import loadmat, savemat
R = analysis_root()
O = R / resource('resource_013')
O.mkdir(exist_ok=True)
P = R / resource('resource_003')
N = R / resource('resource_015') / resource('resource_014')
G = R / resource('resource_015') / resource('resource_016')
sources = [P / resource('resource_002'), N / resource('resource_017'), G / resource('resource_018'), R.parent / resource('resource_019')]
hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
d = loadmat(sources[0])
neg = loadmat(sources[1])
graphsaved = loadmat(sources[2])
ts = loadmat(sources[3])
ids = d['original_index'].ravel().astype(int)
group = d['group'].ravel().astype(int)
labels = d['roi2yeo'].ravel().astype(int)
assert len(ids) == 145 and len(set(ids)) == 145 and np.array_equal(np.unique(group, return_counts=True)[1], [42, 103])
for f in [neg, graphsaved]:
    assert np.array_equal(ids, f['original_index'].ravel()) and np.array_equal(group, f['group'].ravel())
assert np.max(abs(d['NSP_outcomes'] - neg['Absolute_NSP_outcomes'])) < 1e-08
age = d['age'].ravel()
edu = d['education'].ravel()
fd = d['meanFD'].ravel()
sex = (d['sex'].ravel() == 2).astype(float)
X = np.c_[np.ones(145), group == 2, age - age.mean(), sex, edu - edu.mean(), fd - fd.mean()]
assert np.isfinite(X).all() and np.linalg.matrix_rank(X) == 6

def text(v):
    while isinstance(v, np.ndarray) and v.size == 1:
        v = v.item()
    return str(v)
with (P / resource('resource_020')).open(encoding='utf-8-sig') as f:
    prev = list(csv.DictReader(f))
meta = [{k: r.get(k, '') for k in ['scope', 'index', 'feature', 'parcel', 'family_n']} for r in prev if r['model'] == 'Joint_Simple_Age_Sex']
assert len(meta) == 164

def bh(p):
    p = np.asarray(p, float)
    out = np.full_like(p, np.nan)
    valid = np.isfinite(p)
    v = p[valid]
    s = np.argsort(v)
    q = np.empty_like(v)
    q[s] = np.minimum(1, np.minimum.accumulate((v[s] * len(v) / np.arange(1, len(v) + 1))[::-1])[::-1])
    out[valid] = q
    return out

def fit(y, x=X):
    y = np.asarray(y, float)
    y = y[:, None] if y.ndim == 1 else y
    assert np.isfinite(y).all() and np.linalg.matrix_rank(x) == x.shape[1]
    (qr, rr) = np.linalg.qr(x, mode='reduced')
    coef = np.linalg.solve(rr, qr.T @ y)
    df = len(x) - x.shape[1]
    e = y - x @ coef
    inv = np.linalg.inv(rr)
    se = np.sqrt((e * e).sum(axis=0) / df * (inv @ inv.T)[1, 1])
    t = coef[1] / se
    p = 2 * stats.t.sf(abs(t), df)
    crit = stats.t.ppf(0.975, df)
    assert np.max(abs(coef - np.linalg.lstsq(x, y, rcond=None)[0])) < 1e-10
    return (coef[1], se, t, p, coef[1] - crit * se, coef[1] + crit * se, df)

def write(name, rows):
    if not rows:
        return
    with (O / name).open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, list(dict.fromkeys((k for r in rows for k in r))))
        w.writeheader()
        w.writerows(rows)

def report(y, metadata, scheme, model, x=X):
    (b, se, t, p, lo, hi, df) = fit(y, x)
    q = np.empty(len(p))
    families = {}
    for (j, m) in enumerate(metadata):
        families.setdefault(('Global', 'joint') if m['scope'] == 'Global' else (m['scope'], m['index']), []).append(j)
    for (key, jj) in families.items():
        q[jj] = bh(p[jj])
    return [dict(m, input_scheme=scheme, model=model, n=145, n_HC=42, n_ALS=103, df=int(df), beta=float(b[j]), se=float(se[j]), t=float(t[j]), p=float(p[j]), q=float(q[j]), ci_low=float(lo[j]), ci_high=float(hi[j]), mean_HC=float(y[group == 1, j].mean()), sd_HC=float(y[group == 1, j].std(ddof=1)), mean_ALS=float(y[group == 2, j].mean()), sd_ALS=float(y[group == 2, j].std(ddof=1))) for (j, m) in enumerate(metadata)]
yn = ['VIS', 'SMN', 'DAN', 'VAN', 'LIM', 'FPN', 'DMN']
iu = np.triu_indices(68, 1)
allpairs = np.arange(len(iu[0]))
same = labels[:, None] == labels[None, :]
meanrows = []
conventional = []
strong = []
audits = []
for (row, i) in enumerate(ids):
    s = ts['TS82'][i - 1, 0][14:, :128]
    assert s.shape == (68, 128) and np.isfinite(s).all()
    r = np.corrcoef(s)
    z = np.arctanh(np.clip(r, -1 + 1e-12, 1 - 1e-12))
    np.fill_diagonal(z, 0)
    meanrows.append([abs(r[iu]).mean(), np.maximum(r[iu], 0).mean(), r[iu].mean()])
    v = []
    for m in range(1, 8):
        w = (labels[iu[0]] == m) & (labels[iu[1]] == m)
        b = (labels[iu[0]] == m) ^ (labels[iu[1]] == m)
        zw = float(z[iu][w].mean())
        zb = float(z[iu][b].mean())
        assert abs(zw) > 1e-08
        v.append([zw, zb, (zw - zb) / zw])
    conventional.append(np.array(v).T.ravel())
    k = int(np.ceil(0.1 * len(allpairs)))
    order = np.lexsort((allpairs, -abs(r[iu])))
    mask = np.zeros(len(allpairs), bool)
    mask[order[:k]] = True
    sv = []
    for m in range(1, 8):
        w = mask & (labels[iu[0]] == m) & (labels[iu[1]] == m)
        b = mask & ((labels[iu[0]] == m) ^ (labels[iu[1]] == m))
        sv.append([float(r[iu][w].mean()) if w.any() else np.nan, float(r[iu][b].mean()) if b.any() else np.nan])
        audits.append({'original_index': int(i), 'network': yn[m - 1], 'retained_within': int(w.sum()), 'retained_between': int(b.sum()), 'total_retained_undirected': k})
    strong.append(np.array(sv).T.ravel())
means = np.array(meanrows)
nsp = []
for (scheme, mi) in [('Absolute', 0), ('Positive_Only', 1)]:
    y = neg[scheme + '_NSP_outcomes']
    nsp += report(y, meta, scheme, 'Age_Sex_Education_FD')
    nsp += report(y, meta, scheme, 'Age_Sex_Education_FD_InputMeanFC', np.c_[X, means[:, mi] - means[:, mi].mean()])
write(resource('resource_021'), nsp)
baseline = [r for r in nsp if r['input_scheme'] == 'Absolute' and r['model'] == 'Age_Sex_Education_FD']
for (a, b) in zip(baseline, meta):
    pass
old = [r for r in prev if r['model'] == 'Joint_Simple_Age_Sex']
for (a, b) in zip(baseline, old):
    for (key, newkey) in [('beta', 'beta'), ('p', 'p'), ('q_family', 'q')]:
        assert abs(float(b[key]) - a[newkey]) < 1e-10
comparisons = []
for model in ['Age_Sex_Education_FD', 'Age_Sex_Education_FD_InputMeanFC']:
    a = [r for r in nsp if r['input_scheme'] == 'Absolute' and r['model'] == model]
    b = [r for r in nsp if r['input_scheme'] == 'Positive_Only' and r['model'] == model]
    comparisons += [dict(scope=x['scope'], index=x['index'], feature=x['feature'], model=model, absolute_beta=x['beta'], positive_beta=y['beta'], absolute_p=x['p'], positive_p=y['p'], absolute_q=x['q'], positive_q=y['q'], same_direction=bool(x['beta'] * y['beta'] > 0)) for (x, y) in zip(a, b)]
write(resource('resource_022'), comparisons)
graphout = []
gm = [dict(scope='Global', index='Integration', feature='PC', family_n=2), dict(scope='Global', index='Segregation', feature='Q_fixed_Yeo7', family_n=2)]
for (scheme, mi) in [('Absolute', 0), ('Positive_Only', 1)]:
    graphout += report(graphsaved[scheme + '_graph'], gm, scheme, 'Age_Sex_Education_FD')
    graphout += report(graphsaved[scheme + '_graph'], gm, scheme, 'Age_Sex_Education_FD_InputMeanFC', np.c_[X, means[:, mi] - means[:, mi].mean()])
    nodes = graphsaved[scheme + '_PC_nodes']
    ym = np.column_stack([nodes[:, labels == m].mean(axis=1) for m in range(1, 8)])
    graphout += report(ym, [dict(scope='Yeo7', index='PC', feature=n, family_n=7) for n in yn], scheme, 'Age_Sex_Education_FD')
write(resource('resource_023'), graphout)
cm = [dict(scope='Yeo7', index=metric, feature=n, family_n=7) for metric in ['Within_FisherZ', 'Between_FisherZ', 'System_Segregation'] for n in yn]
traditional = report(np.array(conventional), cm, 'Signed_FisherZ_All_Edges', 'Age_Sex_Education_FD')
write(resource('resource_024'), traditional)
write(resource('resource_025'), audits)
associations = []
for scheme in ['Absolute', 'Positive_Only']:
    current = []
    for g in [1, 2]:
        keep = group == g
        zz = X[keep][:, [0, 2, 3, 4, 5]]
        qz = np.linalg.qr(zz, mode='reduced')[0]
        n = int(keep.sum())
        rank = np.linalg.matrix_rank(zz)
        for (j, pair) in enumerate(['PC_HIN', 'Q_HSE']):
            a = graphsaved[scheme + '_graph'][keep, j]
            b = neg[scheme + '_NSP_outcomes'][keep, j]
            a = a - qz @ (qz.T @ a)
            b = b - qz @ (qz.T @ b)
            rr = float(np.corrcoef(a, b)[0, 1])
            df = n - rank - 1
            tt = rr * np.sqrt(df / (1 - rr ** 2))
            pv = float(2 * stats.t.sf(abs(tt), df))
            delta = stats.norm.ppf(0.975) / np.sqrt(n - (rank - 1) - 3)
            (lo, hi) = np.tanh(np.arctanh(rr) + np.array([-delta, delta]))
            current.append(dict(input_scheme=scheme, group='HC' if g == 1 else 'ALS', pair=pair, n=n, df=int(df), r=rr, p=pv, ci_low=float(lo), ci_high=float(hi)))
    for (row, q) in zip(current, bh([r['p'] for r in current])):
        row['q'] = float(q)
    associations += current
write(resource('resource_026'), associations)
counts = {}
for scheme in ['Absolute', 'Positive_Only']:
    for model in ['Age_Sex_Education_FD', 'Age_Sex_Education_FD_InputMeanFC']:
        subset = [r for r in nsp if r['input_scheme'] == scheme and r['model'] == model]
        counts[scheme + ' / ' + model] = {scope + '_' + kind: sum((r['q'] < 0.05 for r in subset if r['scope'] == scope and r['index'] == kind)) for scope in ['Global', 'Regional', 'Yeo7', 'VonEconomo6'] for kind in ['Integration', 'Segregation']}
summary = {'n': 145, 'HC': 42, 'ALS': 103, 'group_model': 'intercept + group(ALS=1) + age + sex(Female=1) + education + meanFD', 'df_primary': 139, 'df_meanFC': 138, 'FDR': 'Global2; Regional68 per index; Yeo7 per index; Von6 per index; PC/Q global2; network graph7 per metric; four fixed graph/NSP associations per input', 'pointwise_CI': True, 'negative_input': 'abs(r) versus max(r,0), diagonal retained in NSP; same original group-calibration algorithm, separately estimated per edge input', 'Barragan_reference': 'Gutierrez-Barragan et al.2022, Current Biology32:631-644.e6, DOI10.1016/j.cub.2021.12.015; mouse module comparison adapted to human Yeo7, not exact replication; not the source of PC/Q equations', 'network_adaptation': 'signed Fisher-z all edges, within-module mean, module-to-other mean, (within-between)/within; additional subject top10% abs(r) ranking, retained-link signed-r means, density ceil(0.1*2278)=228; tie broken by fixed edge order', 'sources_sha256': hashes, 'counts': counts, 'global_NSP': [r for r in nsp if r['scope'] == 'Global'], 'global_graph': [r for r in graphout if r['scope'] == 'Global'], 'native_calibration_reestimated': False, 'new_plot': False, 'source_files_unchanged': all((hashlib.sha256(p.read_bytes()).hexdigest() == h for (p, h) in zip(sources, hashes.values())))}
assert summary['source_files_unchanged']
savemat(O / resource('resource_027'), {'original_index': ids, 'group': group, 'simple_design': X, 'feature_names': d['feature_names'], 'NSP_Absolute': neg['Absolute_NSP_outcomes'], 'NSP_PositiveOnly': neg['Positive_Only_NSP_outcomes'], 'mean_FC': means, 'Yeo_Conventional': np.array(conventional), 'Yeo_StrongestLinks': np.array(strong), 'Yeo_labels': labels})
(O / resource('resource_028')).write_text(json.dumps(summary, indent=2), encoding='utf-8')
print(json.dumps({'counts': counts, 'global_NSP': summary['global_NSP'], 'global_graph': summary['global_graph']}, indent=2))
