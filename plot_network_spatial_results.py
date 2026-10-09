from release_config import resource
from release_config import analysis_root
from pathlib import Path
import os, sys, csv, json, hashlib
os.environ['MPLBACKEND'] = 'Agg'
import numpy as np, matplotlib.pyplot as plt, nibabel as nib
from scipy.io import loadmat, savemat
from scipy.stats import gaussian_kde, false_discovery_control
from scipy.spatial import cKDTree
from matplotlib.colors import LogNorm
from nilearn import plotting, datasets
from nilearn.surface import load_surf_mesh
R = analysis_root()
ROOT = R / resource('resource_013')
O = ROOT / resource('resource_037')
F = O / resource('resource_038')
U = O / resource('resource_058')
U.mkdir(exist_ok=True)

def read(p):
    with p.open(encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
p1 = R / resource('resource_059')
p2 = ROOT / resource('resource_060')
p3 = ROOT / resource('resource_021')
p4 = O / resource('resource_061')
d = loadmat(p1, simplify_cells=True)
m = loadmat(p2, simplify_cells=True)
g = d['group']
Y = d['NSP_outcomes']
assert np.array_equal(m['original_index'], d['original_index']) and np.array_equal(m['group'], g)
ns = [x for x in read(p3) if x['input_scheme'] == 'Absolute' and x['model'] == 'Age_Sex_Education_FD']
ss = read(p4)
maps = {'FC': m['FC_OriginalSaved_G1'][g == 1].mean(0), 'NOS': m['SC_OriginalSaved_G1'][g == 1].mean(0), 'HC_Integration': Y[g == 1, 2:70].mean(0), 'HC_Segregation': Y[g == 1, 70:138].mean(0)}
tmaps = {k: np.array([float(ns[j]['t']) for j in js]) for (k, js) in [('Integration', range(2, 70)), ('Segregation', range(70, 138))]}
W = np.load(R / resource('resource_062'), mmap_mode='r')
nulls = {}
for (k, x) in tmaps.items():
    a = np.einsum('pij,j->pi', W, x)
    a -= a.mean(1, keepdims=True)
    for ref in ['HC_' + k, 'FC', 'NOS']:
        z = next((s for s in ss if s['outcome'] == k and s['reference'] == ref))
        y = maps[ref]
        r = np.corrcoef(x, y)[0, 1]
        yc = y - y.mean()
        rn = a @ yc / np.sqrt((a * a).sum(1) * (yc @ yc))
        p = (1 + np.sum(abs(rn) >= abs(r) - 1e-14)) / 5001
        assert abs(r - float(z['r'])) < 1e-12 and abs(p - float(z['p_spin'])) < 1e-14
        nulls[k + '__' + ref] = rn
assert np.max(abs(false_discovery_control([float(s['p_spin']) for s in ss]) - np.array([float(s['q']) for s in ss]))) < 1e-14
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42})
FS = R.parent / resource('resource_063')
labs = json.loads((R / resource('resource_064')).read_text(encoding='utf-8'))['canonical_SC_labels']['regionDescriptions'][14:]
assert len(labs) == 68 and len(set(labs)) == 68
surf = datasets.fetch_surf_fsaverage(mesh='fsaverage5', data_dir=str(ROOT / resource('resource_065')))
mapping = {}
for (hemi, short) in [('left', 'lh'), ('right', 'rh')]:
    (annot, ct, nn) = nib.freesurfer.read_annot(FS / resource('resource_054') / (short + resource('resource_055')))
    full = np.full(len(annot), -1, int)
    for (k, name) in enumerate(nn):
        key = 'ctx-' + short + '-' + name.decode()
        if key in labs:
            full[annot == k] = labs.index(key)
    (source, _) = nib.freesurfer.read_geometry(FS / resource('resource_056') / (short + '.sphere'))
    target = load_surf_mesh(surf['sphere_' + hemi]).coordinates
    source /= np.linalg.norm(source, axis=1, keepdims=True)
    target /= np.linalg.norm(target, axis=1, keepdims=True)
    (dist, ix) = cKDTree(source).query(target)
    assert dist.max() < 0.001
    mapping[hemi] = full[ix]
    assert set(full[full >= 0]) == set(range(0, 34) if hemi == 'left' else range(34, 68))
fig = plt.figure(figsize=(9.2, 5.3))
lim = float(np.ceil(max(abs(maps['FC']).max(), abs(maps['NOS']).max()) * 2) / 2)
for (col, (ref, title)) in enumerate([('FC', 'a  Functional G1'), ('NOS', 'b  Structural G1 (CSD / NOS)')]):
    fig.text(0.035 + col * 0.49, 0.945, title, fontsize=12)
    for (row, (hemi, view)) in enumerate([('left', 'lateral'), ('left', 'medial'), ('right', 'lateral'), ('right', 'medial')]):
        rr = row // 2
        cc = row % 2
        idx = mapping[hemi]
        v = np.zeros(len(idx))
        v[idx >= 0] = maps[ref][idx[idx >= 0]]
        ax = fig.add_axes([0.005 + col * 0.49 + cc * 0.225, 0.46 - rr * 0.36, 0.225, 0.38], projection='3d')
        plotting.plot_surf_stat_map(surf['infl_' + hemi], v, hemi=hemi, view=view, axes=ax, figure=fig, cmap='RdBu_r', vmin=-lim, vmax=lim, threshold=None, colorbar=False, avg_method='mean')
        ax.set_title(hemi[0].upper() + ' ' + view, fontsize=8, pad=-12)
sm = plt.cm.ScalarMappable(norm=plt.Normalize(-lim, lim), cmap='RdBu_r')
cb = fig.colorbar(sm, cax=fig.add_axes([0.28, 0.085, 0.44, 0.022]), orientation='horizontal')
cb.set_label('HC mean of within-participant standardized G1 coordinates', fontsize=9)
fig.savefig(F / resource('resource_066'), dpi=600, facecolor='white')
fig.savefig(F / resource('resource_067'), facecolor='white')
plt.close(fig)
(fig, axes) = plt.subplots(2, 3, figsize=(9.7, 6.2), gridspec_kw={'width_ratios': [0.9, 1.15, 1.15]})
for (row, (k, short, js)) in enumerate([('Integration', 'HIN', slice(2, 70)), ('Segregation', 'HSE', slice(70, 138))]):
    ax = axes[row, 0]
    for (gg, c, name) in [(1, '#355CBA', 'HC'), (2, '#D62728', 'ALS')]:
        v = np.sort(Y[g == gg, js].ravel())
        ax.step(v, np.arange(1, len(v) + 1) / len(v), where='post', color=c, lw=1.6, label=name)
    ax.set_title(('a' if row == 0 else 'd') + '  ECDF of ' + short, loc='left')
    ax.set_xlabel(short + ' (regional values)')
    ax.set_ylabel('ECDF')
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, fontsize=8, loc='lower right')
    ax.grid(alpha=0.1)
    for (col, (ref, mod)) in enumerate([('FC', 'Functional G1'), ('NOS', 'Structural G1')], start=1):
        ax = axes[row, col]
        x = maps[ref]
        y = tmaps[k]
        z = next((s for s in ss if s['outcome'] == k and s['reference'] == ref))
        dx = np.ptp(x) * 0.1
        dy = np.ptp(y) * 0.1
        xx = np.linspace(x.min() - dx, x.max() + dx, 130)
        yy = np.linspace(y.min() - dy, y.max() + dy, 130)
        (gx, gy) = np.meshgrid(xx, yy)
        den = gaussian_kde(np.vstack([x, y]))(np.vstack([gx.ravel(), gy.ravel()])).reshape(gx.shape)
        top = den.max()
        levels = np.geomspace(top * 0.006, top, 22)
        ax.contourf(gx, gy, den, levels=levels, cmap='turbo', norm=LogNorm(vmin=levels[0], vmax=top), extend='min')
        ax.scatter(x, y, s=7, c='black', alpha=0.28, linewidth=0)
        co = np.polyfit(x, y, 1)
        ax.plot(xx, np.polyval(co, xx), color='#B01724', lw=1.6)
        ax.axhline(0, color='black', lw=0.6, alpha=0.55)
        ax.axvline(0, color='black', lw=0.6, alpha=0.55)
        letter = 'bcef'[row * 2 + col - 1]
        ax.set_title(f"{letter}  r = {float(z['r']):.4f}\nspin p = {float(z['p_spin']):.5g}; q = {float(z['q']):.5g}", loc='left', fontsize=9)
        ax.set_xlabel(mod + ' (HC mean)')
        ax.set_ylabel(short + ' group t (ALS - HC)')
        ax.set_xlim(xx[0], xx[-1])
        ax.set_ylim(yy[0], yy[-1])
fig.tight_layout(pad=1.3, h_pad=2, w_pad=1.3)
fig.savefig(F / resource('resource_068'), dpi=600, facecolor='white')
fig.savefig(F / resource('resource_069'), facecolor='white')
plt.close(fig)
savemat(U / resource('resource_070'), dict(original_index=d['original_index'], group=g, canonical_DK68_labels=np.array(labs, dtype=object), HC_FC_G1=maps['FC'], HC_SC_G1=maps['NOS'], NSP_Integration_t=tmaps['Integration'], NSP_Segregation_t=tmaps['Segregation'], regional_HIN=Y[:, 2:70], regional_HSE=Y[:, 70:138], **nulls))
(U / resource('resource_071')).write_text(json.dumps({'dpi': 600, 'Figure5': 'HC42 mean of original saved DK68 individual G1, pooled ALS+HC alignment reference; no AP label or unverified EV percentage; named FreeSurfer labels mapped to fsaverage5', 'Figure7': 'Original two-by-three layout; raw calibrated participant-by-region ECDF descriptive only; current age/sex/education/fMRI-FD adjusted ALS-minus-HC t, original spin operators, six-comparison BH-FDR', 'spin_verified_against_current_table': True, 'n_HC': 42, 'n_ALS': 103, 'brain_parcels': 68, 'source_sha256': {str(p): sha(p) for p in [p1, p2, p3, p4]}, 'statistics': ss}, indent=2), encoding='utf-8')
print('Figures 5 and 7 saved and verified at 600dpi; Figure7 statistics match SP S16.')
