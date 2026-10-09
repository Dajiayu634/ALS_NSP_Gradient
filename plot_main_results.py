from release_config import resource
from release_config import analysis_root
from pathlib import Path
import sys, os, csv, json
os.environ['MPLBACKEND'] = 'Agg'
import numpy as np, matplotlib.pyplot as plt, nibabel as nib
from scipy.io import loadmat
from scipy.stats import gaussian_kde
from scipy.spatial import cKDTree
from nilearn import plotting, datasets
from nilearn.surface import load_surf_mesh
R = analysis_root()
ROOT = R / resource('resource_013')
O = ROOT / resource('resource_037')
F = O / resource('resource_038')
F.mkdir(exist_ok=True)
d = loadmat(R / resource('resource_003') / resource('resource_002'), simplify_cells=True)
g = d['group']
Y = d['NSP_outcomes']
names = list(d['feature_names'])
nets = ['VIS', 'SMN', 'DAN', 'VAN', 'LIM', 'FPN', 'DMN']
vc = ['ASS', 'LIM', 'P/SS', 'INS', 'MOT', 'VIS']
colors = ['#396AB1', '#DA7C30']
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42})

def read(p):
    with p.open(encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))
r = [x for x in read(ROOT / resource('resource_021')) if x['input_scheme'] == 'Absolute' and x['model'] == 'Age_Sex_Education_FD']
assert len(r) == 164
rng = np.random.default_rng(20261007)

def save(fig, n):
    fig.savefig(F / f'Figure{n}.png', dpi=600, facecolor='white')
    fig.savefig(F / f'Figure{n}.pdf', facecolor='white')
    plt.close(fig)
    print('Figure', n, 'saved 600dpi', flush=True)

def violin(ax, cols, labels, ys, statsrows):
    for (k, c) in enumerate(cols):
        for (ii, gg) in enumerate([1, 2]):
            v = Y[g == gg, c]
            pos = k + (ii - 0.5) * 0.28
            parts = ax.violinplot(v, positions=[pos], widths=0.25, showextrema=False)
            for pc in parts['bodies']:
                pc.set_facecolor(colors[ii])
                pc.set_alpha(0.55)
            ax.scatter(pos + rng.uniform(-0.045, 0.045, len(v)), v, s=4, color=colors[ii], alpha=0.4, linewidth=0, rasterized=True)
            ax.scatter([pos], [np.median(v)], s=12, color='black', marker='_', zorder=5)
        if float(statsrows[k]['q']) < 0.05 and len(cols) > 1:
            ax.text(k, 0.96, '*', ha='center', transform=ax.get_xaxis_transform())
    ax.margins(y=0.15)
    ax.set_xticks(range(len(cols)), labels)
    ax.set_ylabel(ys)
    ax.grid(axis='y', alpha=0.15)
(fig, axes) = plt.subplots(1, 2, figsize=(6.8, 3.5))
for (j, ax) in enumerate(axes):
    violin(ax, [j], ['HC        ALS'], ['Global integration (HIN)', 'Global segregation (HSE)'][j], [r[j]])
    ax.set_title(f"{'ab'[j]}  p = {float(r[j]['p']):.5g}; q = {float(r[j]['q']):.5g}", loc='left', fontsize=9)
fig.tight_layout()
save(fig, 2)
(fig, axes) = plt.subplots(2, 2, figsize=(8.3, 6.2))
for (ax, scope, index) in zip(axes.ravel(), ['Yeo7', 'Yeo7', 'VonEconomo6', 'VonEconomo6'], ['Integration', 'Segregation', 'Integration', 'Segregation']):
    rows = [x for x in r if x['scope'] == scope and x['index'] == index]
    cols = [names.index(scope + '_' + index + '_' + x['feature']) for x in rows]
    labs = [x['feature'] for x in rows]
    violin(ax, cols, labs, index, rows)
    ax.set_title(scope + ' ' + index, loc='left')
from matplotlib.lines import Line2D
fig.legend(handles=[Line2D([0], [0], color=colors[0], lw=5, label='HC (n = 42)'), Line2D([0], [0], color=colors[1], lw=5, label='ALS (n = 103)')], loc='lower center', ncol=2, frameon=False)
fig.tight_layout(rect=[0, 0.045, 1, 1])
save(fig, 4)
m = loadmat(ROOT / resource('resource_040') / resource('resource_039'), simplify_cells=True)
gr = read(ROOT / resource('resource_040') / resource('resource_041'))
(fig, axes) = plt.subplots(2, 2, figsize=(7, 6), sharex='col')
pal = plt.cm.tab10(np.arange(7))
for (col, mod) in enumerate(['FC', 'SC']):
    z = m[mod + '_OriginalSaved_G1']
    v = np.column_stack([z[:, d['roi2yeo'] == i].mean(1) for i in range(1, 8)])
    xx = np.linspace(v.min() - 0.1, v.max() + 0.1, 300)
    for (row, gg) in enumerate([1, 2]):
        ax = axes[row, col]
        for k in range(7):
            density = gaussian_kde(v[g == gg, k])(xx)
            density = density / density.max() * 0.72
            base = 6 - k
            ax.fill_between(xx, base, base + density, color=pal[k], alpha=0.85, lw=0.3)
            stat = next((x for x in gr if x['setting'] == 'DK68_OriginalSaved_DM' and x['modality'] == mod and (x['network'] == nets[k])))
            if row == 0 and float(stat['q']) < 0.05:
                ax.text(xx[-1], base + 0.22, '*', ha='left')
        ax.set_yticks(np.arange(7) + 0.25, nets[::-1])
        ax.set_title(('HC' if gg == 1 else 'ALS') + ' ' + mod + ' G1')
        ax.spines['left'].set_visible(False)
        ax.grid(axis='x', alpha=0.15)
        ax.set_ylim(-0.1, 7)
    axes[1, col].set_xlabel('Mean standardized network G1 coordinate')
fig.tight_layout()
save(fig, 6)
cl = read(O / resource('resource_042'))
cl = [x for x in cl if x['retained_current_figure8'] == 'True']
res = loadmat(O / resource('resource_043'), simplify_cells=True)
(fig, axes) = plt.subplots(1, len(cl), figsize=(7.2, 3.5))
axes = np.atleast_1d(axes)
for (ax, x) in zip(axes, cl):
    a = res[x['feature'] + '__' + x['outcome']]
    ax.scatter(a[:, 0], a[:, 1], s=16, color='#396AB1', alpha=0.7, linewidth=0)
    co = np.polyfit(a[:, 0], a[:, 1], 1)
    xx = np.array([a[:, 0].min(), a[:, 0].max()])
    ax.plot(xx, np.polyval(co, xx), color='#C44E52', lw=1.3)
    title = 'Global HSE / lower-limb score' if x['outcome'] == 'ALSFRS_lower' else 'Right rACC HIN / progression rate'
    ax.set_title(title + '\n' + f"r = {float(x['r_partial_spearman']):.3f}; p = {float(x['p']):.5g}; q = {float(x['q_within_feature_family']):.5g}", fontsize=9)
    ax.set_xlabel('NSP rank residual (age, sex)')
    ax.set_ylabel('Clinical-score rank residual (age, sex)')
    ax.grid(alpha=0.15)
fig.tight_layout()
save(fig, 8)
FS = R.parent / resource('resource_049') / resource('resource_048') / resource('resource_047') / resource('resource_046') / resource('resource_045') / resource('resource_044')
labs = json.loads((R / resource('resource_052') / resource('resource_051') / resource('resource_050')).read_text(encoding='utf-8'))['canonical_SC_labels']['regionDescriptions'][14:]
surf = datasets.fetch_surf_fsaverage(mesh='fsaverage5', data_dir=str(ROOT / resource('resource_040') / resource('resource_053')))
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
fig = plt.figure(figsize=(8.5, 5.3))
lim = max((abs(float(x['beta'])) for x in r if x['scope'] == 'Regional' and x['index'] == 'Integration'))
lim = float(np.ceil(lim * 100) / 100)
for (row, index) in enumerate(['Integration', 'Segregation']):
    rr = [x for x in r if x['scope'] == 'Regional' and x['index'] == index]
    vals = np.array([float(x['beta']) if float(x['q']) < 0.05 else np.nan for x in rr])
    fig.text(0.025, 0.97 - row * 0.45, ('a  ' if row == 0 else 'b  ') + index + ' (68-region BH-FDR)', fontsize=11)
    for (col, (hemi, view)) in enumerate([('left', 'lateral'), ('left', 'medial'), ('right', 'medial'), ('right', 'lateral')]):
        idx = mapping[hemi]
        v = np.zeros(len(idx))
        v[idx >= 0] = np.nan_to_num(vals[idx[idx >= 0]], nan=0)
        ax = fig.add_axes([0.005 + col * 0.215, 0.54 - row * 0.45, 0.215, 0.37], projection='3d')
        plotting.plot_surf_stat_map(surf['infl_' + hemi], v, hemi=hemi, view=view, axes=ax, figure=fig, cmap='RdBu_r', vmin=-lim, vmax=lim, threshold=1e-15, colorbar=False, avg_method='mean')
        ax.set_title(hemi[0].upper() + ' ' + view, fontsize=8, pad=-10)
    if row == 1:
        fig.text(0.28, 0.12, 'No regional segregation effect passed q < 0.05.', fontsize=11)
sm = plt.cm.ScalarMappable(norm=plt.Normalize(-lim, lim), cmap='RdBu_r')
cax = fig.add_axes([0.88, 0.57, 0.012, 0.28])
cb = fig.colorbar(sm, cax=cax)
cb.set_label('Adjusted ALS - HC beta', fontsize=8)
fig.text(0.025, 0.015, 'Color: adjusted regional effect passing FDR. Gray: no displayed significant effect / unassigned cortex.', fontsize=8)
save(fig, 3)
(F / resource('resource_057')).write_text(json.dumps({'dpi': 600, 'Figures': [2, 3, 4, 6, 8], 'Figure6_changed_reason': 'Original significance markers no longer agree with demographic+motion model; regenerated with current DAN/DMN markers.', 'Figure8_panels': len(cl), 'Figure8_residuals': 'Rank residuals to represent partial Spearman; not raw clinical scores', 'Figure3_mapping': 'Named FreeSurfer DK34 LH/RH annotations; separate from the historical spin helper mapping.'}, indent=2), encoding='utf-8')
