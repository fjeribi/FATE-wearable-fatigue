"""Revision figures: Figure 5 (simulation validation) and Figure 6 (spike-in power on the real designs)."""
import common  # sets the working folder to FATE_OUT
import numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family': 'Liberation Serif', 'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.edgecolor': '#555', 'axes.labelcolor': '#222', 'xtick.color': '#444', 'ytick.color': '#444'})
BLUE, ORANGE, GREY, DARK = '#2a78d6', '#eb6834', '#b8b8b8', '#333333'
T1, T2, T3 = '#d9d9d9', '#f2b38f', '#2a78d6'
SC = ['S0_null', 'S1_leakage', 'S2_shortcut', 'S3_fatigue', 'S4_fatigue_shortcut', 'S5_fatigue_noisylab']
SCL = ['S0\nNull', 'S1\nLeakage', 'S2\nShortcut', 'S3\nFatigue', 'S4\nFatigue +\nshortcut', 'S5\nFatigue,\nnoisy label']

# ---------- Figure 5 ----------
R = pd.read_csv('sim_validation_reps.csv'); R = R[R.model == 'LR']
R['truth'] = R.scenario.isin(SC[3:])
dec = {'Random split': R.L1_perm_p_free < .05, 'Grouped CV': R.p_free < .05, 'Confound regression': R.cvcr_p < .05,
       'FATE v1 rule': R.tier_v1 == 3, 'FATE (revised)': R.tier_v2 == 3}
fig = plt.figure(figsize=(180 / 25.4, 5.6)); gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], hspace=0.75, wspace=0.18)
for j, ds in enumerate(['mefar', 'fs']):
    ax = fig.add_subplot(gs[0, j]); e = R[R.design == ds]; x = np.arange(len(SC))
    sh = np.array([[np.mean(e[e.scenario == s].tier_v2 == t) for s in SC] for t in (1, 2, 3)])
    ax.bar(x, sh[0], color=T1, width=0.7, label='Tier 1'); ax.bar(x, sh[1], bottom=sh[0], color=T2, width=0.7, label='Tier 2')
    ax.bar(x, sh[2], bottom=sh[0] + sh[1], color=T3, width=0.7, label='Tier 3')
    exp = [1, 1, 2, 3, 3, 3]
    for xi, (s, t) in enumerate(zip(SC, exp)):
        ax.text(xi, 1.03, f'{np.mean(e[e.scenario == s].tier_v2 == t):.0%}', ha='center', fontsize=7.5, color=DARK)
    ax.set_xticks(x); ax.set_xticklabels(SCL, fontsize=7); ax.set_ylim(0, 1.12); ax.set_yticks([0, .5, 1]); ax.set_yticklabels(['0', '50%', '100%'])
    ax.set_title(['(a) MEFAR-like design', '(b) FatigueSet-like design'][j], loc='left', fontsize=9.5)
    if j == 0: ax.set_ylabel('Share of replications')
axs = [fig.add_subplot(gs[1, j]) for j in range(2)]
cols = [GREY, '#8c8c8c', ORANGE, '#9ecae1', BLUE]
for j, (ax, ds) in enumerate(zip(axs, ['mefar', 'fs'])):
    k = R.design == ds; x = np.arange(len(SC)); w = 0.16
    for i, (name, d) in enumerate(dec.items()):
        acc = [np.mean(d[k & (R.scenario == s)] == R.truth[k & (R.scenario == s)]) for s in SC]
        ax.bar(x + (i - 2) * w, acc, w, color=cols[i], label=name)
    ax.set_xticks(x); ax.set_xticklabels(SCL, fontsize=7); ax.set_ylim(0, 1.05); ax.set_yticks([0, .5, 1]); ax.set_yticklabels(['0', '50%', '100%'])
    ax.set_title(['(c) MEFAR-like design', '(d) FatigueSet-like design'][j], loc='left', fontsize=9.5)
    if j == 0: ax.set_ylabel('Correct conclusion')
h1, l1 = fig.axes[0].get_legend_handles_labels(); h2, l2 = axs[0].get_legend_handles_labels()
fig.legend(h1, l1, loc='upper center', ncol=3, frameon=False, fontsize=7.8, bbox_to_anchor=(0.5, 0.995))
fig.legend(h2, l2, loc='lower center', ncol=5, frameon=False, fontsize=7.6, bbox_to_anchor=(0.5, -0.005))
fig.subplots_adjust(top=0.9, bottom=0.17, left=0.08, right=0.99)
fig.savefig('figs/fig_simulation.png', dpi=300); fig.savefig('figs/fig_simulation.tif', dpi=300, pil_kwargs={'compression': 'tiff_lzw'})

# ---------- Figure 6 ----------
S = pd.read_csv('spikein_summary.csv')
fig, axs = plt.subplots(1, 2, figsize=(180 / 25.4, 3.0), sharey=True)
for ax, ds, title in zip(axs, ['mefar', 'fs'], ['(a) MEFAR (23 workers, 46 sessions)', '(b) FatigueSet (12 participants, 108 blocks)']):
    f = S[(S.dataset == ds) & (S.scenario == 'fatigue')].sort_values('delta'); c = S[(S.dataset == ds) & (S.scenario == 'clock')].sort_values('delta')
    ax.plot(f.delta, f.tier3, color=BLUE, marker='o', ms=4, lw=1.6, label='Fatigue signal: Tier 3 (alpha 0.05)')
    ax.plot(f.delta, f.tier3_holm10, color=BLUE, marker='o', ms=3.5, lw=1.2, ls=(0, (4, 2)), mfc='white', label='Fatigue signal: Tier 3 (alpha 0.005)')
    ax.plot(c.delta, c.tier3, color=ORANGE, marker='s', ms=3.5, lw=1.4, label='Clock signal: Tier 3 (false)')
    ax.plot(c.delta, c.tier2 + c.tier3 * 0, color=ORANGE, marker='s', ms=3, lw=1, ls=(0, (1, 1.5)), mfc='white', label='Clock signal: Tier 2')
    ax.axhline(0.8, color='#999', lw=0.7, ls=(0, (4, 3)))
    for _, r in f.iterrows():
        ax.annotate(f'{r.unit_auc:.2f}', (r.delta, -0.19), xycoords=('data', 'axes fraction'), ha='center', fontsize=6.6, color='#555', annotation_clip=False)
    ax.annotate('AUC', (f.delta.min() - 0.05, -0.19), xycoords=('data', 'axes fraction'), ha='right', fontsize=6.6, color='#555', annotation_clip=False)
    ax.set_xticks(sorted(f.delta.unique())); ax.set_xticklabels([f'{v:g}' for v in sorted(f.delta.unique())], fontsize=7.5)
    ax.set_xlabel('Injected effect (pooled SD)', labelpad=18); ax.set_ylim(-0.02, 1.02); ax.set_title(title, loc='left', fontsize=9.5)
    ax.grid(axis='y', color='#e6e6e6', lw=0.6); ax.set_axisbelow(True)
axs[0].set_ylabel('Share of replications')
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc='lower center', ncol=4, frameon=False, fontsize=7.2, bbox_to_anchor=(0.5, -0.01))
fig.tight_layout(rect=(0, 0.08, 1, 1))
fig.savefig('figs/fig_spikein.png', dpi=300); fig.savefig('figs/fig_spikein.tif', dpi=300, pil_kwargs={'compression': 'tiff_lzw'})
print('figures written')
