"""Reproduces the sample profile (Table 6) and the outcome prevalence chart (Figure 3).

Run from the repository root:
    python src/02_descriptives.py
Outputs: outputs/table6_sample_profile.csv, outputs/figure3_outcomes.png
"""
import os, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prep import load

d = load(os.path.join(ROOT, 'data', 'raw', 'cfpb_nafbs-puf-labeled_2026-02.csv'))
w = d.WEIGHT_FINAL

# ---- Table 6: unweighted n, unweighted % and weighted % (denominator = all respondents)
rows = []
for var, label in [('AGE4', 'Age group'), ('SEX', 'Sex'), ('RACE4', 'Race and ethnicity'), ('INCOME4', 'Household income'),
                   ('TYPE', 'Institution type'), ('MOBILE', 'Mobile banking, past month'), ('WEB', 'Online banking, past month'),
                   ('ACC2_6', 'P2P payment service'), ('ACC2_5', 'Automated chat'), ('ALERT_1', 'Account alerts')]:
    for level in d[var].dropna().unique():
        m = d[var] == level
        rows.append(dict(characteristic=label, level=level, n=int(m.sum()),
                         unweighted_pct=round(100 * m.mean(), 1), weighted_pct=round(100 * w[m].sum() / w.sum(), 1)))
pd.DataFrame(rows).to_csv(os.path.join(ROOT, 'outputs', 'table6_sample_profile.csv'), index=False)

# ---- Figure 3: prevalence of the study outcomes
labels = {'FRAUD_EXP': 'Fraud or scam experienced', 'FALSE_POS': 'False-positive intervention',
          'FIN_LOSS': 'Money taken, among victims', 'UNREC_LOSS': 'Loss not fully recovered, among victims'}
res = []
for k, lab in labels.items():
    s = d[k].dropna()
    res.append((f'{lab}\n(n = {len(s):,})', 100 * s.mean(), 100 * np.average(s, weights=w[s.index])))
fig, ax = plt.subplots(figsize=(6.5, 3.3)); y = np.arange(len(res)); h = 0.34
b1 = ax.barh(y - h / 2 - 0.02, [r[1] for r in res], h, color='#2a78d6', label='Unweighted sample')
b2 = ax.barh(y + h / 2 + 0.02, [r[2] for r in res], h, color='#eb6834', hatch='////', edgecolor='white', linewidth=0, label='Weighted (WEIGHT_FINAL)')
for bars in (b1, b2):
    for r in bars:
        ax.text(r.get_width() + 1, r.get_y() + r.get_height() / 2, f'{r.get_width():.1f}%', va='center', fontsize=9)
ax.set_yticks(y); ax.set_yticklabels([r[0] for r in res], fontsize=9); ax.invert_yaxis(); ax.set_xlim(0, 72)
ax.set_xlabel('Percentage of respondents with a valid answer', fontsize=9)
for sp in ('top', 'right', 'left'): ax.spines[sp].set_visible(False)
ax.tick_params(axis='y', length=0); ax.xaxis.grid(True, color='#e6e5e0', lw=0.6); ax.set_axisbelow(True); ax.legend(frameon=False, fontsize=9, loc='lower right')
fig.savefig(os.path.join(ROOT, 'outputs', 'figure3_outcomes.png'), dpi=200, bbox_inches='tight', facecolor='white')
for lab, u, wt in res: print(f"{lab.splitlines()[0]:42s} unweighted {u:5.1f}%  weighted {wt:5.1f}%")
