"""Interim report, part 1: data-cleaning log and exploratory data analysis (EDA).
usage: python src/03_cleaning_eda.py  (optional args: <raw.csv> <outdir>)
Writes cleaning/EDA numbers to <outdir>/eda.json and figures to <outdir>/fig_*.png
"""
import sys, os, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency, fisher_exact
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prep import load

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, 'data', 'raw', 'cfpb_nafbs-puf-labeled_2026-02.csv')
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, 'outputs'); os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
DARK, MID, LIGHT = '#1f3b5c', '#5b84b1', '#b9cde5'
raw = pd.read_csv(RAW, low_memory=False)
d = load(RAW)
res = {}

# ------------------------------------------------------------------ cleaning log
SPECIAL = ['SKIPPED ON WEB', "DON'T KNOW", 'REFUSED', "I'm not sure", 'Unknown', 'I don’t know.']
res['shape'] = list(raw.shape)
res['dup_rows'] = int(raw.duplicated().sum())
res['dup_id'] = int(raw.CASEID.duplicated().sum())
res['dup_no_id'] = int(raw.drop(columns=['CASEID']).duplicated().sum())
src = ['FRAUD1', 'FRAUD2', 'FRAUD8', 'MOBILE', 'WEB', 'ACC2_5', 'ACC2_6', 'ALERT_1', 'TYPE', 'BIGBANK', 'DEVICE1', 'DEVICE2',
       'AGE4', 'SEX', 'RACETHNICITY', 'EDUC5', 'INCOME4', 'URBAN3', 'TRUST1', 'FRAUD3_1', 'FRAUD5_1', 'SCAM_3']
miss = {}
for c in src:
    s = raw[c].astype(str).str.strip()
    victim_only = c in ['FRAUD2', 'FRAUD3_1', 'FRAUD5_1', 'SCAM_3']
    blank = int((s == '').sum() + raw[c].isna().sum())
    spec = int(s.isin(SPECIAL).sum())
    base = len(raw) - blank if victim_only else len(raw)
    miss[c] = dict(special=spec, not_asked=blank, base=base, pct=spec / base)
res['missing'] = miss
# weight outliers (the only continuous analysis variable)
w = raw.WEIGHT_FINAL
q1, q3 = w.quantile(.25), w.quantile(.75); iqr = q3 - q1
res['weight'] = dict(min=float(w.min()), q1=float(q1), median=float(w.median()), q3=float(q3), max=float(w.max()), mean=float(w.mean()),
                     upper_fence=float(q3 + 1.5 * iqr), n_out=int((w > q3 + 1.5 * iqr).sum()),
                     pct_out=float((w > q3 + 1.5 * iqr).mean()), n_out60=int(((w > q3 + 1.5 * iqr) & (raw.AGE4 != '60+')).sum()))
# logical consistency: victim-only items answered only by victims; REPEAT_VICTIM defined only for victims
victim = d.FRAUD1.isin(['One time', 'Two or three times', 'More than three times'])
res['routing'] = dict(victims=int(victim.sum()), fraud2_answered=int((raw.FRAUD2.astype(str).str.strip() != '').sum()),
                      nonvictim_with_fraud2=int(((raw.FRAUD2.astype(str).str.strip() != '') & ~victim).sum()))
# analysis-variable completeness after recoding
av = ['FRAUD_EXP', 'FALSE_POS', 'MOBILE_USE', 'WEB_USE', 'P2P_USE', 'CHAT_USE', 'ALERT_USE', 'CREDIT_UNION', 'TRUST_HIGH']
res['complete'] = {c: dict(valid=int(d[c].notna().sum()), missing=int(d[c].isna().sum())) for c in av}
for c in ['FIN_LOSS', 'UNREC_LOSS', 'DEBIT_INVOLVED', 'BANK_DETECTED', 'REPEAT_VICTIM', 'PHISHING']:
    s = d.loc[d.FRAUD_EXP == 1, c]; res['complete'][c] = dict(valid=int(s.notna().sum()), missing=int(s.isna().sum()))
# TYPE "not sure" share
res['type_notsure'] = int(raw.TYPE.isin(["I'm not sure", 'SKIPPED ON WEB']).sum())

# ------------------------------------------------------------------ helpers
def rate(sub, y, x=None):
    """rate of y (0/1) with Wilson 95% CI; by x if given"""
    def one(s):
        s = s.dropna(); n = len(s); p = s.mean() if n else np.nan; z = 1.96
        den = 1 + z * z / n; c = (p + z * z / (2 * n)) / den; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
        return dict(n=int(n), events=int(s.sum()), p=float(p), lo=float(c - h), hi=float(c + h))
    if x is None: return one(sub[y])
    return {str(k): one(g[y]) for k, g in sub.groupby(x)}
def wrate(sub, y):
    s = sub.dropna(subset=[y]); return float(np.average(s[y], weights=s.WEIGHT_FINAL))
def chi(sub, y, x):
    t = sub[[x, y]].dropna(); tab = pd.crosstab(t[x], t[y])
    c2, p, dof, exp = chi2_contingency(tab, correction=False)
    v = np.sqrt(c2 / (len(t) * (min(tab.shape) - 1)))
    out = dict(n=int(len(t)), chi2=float(c2), p=float(p), V=float(v), min_exp=float(exp.min()))
    if tab.shape == (2, 2) and 1 in tab.index:
        a, b, c, dd = tab.loc[1, 1], tab.loc[1, 0], tab.loc[0, 1], tab.loc[0, 0]
        out.update(OR=float(a * dd / (b * c)), p1=float(a / (a + b)), p0=float(c / (c + dd)))
        if exp.min() < 5: out['fisher_p'] = float(fisher_exact(tab.values)[1])
    return out

# ------------------------------------------------------------------ EDA numbers
o = {}
for y in ['FRAUD_EXP', 'FIN_LOSS', 'UNREC_LOSS', 'FALSE_POS']:
    r = rate(d, y); r['wtd'] = wrate(d, y); o[y] = r
res['outcomes'] = o
rq1 = d.dropna(subset=['FRAUD_EXP']); vic = d[d.FIN_LOSS.notna()]; rq3 = d.dropna(subset=['FALSE_POS', 'FRAUD_EXP'])
F1 = ['MOBILE_USE', 'WEB_USE', 'P2P_USE', 'CHAT_USE']; F2 = ['DEBIT_INVOLVED', 'BANK_DETECTED', 'REPEAT_VICTIM', 'PHISHING']
F3 = F1 + ['CREDIT_UNION']
res['chi'] = {'RQ1': {x: chi(rq1, 'FRAUD_EXP', x) for x in F1},
              'RQ2': {x: chi(vic, 'FIN_LOSS', x) for x in F2},
              'RQ3': {x: chi(rq3, 'FALSE_POS', x) for x in F3}}
res['chi_demo'] = {y: {x: chi(sub, y, x) for x in ['AGE4', 'SEX', 'RACE4', 'EDUC4', 'INCOME4']}
                   for y, sub in [('FRAUD_EXP', rq1), ('FALSE_POS', rq3)]}
res['by_age'] = {y: rate(sub, y, 'AGE4') for y, sub in [('FRAUD_EXP', rq1), ('FALSE_POS', rq3)]}
res['by_income'] = {y: rate(sub, y, 'INCOME4') for y, sub in [('FRAUD_EXP', rq1), ('FALSE_POS', rq3)]}
res['digital_by_age'] = {x: d.groupby('AGE4')[x].mean().round(4).to_dict() for x in F1}
res['fp_by_fraud'] = rate(rq3, 'FALSE_POS', 'FRAUD_EXP')
res['fp_cross'] = {x: {f'{int(k[0])}_{int(k[1])}': v for k, v in {(fe, xv): rate(g, 'FALSE_POS') for (fe, xv), g in rq3.groupby(['FRAUD_EXP', x])}.items()} for x in F3}
t = d.dropna(subset=['TRUST_HIGH', 'FALSE_POS'])
res['trust'] = dict(by_fp=rate(t, 'TRUST_HIGH', 'FALSE_POS'), chi=chi(t, 'TRUST_HIGH', 'FALSE_POS'),
                    by_fp_nonvictim=rate(t[t.FRAUD_EXP == 0], 'TRUST_HIGH', 'FALSE_POS'))
res['loss_detail'] = dict(n_victims=int((d.FRAUD_EXP == 1).sum()), lost=o['FIN_LOSS']['events'],
                          all_rec=int((d.FRAUD2.str.contains('all of it', na=False)).sum()),
                          some_rec=int((d.FRAUD2.str.contains('some of it', na=False)).sum()),
                          none_rec=int((d.FRAUD2.str.contains('none of it', na=False)).sum()))
# cumulative overlap of the two problems (protection vs friction)
res['overlap'] = pd.crosstab(rq3.FRAUD_EXP, rq3.FALSE_POS).values.tolist()

# ------------------------------------------------------------------ figures
LAB = {'MOBILE_USE': 'Mobile banking', 'WEB_USE': 'Online banking', 'P2P_USE': 'P2P payments', 'CHAT_USE': 'Automated chat',
       'CREDIT_UNION': 'Credit union', 'DEBIT_INVOLVED': 'Debit card involved', 'BANK_DETECTED': 'Bank identified fraud',
       'REPEAT_VICTIM': 'Repeat victim (2+)', 'PHISHING': 'Phishing scam'}
def save(fig, name): fig.tight_layout(); fig.savefig(os.path.join(OUT, name), dpi=200); plt.close(fig)

# Figure: missingness of source items (special responses)
mm = sorted(((c, v['pct'] * 100, v['special']) for c, v in miss.items() if v['special'] > 0), key=lambda z: z[1])
fig, ax = plt.subplots(figsize=(7, 3.6))
ax.barh([m[0] for m in mm], [m[1] for m in mm], color=MID)
for i, m in enumerate(mm): ax.text(m[1] + .1, i, f'{m[2]} ({m[1]:.1f}%)', va='center', fontsize=8.5)
ax.set_xlabel('Special responses (skipped, not sure, don\'t know) as % of respondents asked'); ax.set_xlim(0, max(m[1] for m in mm) * 1.3)
save(fig, 'fig_missing.png')

# Figure: weight distribution (outlier check)
fig, ax = plt.subplots(1, 2, figsize=(7.5, 3), gridspec_kw={'width_ratios': [3, 1.3]})
ax[0].hist(w, bins=50, color=MID, edgecolor='white'); ax[0].axvline(res['weight']['upper_fence'], color=DARK, ls='--', lw=1)
ax[0].text(res['weight']['upper_fence'] + .15, ax[0].get_ylim()[1] * .85, f"IQR fence = {res['weight']['upper_fence']:.2f}", fontsize=8.5)
ax[0].set_xlabel('WEIGHT_FINAL'); ax[0].set_ylabel('Respondents')
g = [w[raw.AGE4 == a] for a in ['18-29', '30-44', '45-59', '60+']]
ax[1].boxplot(g, tick_labels=['18–29', '30–44', '45–59', '60+'], flierprops=dict(markersize=2, markerfacecolor=MID, markeredgecolor=MID), medianprops=dict(color=DARK))
ax[1].set_xlabel('Age group'); ax[1].set_ylabel('WEIGHT_FINAL')
save(fig, 'fig_weights.png')

# Figure: outcome prevalence (unweighted vs weighted)
labs = [('FRAUD_EXP', f"Fraud or scam experienced\n(n = {o['FRAUD_EXP']['n']:,})"), ('FALSE_POS', f"False-positive intervention\n(n = {o['FALSE_POS']['n']:,})"),
        ('FIN_LOSS', f"Money taken, among victims\n(n = {o['FIN_LOSS']['n']:,})"), ('UNREC_LOSS', f"Loss not fully recovered,\namong victims (n = {o['UNREC_LOSS']['n']:,})")]
fig, ax = plt.subplots(figsize=(7, 3.4)); yy = np.arange(len(labs))[::-1]
u = [o[k]['p'] * 100 for k, _ in labs]; wv = [o[k]['wtd'] * 100 for k, _ in labs]
ax.barh(yy + .2, u, .38, color=DARK, label='Unweighted sample'); ax.barh(yy - .2, wv, .38, color=LIGHT, label='Weighted (WEIGHT_FINAL)')
for i in range(len(labs)): ax.text(u[i] + .8, yy[i] + .2, f'{u[i]:.1f}%', va='center', fontsize=8.5); ax.text(wv[i] + .8, yy[i] - .2, f'{wv[i]:.1f}%', va='center', fontsize=8.5)
ax.set_yticks(yy, [l for _, l in labs]); ax.set_xlabel('Percentage of respondents with a valid answer'); ax.set_xlim(0, 70); ax.legend(frameon=False, loc='lower right')
save(fig, 'fig_outcomes.png')

# Figure: fraud and false-positive rate by digital behavior (RQ1 and RQ3)
fig, axs = plt.subplots(1, 2, figsize=(7.5, 3.4), sharey=True)
for ax, (y, sub, title) in zip(axs, [('FRAUD_EXP', rq1, 'Fraud experience (RQ1)'), ('FALSE_POS', rq3, 'False-positive intervention (RQ3)')]):
    for i, x in enumerate(F1):
        r = rate(sub, y, x)
        for j, (k, col) in enumerate([('0.0', LIGHT), ('1.0', DARK)]):
            rr = r.get(k) or r.get(k[0]); p = rr['p'] * 100
            ax.bar(i + (j - .5) * .38, p, .36, color=col, yerr=[[p - rr['lo'] * 100], [rr['hi'] * 100 - p]], capsize=2, ecolor='#444', error_kw=dict(lw=.8))
            ax.text(i + (j - .5) * .38, rr['hi'] * 100 + .8, f'{p:.0f}', ha='center', fontsize=7.5)
    ax.set_xticks(range(len(F1)), [LAB[x].replace(' ', '\n', 1) for x in F1], fontsize=8.5); ax.set_title(title, fontsize=10)
axs[0].set_ylabel('% of respondents (95% CI)'); axs[0].set_ylim(0, 55)
from matplotlib.patches import Patch
axs[1].legend([Patch(color=LIGHT), Patch(color=DARK)], ['Does not use', 'Uses'], frameon=False, loc='upper right', fontsize=8.5)
save(fig, 'fig_digital.png')

# Figure: by age group
fig, ax = plt.subplots(figsize=(6.5, 3.2)); ages = ['18-29', '30-44', '45-59', '60+']
for j, (y, col, lab) in enumerate([('FRAUD_EXP', DARK, 'Fraud experience'), ('FALSE_POS', LIGHT, 'False-positive intervention')]):
    r = res['by_age'][y]; p = [r[a]['p'] * 100 for a in ages]
    ax.bar(np.arange(4) + (j - .5) * .38, p, .36, color=col, label=lab, yerr=[[p[i] - r[a]['lo'] * 100 for i, a in enumerate(ages)], [r[a]['hi'] * 100 - p[i] for i, a in enumerate(ages)]], capsize=2, ecolor='#444', error_kw=dict(lw=.8))
    for i, a in enumerate(ages): ax.text(i + (j - .5) * .38, r[a]['hi'] * 100 + .8, f'{p[i]:.0f}', ha='center', fontsize=8)
ax.set_xticks(range(4), [f"{a.replace('-', '–')}\n(n = {res['by_age']['FRAUD_EXP'][a]['n']:,})" for a in ages]); ax.set_xlabel('Age group (years)')
ax.set_ylabel('% of respondents (95% CI)'); ax.set_ylim(0, 58); ax.legend(frameon=False, ncol=2, loc='upper center')
save(fig, 'fig_age.png')

# Figure: loss among victims by incident characteristics (RQ2)
fig, ax = plt.subplots(figsize=(6.8, 3.2))
for i, x in enumerate(F2):
    r = rate(vic, 'FIN_LOSS', x)
    for j, (k, col) in enumerate([('0.0', LIGHT), ('1.0', DARK)]):
        rr = r[k]; p = rr['p'] * 100
        ax.bar(i + (j - .5) * .38, p, .36, color=col, yerr=[[p - rr['lo'] * 100], [rr['hi'] * 100 - p]], capsize=2, ecolor='#444', error_kw=dict(lw=.8))
        ax.text(i + (j - .5) * .38, rr['hi'] * 100 + 1, f'{p:.0f}', ha='center', fontsize=8)
ax.axhline(o['FIN_LOSS']['p'] * 100, color='#888', ls=':', lw=1)
ax.set_xticks(range(4), [LAB[x] for x in F2], fontsize=8.5); ax.set_ylabel('% of victims who had money taken (95% CI)'); ax.set_ylim(0, 100)
ax.legend([Patch(color=LIGHT), Patch(color=DARK)], ['No', 'Yes'], frameon=False, loc='upper right', ncol=2, fontsize=8.5)
save(fig, 'fig_loss.png')

# Figure: false positives by fraud status, and trust by false-positive status (RQ3)
fig, axs = plt.subplots(1, 2, figsize=(7.5, 3.2), gridspec_kw={'width_ratios': [1.5, 1]})
ax = axs[0]
for j, (fe, col, lab) in enumerate([(0, LIGHT, 'No fraud in past year'), (1, DARK, 'Experienced fraud')]):
    p = [res['fp_cross'][x][f'{fe}_1']['p'] * 100 for x in F3]
    ax.bar(np.arange(len(F3)) + (j - .5) * .38, p, .36, color=col, label=lab)
    for i in range(len(F3)): ax.text(i + (j - .5) * .38, p[i] + .8, f'{p[i]:.0f}', ha='center', fontsize=7.5)
ax.set_xticks(range(len(F3)), [LAB[x].replace(' ', '\n', 1) for x in F3], fontsize=8); ax.set_ylabel('% with a false-positive intervention\n(among users of each feature)'); ax.set_ylim(0, 95); ax.legend(frameon=False, fontsize=8, ncol=2, loc='upper center')
ax = axs[1]; r = res['trust']['by_fp']; ks = sorted(r)
p = [r[k]['p'] * 100 for k in ks]
ax.bar([0, 1], p, .55, color=[LIGHT, DARK], yerr=[[p[i] - r[k]['lo'] * 100 for i, k in enumerate(ks)], [r[k]['hi'] * 100 - p[i] for i, k in enumerate(ks)]], capsize=3, ecolor='#444')
for i in range(2): ax.text(i, p[i] / 2, f'{p[i]:.0f}%', ha='center', color='white' if i else DARK, fontsize=9, fontweight='bold')
ax.set_xticks([0, 1], ['No false\npositive', 'False\npositive']); ax.set_ylabel('% who strongly agree they\ntrust their institution'); ax.set_ylim(0, 80)
save(fig, 'fig_fp_trust.png')

json.dump(res, open(os.path.join(OUT, 'eda.json'), 'w'), indent=1, default=float)
print(json.dumps({k: res[k] for k in ['shape', 'dup_rows', 'dup_id', 'weight', 'routing', 'outcomes', 'trust', 'fp_by_fraud', 'loss_detail', 'overlap']}, indent=1, default=float)[:4000])
for rq, v in res['chi'].items():
    for x, c in v.items(): print(rq, x, {k: round(z, 4) for k, z in c.items()})
for y, v in res['chi_demo'].items():
    for x, c in v.items(): print('demo', y, x, round(c['chi2'], 2), round(c['p'], 4), round(c['V'], 3))
print(res['by_age']); print(res['digital_by_age'])

# ------------------------------------------------------------------ processed analysis file
keep = ['CASEID', 'WEIGHT_FINAL', 'FRAUD_EXP', 'FIN_LOSS', 'UNREC_LOSS', 'FALSE_POS', 'MOBILE_USE', 'WEB_USE', 'P2P_USE', 'CHAT_USE', 'ALERT_USE',
        'CREDIT_UNION', 'DEBIT_INVOLVED', 'BANK_DETECTED', 'REPEAT_VICTIM', 'PHISHING', 'TRUST_HIGH', 'AGE4', 'SEX', 'RACE4', 'EDUC4', 'INCOME4',
        'TYPE', 'DEVICE1', 'DEVICE2', 'URBAN3']
PROC = os.path.join(ROOT, 'data', 'processed'); os.makedirs(PROC, exist_ok=True)
d[keep].to_csv(os.path.join(PROC, 'nafbs_analysis.csv'), index=False)
print('processed file', d[keep].shape)
