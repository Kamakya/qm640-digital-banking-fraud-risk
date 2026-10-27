"""Mentor feedback 2 and 3: why precision and recall are what they are, and what the false positives and
false negatives look like. Uses the selected interim model (Random Forest, original 14 features) and the
same 80/20 split as 04_models.py.
usage: python src/07_rq4_errors.py  (optional args: <raw.csv> <outdir>)
"""
import sys, os, json, warnings
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, roc_auc_score, average_precision_score
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prep import load
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, 'data', 'raw', 'cfpb_nafbs-puf-labeled_2026-02.csv')
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, 'outputs'); os.makedirs(OUT, exist_ok=True)
SEED = 640
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
DARK, MID, LIGHT, ACC = '#1f3b5c', '#5b84b1', '#b9cde5', '#c0504d'
d = load(RAW)
d['TYPE3'] = d.TYPE.where(d.TYPE.isin(['Bank', 'Credit Union', 'Online-only bank']))
d['COMPUTER_VF'] = np.where(d.DEVICE1.isna() | (d.DEVICE1 == 'SKIPPED ON WEB'), np.nan, (d.DEVICE1 == 'Very frequently').astype(float))
d['PHONE_VF'] = np.where(d.DEVICE2.isna() | (d.DEVICE2 == 'SKIPPED ON WEB'), np.nan, (d.DEVICE2 == 'Very frequently').astype(float))
BIN = ['MOBILE_USE', 'WEB_USE', 'P2P_USE', 'CHAT_USE', 'ALERT_USE', 'COMPUTER_VF', 'PHONE_VF']
CAT = ['TYPE3', 'AGE4', 'SEX', 'RACE4', 'EDUC4', 'INCOME4', 'URBAN3']
m = d.dropna(subset=['FRAUD_EXP']).reset_index(drop=True)
X = m[BIN + CAT].copy(); y = m.FRAUD_EXP.astype(int).values
for c in CAT: X[c] = X[c].astype(object)
Xtr, Xte, ytr, yte, itr, ite = train_test_split(X, y, np.arange(len(y)), test_size=.2, stratify=y, random_state=SEED)
pre = ColumnTransformer([('bin', SimpleImputer(strategy='most_frequent'), BIN),
                         ('cat', Pipeline([('imp', SimpleImputer(strategy='most_frequent')), ('oh', OneHotEncoder(drop='first', handle_unknown='ignore'))]), CAT)])
cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
# the tuned settings selected in 04_models.py
rf = Pipeline([('pre', pre), ('clf', RandomForestClassifier(n_estimators=400, max_depth=3, min_samples_leaf=50, max_features='sqrt', random_state=SEED, n_jobs=-1))]).fit(Xtr, ytr)
p = rf.predict_proba(Xte)[:, 1]
oof = cross_val_predict(rf, Xtr, ytr, cv=cv, method='predict_proba')[:, 1]
pr, rc, th = precision_recall_curve(ytr, oof); f = 2 * pr * rc / np.maximum(pr + rc, 1e-9); THR = float(th[np.argmax(f[:-1])])
res = dict(test_auc=float(roc_auc_score(yte, p)), pr_auc=float(average_precision_score(yte, p)), prevalence=float(yte.mean()), threshold=THR, n_test=int(len(yte)), events=int(yte.sum()))
print('check: test AUC', round(res['test_auc'], 4), 'threshold', round(THR, 4))

# ---------------------------------------------------------------- 1. threshold sweep (test set)
def at(t):
    yh = (p >= t).astype(int); TP = int(((yh == 1) & (yte == 1)).sum()); FP = int(((yh == 1) & (yte == 0)).sum())
    FN = int(((yh == 0) & (yte == 1)).sum()); TN = int(((yh == 0) & (yte == 0)).sum())
    prec = TP / (TP + FP) if TP + FP else np.nan; rec = TP / (TP + FN)
    return dict(threshold=round(t, 3), flagged=TP + FP, flagged_pct=(TP + FP) / len(yte), TP=TP, FP=FP, FN=FN, TN=TN, precision=prec, recall=rec,
                specificity=TN / (TN + FP), f1=2 * prec * rec / (prec + rec) if TP else 0, lift=prec / yte.mean() if TP + FP else np.nan)
q = np.quantile(p, [.9, .8, .667, .5])                                   # flag top 10%, 20%, 33%, 50% of customers
sweep = [dict(rule='Selected (max F1 on training)', **at(THR))] + [dict(rule=f'Flag top {k}', **at(float(t))) for k, t in zip(['10%', '20%', '33%', '50%'], q)]
res['sweep'] = sweep
print(pd.DataFrame(sweep).round(3).to_string(index=False))

# ---------------------------------------------------------------- 2. error groups and their profiles
te = m.iloc[ite].copy(); te['score'] = p; te['pred'] = (p >= THR).astype(int); te['y'] = yte
te['group'] = np.select([(te.y == 1) & (te.pred == 1), (te.y == 0) & (te.pred == 1), (te.y == 1) & (te.pred == 0), (te.y == 0) & (te.pred == 0)],
                        ['TP: fraud, flagged', 'FP: no fraud, flagged', 'FN: fraud, missed', 'TN: no fraud, not flagged'], default='')
prof = te.groupby('group').agg(n=('y', 'size'), mean_score=('score', 'mean'), aged_60_plus=('AGE4', lambda s: (s == '60+').mean()),
    under_45=('AGE4', lambda s: s.isin(['18-29', '30-44']).mean()), mobile=('MOBILE_USE', 'mean'), p2p=('P2P_USE', 'mean'), chat=('CHAT_USE', 'mean'),
    alerts=('ALERT_USE', 'mean'), income_100k=('INCOME4', lambda s: (s == '$100,000 or more').mean()), income_u30k=('INCOME4', lambda s: (s == 'Less than $30,000').mean()),
    bank_false_pos=('FALSE_POS', 'mean'))
order = ['TP: fraud, flagged', 'FN: fraud, missed', 'FP: no fraud, flagged', 'TN: no fraud, not flagged']
prof = prof.loc[order]; res['profile'] = prof.round(4).reset_index().to_dict(orient='records')
print(prof.round(3).to_string())
# among actual victims: what kind of fraud did the model catch vs miss?
vic = te[te.y == 1].copy()
SCAMS = {'SCAM_3': 'Phishing', 'SCAM_7': 'Online shopping scam', 'SCAM_2': 'Imposter scam'}
vrows = {}
for g, sub in vic.groupby('group'):
    vrows[g] = dict(n=len(sub), repeat_victim=sub.REPEAT_VICTIM.mean(), debit_card=sub.DEBIT_INVOLVED.mean(), money_taken=sub.FIN_LOSS.mean(),
                    **{nm: (sub[c] == 'Yes').mean() for c, nm in SCAMS.items()})
res['victims'] = {k: {kk: float(vv) for kk, vv in v.items()} for k, v in vrows.items()}
print(pd.DataFrame(vrows).T.round(3).to_string())

# ---------------------------------------------------------------- 3. example records (real test-set respondents)
show = ['CASEID', 'AGE4', 'SEX', 'INCOME4', 'MOBILE_USE', 'P2P_USE', 'CHAT_USE', 'ALERT_USE', 'TYPE3', 'score', 'y', 'pred', 'FRAUD1', 'FALSE_POS']
ex = []
for g, asc in [('TP: fraud, flagged', False), ('FP: no fraud, flagged', False), ('FN: fraud, missed', True), ('TN: no fraud, not flagged', True)]:
    sub = te[te.group == g].sort_values('score', ascending=asc).head(3)
    for _, r in sub.iterrows(): ex.append(dict(group=g, **{k: (r[k] if not isinstance(r[k], (np.floating, float)) else float(r[k])) for k in show}))
res['examples'] = ex
print(pd.DataFrame(ex).to_string(index=False))

# ---------------------------------------------------------------- figures
fig, axs = plt.subplots(1, 2, figsize=(7.6, 3.4))
ts = np.linspace(.15, .45, 61); P = [at(t) for t in ts]
axs[0].plot(ts, [r['precision'] for r in P], color=DARK, lw=1.8, label='Precision'); axs[0].plot(ts, [r['recall'] for r in P], color=ACC, lw=1.8, ls='--', label='Recall')
axs[0].plot(ts, [r['flagged_pct'] for r in P], color='#999', lw=1.2, ls=':', label='Share flagged')
axs[0].axhline(yte.mean(), color=MID, lw=.8); axs[0].text(.452, yte.mean() + .015, f'fraud rate {yte.mean():.2f}', fontsize=7.5, color=MID, ha='right')
axs[0].axvline(THR, color='#555', lw=.8, ls='-.'); axs[0].text(THR + .004, .93, f'chosen\n{THR:.2f}', fontsize=7.5, color='#555')
axs[0].set_xlabel('Classification threshold (score at or above = flag)'); axs[0].set_ylabel('Value on the test set'); axs[0].set_ylim(0, 1.02)
axs[0].legend(frameon=False, fontsize=7.5, loc='center left', bbox_to_anchor=(0.0, 0.62)); axs[0].set_title('A. The precision-recall trade-off', fontsize=10, loc='left')
pr2, rc2, _ = precision_recall_curve(yte, p)
axs[1].plot(rc2, pr2, color=DARK, lw=1.6, label=f"Random Forest (PR-AUC {res['pr_auc']:.2f})")
axs[1].axhline(yte.mean(), color='#999', ls=':', lw=1, label=f'No-skill (fraud rate {yte.mean():.2f})')
for r in sweep:
    axs[1].plot(r['recall'], r['precision'], 'o', color=ACC if r['rule'].startswith('Selected') else MID, ms=5)
    sel_ = r['rule'].startswith('Selected')
    axs[1].annotate('chosen' if sel_ else r['rule'].replace('Flag ', ''), (r['recall'], r['precision']), xytext=(4, -11) if sel_ else (4, 5), textcoords='offset points', fontsize=7)
axs[1].set_xlabel('Recall (share of fraud victims found)'); axs[1].set_ylabel('Precision (share of flagged who are victims)'); axs[1].set_ylim(0, 1); axs[1].set_xlim(0, 1.02)
axs[1].legend(frameon=False, fontsize=7.5, loc='upper right'); axs[1].set_title('B. Precision-recall curve', fontsize=10, loc='left')
fig.tight_layout(); fig.savefig(os.path.join(OUT, 'fig_pr_tradeoff.png'), dpi=200); plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 3.2))
bins = np.linspace(p.min(), p.max(), 26)
ax.hist(p[yte == 0], bins=bins, color=LIGHT, label=f'No fraud (n = {(yte == 0).sum()})'); ax.hist(p[yte == 1], bins=bins, color=DARK, alpha=.85, label=f'Fraud (n = {(yte == 1).sum()})')
ax.axvline(THR, color=ACC, ls='--', lw=1.2); ax.text(THR + .003, ax.get_ylim()[1] * .9, f'threshold {THR:.2f}\nright of line = flagged', fontsize=8, color=ACC)
ax.set_xlabel('Predicted fraud risk score'); ax.set_ylabel('Test respondents'); ax.legend(frameon=False, fontsize=8, loc='upper right')
fig.tight_layout(); fig.savefig(os.path.join(OUT, 'fig_score_overlap.png'), dpi=200); plt.close(fig)
json.dump(res, open(os.path.join(OUT, 'rq4_errors.json'), 'w'), indent=1, default=float)
