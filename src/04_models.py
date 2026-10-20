"""Interim report, part 2: preliminary models.
RQ1-RQ3: multivariable binary logistic regression (statsmodels) with diagnostics and a weighted sensitivity model.
RQ4: baseline + regularized logistic regression, Random Forest, Gradient Boosting (scikit-learn), stratified 80/20 split,
5-fold CV tuning on training data only, single test-set evaluation, risk segments.
usage: python src/04_models.py  (optional args: <raw.csv> <outdir>)
"""
import sys, os, json, warnings
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import statsmodels.api as sm, statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.multitest import multipletests
from scipy.stats import chi2
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (roc_auc_score, average_precision_score, brier_score_loss, precision_score, recall_score,
                             f1_score, accuracy_score, confusion_matrix, roc_curve, precision_recall_curve)
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prep import load
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, 'data', 'raw', 'cfpb_nafbs-puf-labeled_2026-02.csv')
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, 'outputs'); os.makedirs(OUT, exist_ok=True)
SEED = 640
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
DARK, MID, LIGHT, ACC = '#1f3b5c', '#5b84b1', '#b9cde5', '#c0504d'
d = load(RAW); res = {}
CONTROLS = "C(AGE4, Treatment('60+')) + C(SEX) + C(RACE4, Treatment('White NH')) + C(EDUC4, Treatment('HS or less')) + C(INCOME4, Treatment('Less than $30,000'))"

# ======================================================================= RQ1-RQ3 logistic regression
def nagelkerke(m, n):
    l1, l0 = m.llf, m.llnull; cs = 1 - np.exp(2 * (l0 - l1) / n); return cs / (1 - np.exp(2 * l0 / n))
def hosmer(y, p, g=10):
    q = pd.qcut(p, g, duplicates='drop'); t = pd.DataFrame({'y': y, 'p': p, 'g': q}).groupby('g', observed=True)
    O, E, n = t.y.sum(), t.p.sum(), t.y.count(); C = (((O - E) ** 2) / (E * (1 - E / n))).sum(); k = len(O)
    return float(C), float(1 - chi2.cdf(C, k - 2))
def logit_rq(name, y, focal, sub, extra=''):
    cols = [y] + focal + ['AGE4', 'SEX', 'RACE4', 'EDUC4', 'INCOME4', 'WEIGHT_FINAL'] + (['FRAUD_EXP'] if 'FRAUD_EXP' in extra else []) + (['MOBILE_USE'] if 'MOBILE_USE' in extra else [])
    s = sub[list(dict.fromkeys(cols))].dropna().copy()
    f_full = f"{y} ~ {' + '.join(focal)} + {CONTROLS}{extra}"; f_red = f"{y} ~ {CONTROLS}{extra}"
    m = smf.logit(f_full, data=s).fit(disp=0); m0 = smf.logit(f_red, data=s).fit(disp=0)
    LR = 2 * (m.llf - m0.llf); dfLR = len(focal); pLR = 1 - chi2.cdf(LR, dfLR)
    # weighted sensitivity: GLM binomial with normalized frequency weights
    ww = s.WEIGHT_FINAL / s.WEIGHT_FINAL.mean()
    mw = smf.glm(f_full, data=s, family=sm.families.Binomial(), var_weights=ww).fit(cov_type='HC1')
    X = m.model.exog; names = m.model.exog_names
    vif = {names[i]: float(variance_inflation_factor(X, i)) for i in range(1, X.shape[1])}
    p = m.predict(s); HLC, pHL = hosmer(s[y].values, p.values)
    # influence: Cook's distance from GLM
    g = smf.glm(f_full, data=s, family=sm.families.Binomial()).fit(); cook = g.get_influence().cooks_distance[0]
    ci = m.conf_int(); rows = {}
    raw_p = [m.pvalues[x] for x in focal]; bh = multipletests(raw_p, method='fdr_bh')[1]
    for i, x in enumerate(focal):
        rows[x] = dict(b=float(m.params[x]), se=float(m.bse[x]), OR=float(np.exp(m.params[x])), lo=float(np.exp(ci.loc[x, 0])), hi=float(np.exp(ci.loc[x, 1])),
                       z=float(m.tvalues[x]), p=float(m.pvalues[x]), p_bh=float(bh[i]), OR_w=float(np.exp(mw.params[x])),
                       lo_w=float(np.exp(mw.conf_int().loc[x, 0])), hi_w=float(np.exp(mw.conf_int().loc[x, 1])), p_w=float(mw.pvalues[x]))
    ctrl = {k: dict(OR=float(np.exp(m.params[k])), lo=float(np.exp(ci.loc[k, 0])), hi=float(np.exp(ci.loc[k, 1])), p=float(m.pvalues[k]))
            for k in names if k not in focal and k != 'Intercept'}
    from sklearn.metrics import roc_auc_score as auc
    out = dict(n=int(len(s)), events=int(s[y].sum()), k=int(len(m.params) - 1), LR=float(LR), dfLR=dfLR, pLR=float(pLR),
               LR_model=float(m.llr), p_model=float(m.llr_pvalue), AIC=float(m.aic), BIC=float(m.bic), AIC0=float(m0.aic), BIC0=float(m0.bic),
               R2N=float(nagelkerke(m, len(s))), HL=HLC, pHL=pHL, AUC=float(auc(s[y], p)), maxVIF=float(max(vif.values())),
               maxVIF_name=max(vif, key=vif.get), maxCook=float(cook.max()), nCook=int((cook > 4 / len(s)).sum()), focal=rows, controls=ctrl)
    print(name, 'n', out['n'], 'events', out['events'], 'LR', round(LR, 2), 'p', pLR, 'AUC', round(out['AUC'], 3), 'R2N', round(out['R2N'], 3), 'HL p', round(pHL, 3), 'maxVIF', round(out['maxVIF'], 2))
    for x, r in rows.items(): print('   ', x, {k: round(v, 3) for k, v in r.items()})
    return out
rq1 = d.dropna(subset=['FRAUD_EXP']); vic = d[d.FIN_LOSS.notna()]; rq3 = d.dropna(subset=['FALSE_POS', 'FRAUD_EXP'])
res['RQ1'] = logit_rq('RQ1', 'FRAUD_EXP', ['MOBILE_USE', 'WEB_USE', 'P2P_USE', 'CHAT_USE'], rq1)
res['RQ2'] = logit_rq('RQ2', 'FIN_LOSS', ['DEBIT_INVOLVED', 'BANK_DETECTED', 'REPEAT_VICTIM', 'PHISHING'], vic, ' + MOBILE_USE')
res['RQ3'] = logit_rq('RQ3', 'FALSE_POS', ['MOBILE_USE', 'WEB_USE', 'P2P_USE', 'CHAT_USE', 'CREDIT_UNION'], rq3, ' + FRAUD_EXP')
# RQ3 interaction check (fraud experience x mobile use)
s3 = rq3.dropna(subset=['MOBILE_USE', 'WEB_USE', 'CREDIT_UNION'])
mi = smf.logit(f"FALSE_POS ~ MOBILE_USE*FRAUD_EXP + WEB_USE + P2P_USE + CHAT_USE + CREDIT_UNION + {CONTROLS}", data=s3).fit(disp=0)
res['RQ3']['interaction'] = dict(OR=float(np.exp(mi.params['MOBILE_USE:FRAUD_EXP'])), p=float(mi.pvalues['MOBILE_USE:FRAUD_EXP']))
print('RQ3 interaction', res['RQ3']['interaction'])

# forest plot of adjusted ORs
LAB = {'MOBILE_USE': 'Mobile banking', 'WEB_USE': 'Online banking', 'P2P_USE': 'P2P payments', 'CHAT_USE': 'Automated chat',
       'CREDIT_UNION': 'Credit union (vs. bank)', 'DEBIT_INVOLVED': 'Debit card involved', 'BANK_DETECTED': 'Bank identified the fraud',
       'REPEAT_VICTIM': 'Repeat victim (2+ times)', 'PHISHING': 'Phishing scam'}
fig, ax = plt.subplots(figsize=(7, 5.2)); yl, ylab, ypos = [], [], 0
for rq, ttl in [('RQ1', 'RQ1: Fraud experience'), ('RQ2', 'RQ2: Money taken (victims)'), ('RQ3', 'RQ3: False-positive intervention')]:
    ylab.append(('**' + ttl, ypos)); ypos -= 1
    for x, r in res[rq]['focal'].items():
        col = DARK if r['p_bh'] < .05 else '#9a9a9a'
        ax.plot([r['lo'], r['hi']], [ypos, ypos], color=col, lw=1.6); ax.plot(r['OR'], ypos, 's', color=col, ms=6)
        ax.text(9.2, ypos, f"{r['OR']:.2f} [{r['lo']:.2f}, {r['hi']:.2f}]", va='center', fontsize=8.5)
        ylab.append((LAB[x], ypos)); ypos -= 1
    ypos -= .4
ax.axvline(1, color='#555', ls='--', lw=.8); ax.set_xscale('log'); ax.set_xlim(.2, 8); ax.set_xticks([.25, .5, 1, 2, 4], ['0.25', '0.5', '1', '2', '4']); ax.minorticks_off()
ax.set_yticks([p for _, p in ylab], [t.replace('**', '') for t, _ in ylab])
for t in ax.get_yticklabels():
    if t.get_text().startswith('RQ'): t.set_fontweight('bold')
ax.text(9.2, 0.6, 'OR [95% CI]', fontsize=8.5, fontweight='bold'); ax.set_xlabel('Adjusted odds ratio (log scale)')
ax.tick_params(axis='y', length=0); fig.subplots_adjust(right=.72); fig.savefig(os.path.join(OUT, 'fig_forest.png'), dpi=200, bbox_inches='tight'); plt.close(fig)

# ======================================================================= RQ4 machine learning
FE_BIN = ['MOBILE_USE', 'WEB_USE', 'P2P_USE', 'CHAT_USE', 'ALERT_USE']
d['TYPE3'] = d.TYPE.where(d.TYPE.isin(['Bank', 'Credit Union', 'Online-only bank']))
d['COMPUTER_VF'] = np.where(d.DEVICE1.isin(['SKIPPED ON WEB']) | d.DEVICE1.isna(), np.nan, (d.DEVICE1 == 'Very frequently').astype(float))
d['PHONE_VF'] = np.where(d.DEVICE2.isin(['SKIPPED ON WEB']) | d.DEVICE2.isna(), np.nan, (d.DEVICE2 == 'Very frequently').astype(float))
FE_BIN += ['COMPUTER_VF', 'PHONE_VF']
FE_CAT = ['TYPE3', 'AGE4', 'SEX', 'RACE4', 'EDUC4', 'INCOME4', 'URBAN3']
m4 = d.dropna(subset=['FRAUD_EXP']).reset_index(drop=True)
X = m4[FE_BIN + FE_CAT].copy(); y = m4.FRAUD_EXP.astype(int).values
for c in FE_CAT: X[c] = X[c].astype(object)
res['RQ4_missing'] = {c: int(X[c].isna().sum()) for c in X.columns}
Xtr, Xte, ytr, yte, itr, ite = train_test_split(X, y, np.arange(len(y)), test_size=.2, stratify=y, random_state=SEED)
pre = ColumnTransformer([
    ('bin', SimpleImputer(strategy='most_frequent'), FE_BIN),
    ('cat', Pipeline([('imp', SimpleImputer(strategy='most_frequent')), ('oh', OneHotEncoder(drop='first', handle_unknown='ignore'))]), FE_CAT)])
cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
models = {
    'Baseline (majority class)': (DummyClassifier(strategy='prior'), {}),
    'Logistic regression (L2)': (LogisticRegression(max_iter=2000, solver='liblinear'), {'clf__C': [.01, .03, .1, .3, 1, 3]}),
    'Random Forest': (RandomForestClassifier(n_estimators=400, random_state=SEED, n_jobs=-1),
                      {'clf__max_depth': [3, 5, 8], 'clf__min_samples_leaf': [10, 25, 50], 'clf__max_features': ['sqrt', .5]}),
    'Gradient Boosting': (GradientBoostingClassifier(random_state=SEED),
                          {'clf__n_estimators': [100, 200], 'clf__learning_rate': [.03, .1], 'clf__max_depth': [2, 3], 'clf__subsample': [.8]}),
}
rng = np.random.default_rng(SEED); BOOT = [rng.integers(0, len(yte), len(yte)) for _ in range(1000)]
def boot_ci(fn, p):
    v = [fn(yte[b], p[b]) for b in BOOT if 0 < yte[b].sum() < len(b)]; return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))
R4, fitted, curves = {}, {}, {}
for name, (clf, grid) in models.items():
    pipe = Pipeline([('pre', pre), ('clf', clf)])
    gs = GridSearchCV(pipe, grid or {'clf__strategy': ['prior']}, scoring='roc_auc', cv=cv, n_jobs=-1, refit=True).fit(Xtr, ytr)
    best = gs.best_estimator_; fitted[name] = best
    oof = cross_val_predict(best, Xtr, ytr, cv=cv, method='predict_proba')[:, 1]
    # threshold chosen on training out-of-fold predictions only (maximizes F1)
    if name.startswith('Baseline'): thr = float(ytr.mean())
    else:
        pr, rc, th = precision_recall_curve(ytr, oof); f = 2 * pr * rc / np.maximum(pr + rc, 1e-9); thr = float(th[np.argmax(f[:-1])])
    p = best.predict_proba(Xte)[:, 1]; yh = (p >= thr).astype(int)
    cm = confusion_matrix(yte, yh, labels=[1, 0])
    r = dict(cv_auc=float(gs.best_score_), cv_auc_sd=float(gs.cv_results_['std_test_score'][gs.best_index_]), params={k.replace('clf__', ''): v for k, v in gs.best_params_.items()},
             thr=thr, auc=float(roc_auc_score(yte, p)), auc_ci=boot_ci(roc_auc_score, p), prauc=float(average_precision_score(yte, p)),
             brier=float(brier_score_loss(yte, p)), acc=float(accuracy_score(yte, yh)), prec=float(precision_score(yte, yh, zero_division=0)),
             rec=float(recall_score(yte, yh)), f1=float(f1_score(yte, yh)), spec=float(cm[1, 1] / cm[1].sum()), cm=cm.tolist(),
             train_auc=float(roc_auc_score(ytr, best.predict_proba(Xtr)[:, 1])))
    r['brier_ref'] = float(np.mean((yte - ytr.mean()) ** 2))
    R4[name] = r; curves[name] = p
    print(name, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
res['RQ4'] = dict(models=R4, n=len(y), n_train=len(ytr), n_test=len(yte), ev_train=int(ytr.sum()), ev_test=int(yte.sum()),
                  n_params=int(fitted['Logistic regression (L2)'].named_steps['pre'].fit_transform(Xtr).shape[1]))
# selection: highest CV AUC among non-baseline (decided on training data only)
cands = {k: v for k, v in R4.items() if not k.startswith('Baseline')}
sel = max(cands, key=lambda k: cands[k]['cv_auc']); res['RQ4']['selected'] = sel; best = fitted[sel]; ps = curves[sel]
# DeLong-free paired bootstrap difference vs logistic regression
lr = curves['Logistic regression (L2)']
for k in cands:
    dif = [roc_auc_score(yte[b], curves[k][b]) - roc_auc_score(yte[b], lr[b]) for b in BOOT]
    R4[k]['d_auc_vs_lr'] = (float(np.mean(dif)), float(np.percentile(dif, 2.5)), float(np.percentile(dif, 97.5)))
# risk segments: tertile cut points fixed on training out-of-fold predictions of the selected model
oof = cross_val_predict(best, Xtr, ytr, cv=cv, method='predict_proba')[:, 1]
c1, c2 = np.quantile(oof, [1 / 3, 2 / 3])
seg = np.where(ps < c1, 'Lower', np.where(ps < c2, 'Moderate', 'Higher'))
base = yte.mean(); segs = {}
for s_ in ['Lower', 'Moderate', 'Higher']:
    mk = seg == s_; segs[s_] = dict(n=int(mk.sum()), events=int(yte[mk].sum()), rate=float(yte[mk].mean()), lift=float(yte[mk].mean() / base), mean_p=float(ps[mk].mean()))
res['RQ4']['segments'] = dict(cuts=[float(c1), float(c2)], base=float(base), seg=segs, ratio_hi_lo=segs['Higher']['rate'] / segs['Lower']['rate'])
# profile of segments on the test set
te = m4.iloc[ite].copy(); te['SEG'] = seg
prof = {}
for s_ in ['Lower', 'Moderate', 'Higher']:
    g = te[te.SEG == s_]
    prof[s_] = dict(age60=float((g.AGE4 == '60+').mean()), mobile=float(g.MOBILE_USE.mean()), p2p=float(g.P2P_USE.mean()), chat=float(g.CHAT_USE.mean()),
                    inc100=float((g.INCOME4 == '$100,000 or more').mean()), fp=float(g.FALSE_POS.mean()))
res['RQ4']['profile'] = prof
# fairness check: test AUC by age group and sex
fair = {}
for col, lv in [('AGE4', ['60+']), ('SEX', ['Female', 'Male'])]:
    for v in (lv if col == 'SEX' else ['60+', 'under 60']):
        mk = (te[col] == v).values if v != 'under 60' else (te[col] != '60+').values
        fair[v] = dict(n=int(mk.sum()), auc=float(roc_auc_score(yte[mk], ps[mk])), rate=float(yte[mk].mean()), flagged=float((seg[mk] == 'Higher').mean()))
res['RQ4']['fairness'] = fair
# permutation importance on the test set (selected model), grouped by original feature
pi = permutation_importance(best, Xte, yte, scoring='roc_auc', n_repeats=30, random_state=SEED)
imp = sorted(zip(X.columns, pi.importances_mean, pi.importances_std), key=lambda z: -z[1])
res['RQ4']['importance'] = [(a, float(b), float(c)) for a, b, c in imp]
print('selected', sel, 'segments', segs, 'fair', fair); print('importance', imp[:8])

# figures: ROC + calibration
fig, axs = plt.subplots(1, 2, figsize=(7.6, 3.5))
sty = {'Logistic regression (L2)': (DARK, '-'), 'Random Forest': (MID, '--'), 'Gradient Boosting': (ACC, '-.'), 'Baseline (majority class)': ('#999', ':')}
for k, (c, ls) in sty.items():
    fpr, tpr, _ = roc_curve(yte, curves[k]); axs[0].plot(fpr, tpr, color=c, ls=ls, lw=1.6, label=f"{k.replace(' (majority class)', '').replace(' (L2)', '')} ({R4[k]['auc']:.3f})")
axs[0].set_xlabel('False positive rate (1 − specificity)'); axs[0].set_ylabel('True positive rate (recall)'); axs[0].legend(frameon=False, fontsize=7.5, title='Model (test AUC)', title_fontsize=8, loc='lower right')
for k, (c, ls) in sty.items():
    if k.startswith('Baseline'): continue
    fp_, mp_ = calibration_curve(yte, curves[k], n_bins=8, strategy='quantile'); axs[1].plot(mp_, fp_, marker='o', ms=3.5, color=c, ls=ls, lw=1.3, label=k.replace(' (L2)', ''))
axs[1].plot([0, .7], [0, .7], color='#999', ls=':', lw=1); axs[1].set_xlim(0, .7); axs[1].set_ylim(0, .7)
axs[1].set_xlabel('Mean predicted probability'); axs[1].set_ylabel('Observed fraud rate'); axs[1].legend(frameon=False, fontsize=7.5)
axs[0].set_title('A. ROC curves', fontsize=10, loc='left'); axs[1].set_title('B. Calibration', fontsize=10, loc='left')
fig.tight_layout(); fig.savefig(os.path.join(OUT, 'fig_roc.png'), dpi=200); plt.close(fig)
# segments + importance
FL = {'MOBILE_USE': 'Mobile banking', 'WEB_USE': 'Online banking', 'P2P_USE': 'P2P payments', 'CHAT_USE': 'Automated chat', 'ALERT_USE': 'Account alerts',
      'COMPUTER_VF': 'Computer use (very frequent)', 'PHONE_VF': 'Smartphone use (very frequent)', 'TYPE3': 'Institution type', 'AGE4': 'Age group', 'SEX': 'Sex',
      'RACE4': 'Race and ethnicity', 'EDUC4': 'Education', 'INCOME4': 'Household income', 'URBAN3': 'Urbanicity'}
fig, axs = plt.subplots(1, 2, figsize=(7.6, 3.5), gridspec_kw={'width_ratios': [1, 1.35]})
ss = ['Lower', 'Moderate', 'Higher']; rt = [segs[s_]['rate'] * 100 for s_ in ss]
axs[0].bar(ss, rt, color=[LIGHT, MID, DARK], width=.6); axs[0].axhline(base * 100, color='#555', ls='--', lw=.9)
axs[0].text(-.4, base * 100 + 1, f'overall {base * 100:.1f}%', ha='left', fontsize=8, color='#555')
for i, s_ in enumerate(ss): axs[0].text(i, rt[i] + 1, f"{rt[i]:.1f}%\n(lift {segs[s_]['lift']:.2f})", ha='center', fontsize=8)
axs[0].set_ylim(0, max(rt) * 1.35); axs[0].set_ylabel('Observed fraud rate, test set'); axs[0].set_xlabel('Risk segment'); axs[0].set_title('A. Risk segments', fontsize=10, loc='left')
top = imp[:10][::-1]
axs[1].barh([FL[a] for a, _, _ in top], [b for _, b, _ in top], xerr=[c for _, _, c in top], color=MID, ecolor='#555', capsize=2)
axs[1].set_xlabel('Drop in test ROC-AUC when shuffled'); axs[1].set_title('B. Permutation importance', fontsize=10, loc='left'); axs[1].tick_params(axis='y', labelsize=8.5)
fig.tight_layout(); fig.savefig(os.path.join(OUT, 'fig_segments.png'), dpi=200); plt.close(fig)
json.dump(res, open(os.path.join(OUT, 'models.json'), 'w'), indent=1, default=float)
