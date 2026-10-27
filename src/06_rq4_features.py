"""Mentor feedback 1: do more input variables improve RQ4?
Tests the original 14 features against expanded sets of survey items that are (a) asked of every respondent,
(b) not about fraud, and (c) not plausibly a reaction to fraud. Fraud-alert items, trust, fraud-contact preferences,
and health/disability items are excluded (leakage risk or sensitive). Same leakage-safe design as 04_models.py:
same 80/20 split, imputation and encoding inside folds, 5-fold CV tuning, single test evaluation.
usage: python src/06_rq4_features.py  (optional args: <raw.csv> <outdir>)
"""
import sys, os, json, warnings
import numpy as np, pandas as pd
from math import log, ceil
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prep import load
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, 'data', 'raw', 'cfpb_nafbs-puf-labeled_2026-02.csv')
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, 'outputs'); os.makedirs(OUT, exist_ok=True)
SEED = 640
d = load(RAW)
SPECIAL = ['SKIPPED ON WEB', "DON'T KNOW", 'REFUSED', "I'm not sure", 'Unknown', "I don't know", 'I don’t know', 'I don’t know.', "Don't know", 'Prefer not to say', '98', '']
def clean(s):
    s = s.astype(str).str.strip(); return s.where(~s.isin(SPECIAL + ['nan']))
YN = lambda c: clean(d[c]).map({'Yes': 1.0, 'No': 0.0})

# ---------------------------------------------------------------- original features (as in 04_models.py)
d['TYPE3'] = d.TYPE.where(d.TYPE.isin(['Bank', 'Credit Union', 'Online-only bank']))
d['COMPUTER_VF'] = np.where(d.DEVICE1.isna() | (d.DEVICE1 == 'SKIPPED ON WEB'), np.nan, (d.DEVICE1 == 'Very frequently').astype(float))
d['PHONE_VF'] = np.where(d.DEVICE2.isna() | (d.DEVICE2 == 'SKIPPED ON WEB'), np.nan, (d.DEVICE2 == 'Very frequently').astype(float))
BASE_BIN = ['MOBILE_USE', 'WEB_USE', 'P2P_USE', 'CHAT_USE', 'ALERT_USE', 'COMPUTER_VF', 'PHONE_VF']
BASE_CAT = ['TYPE3', 'AGE4', 'SEX', 'RACE4', 'EDUC4', 'INCOME4', 'URBAN3']

# ---------------------------------------------------------------- candidate additional features
# Group A: finer demographics and household
for c in ['AGE7', 'INCOME9', 'EDUC5', 'MARITAL', 'EMPLOY', 'HHSIZE', 'HOUSING', 'REGION4', 'PHONESERVICE']: d[c + '_c'] = clean(d[c])
for c in ['CHILDREN', 'LANGOTHR', 'VETERAN']: d[c + '_b'] = YN(c)
d['METRO_b'] = (d.METRO == 'Metro Area').astype(float); d['INTERNET_b'] = (d.INTERNET == 'Internet Household').astype(float)
A_CAT = ['AGE7_c', 'INCOME9_c', 'EDUC5_c', 'MARITAL_c', 'EMPLOY_c', 'HHSIZE_c', 'HOUSING_c', 'REGION4_c', 'PHONESERVICE_c']
A_BIN = ['CHILDREN_b', 'LANGOTHR_b', 'VETERAN_b', 'METRO_b', 'INTERNET_b']
# Group B: banking relationship, products and channels
d['LENGTH_c'] = clean(d.LENGTH); d['BIGBANK_b'] = YN('BIGBANK'); d['OVER_c'] = clean(d.OVER); d['DEVICE3_c'] = clean(d.DEVICE3)
for c in ['ATM', 'TRAIN1']: d[c + '_b'] = YN(c)
for c in ['BRANCH', 'WIN']: d[c + '_b'] = YN(c)            # not asked of online-only bank customers -> missing, imputed
d['PHONE_REP_b'] = YN('PHONE1_1'); d['PHONE_AUTO_b'] = YN('PHONE1_2')
for i in [1, 2, 3, 4]: d[f'ACC2_{i}_b'] = YN(f'ACC2_{i}')     # paper statements, larger font, ATM video teller, passenger-side ATM
for i in [3, 4, 6]: d[f'FEAT1_{i}_b'] = YN(f'FEAT1_{i}')     # bill-pay reminders, online/automatic bill pay, overdraft protection
for i in range(1, 7): d[f'ASSET_{i}_b'] = YN(f'ASSETS_{i}')
for i in range(1, 6): d[f'DEBT_{i}_b'] = YN(f'DEBT2_{i}')
d['ASSETS_n'] = pd.to_numeric(d.ASSETS_COUNT, errors='coerce'); d['DEBT_n'] = pd.to_numeric(d.DEBT2_COUNT, errors='coerce')
B_CAT = ['LENGTH_c', 'OVER_c', 'DEVICE3_c']
B_BIN = ['BIGBANK_b', 'ATM_b', 'TRAIN1_b', 'BRANCH_b', 'WIN_b', 'PHONE_REP_b', 'PHONE_AUTO_b', 'ACC2_1_b', 'ACC2_2_b', 'ACC2_3_b', 'ACC2_4_b',
         'FEAT1_3_b', 'FEAT1_4_b', 'FEAT1_6_b'] + [f'ASSET_{i}_b' for i in range(1, 7)] + [f'DEBT_{i}_b' for i in range(1, 6)]
B_NUM = ['ASSETS_n', 'DEBT_n']
# Group C: financial situation, life events (non-health), and support arrangements
d['FWB_n'] = pd.to_numeric(d.FWB, errors='coerce'); d['SKILL_c'] = clean(d.SKILL); d['CREDREC_c'] = clean(d.CREDREC)
for c in ['HELP', 'CG', 'CONT']: d[c + '_b'] = YN(c)
d['POA1_b'] = YN('POA1'); d['CONT3_b'] = YN('CONT3')
SH = {1: 'widowed', 2: 'moved', 3: 'disaster_loss', 8: 'became_caregiver', 9: 'fully_retired', 10: 'part_retired', 11: 'reduced_hours', 12: 'reduced_income'}
for i, nm in SH.items(): d[f'SHOCK_{nm}_b'] = YN(f'SHOCK1_{i}')
C_CAT = ['SKILL_c', 'CREDREC_c']
C_BIN = ['HELP_b', 'CG_b', 'CONT_b', 'POA1_b', 'CONT3_b'] + [f'SHOCK_{nm}_b' for nm in SH.values()]
C_NUM = ['FWB_n']

SETS = {
    'S0: original 14 features': (BASE_BIN, BASE_CAT, []),
    'S1: + finer demographics and household': (BASE_BIN + A_BIN, BASE_CAT + A_CAT, []),
    'S2: + banking relationship and channels': (BASE_BIN + A_BIN + B_BIN, BASE_CAT + A_CAT + B_CAT, B_NUM),
    'S3: + financial situation, life events, support (all)': (BASE_BIN + A_BIN + B_BIN + C_BIN, BASE_CAT + A_CAT + B_CAT + C_CAT, B_NUM + C_NUM),
}
# strict version: drop every item that could plausibly be a reaction to a past fraud (reverse causation)
SUSPECT = ['ALERT_USE', 'PHONE_REP_b', 'PHONE_AUTO_b', 'TRAIN1_b', 'CREDREC_c', 'FWB_n', 'SHOCK_reduced_income_b']
b3, c3, n3 = SETS['S3: + financial situation, life events, support (all)']
SETS['S4: all, excluding possible reactions to fraud (strict)'] = ([x for x in b3 if x not in SUSPECT], [x for x in c3 if x not in SUSPECT], [x for x in n3 if x not in SUSPECT])

m = d.dropna(subset=['FRAUD_EXP']).reset_index(drop=True); y = m.FRAUD_EXP.astype(int).values
idx_tr, idx_te = train_test_split(np.arange(len(y)), test_size=.2, stratify=y, random_state=SEED)   # identical split to 04_models.py
ytr, yte = y[idx_tr], y[idx_te]
cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
rng = np.random.default_rng(SEED); BOOT = [rng.integers(0, len(yte), len(yte)) for _ in range(1000)]
def boot(p): v = [roc_auc_score(yte[b], p[b]) for b in BOOT]; return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))

def pre(BIN, CAT, NUM):
    t = [('bin', SimpleImputer(strategy='most_frequent'), BIN),
         ('cat', Pipeline([('imp', SimpleImputer(strategy='most_frequent')), ('oh', OneHotEncoder(drop='first', handle_unknown='ignore'))]), CAT)]
    if NUM: t.append(('num', Pipeline([('imp', SimpleImputer(strategy='median')), ('sc', StandardScaler())]), NUM))
    return ColumnTransformer(t)
MODELS = {
    'Logistic regression (L2)': (LogisticRegression(max_iter=4000, solver='liblinear'), {'clf__C': [.003, .01, .03, .1, .3, 1]}),
    'Logistic regression (L1, selects features)': (LogisticRegression(max_iter=4000, solver='liblinear', penalty='l1'), {'clf__C': [.01, .03, .1, .3, 1]}),
    'Random Forest': (RandomForestClassifier(n_estimators=400, random_state=SEED, n_jobs=-1),
                      {'clf__max_depth': [3, 5, 8], 'clf__min_samples_leaf': [10, 25, 50], 'clf__max_features': ['sqrt', .3]}),
    'Gradient Boosting': (GradientBoostingClassifier(random_state=SEED),
                          {'clf__n_estimators': [100, 200], 'clf__learning_rate': [.03, .1], 'clf__max_depth': [2, 3], 'clf__subsample': [.8]}),
}
res, preds = {}, {}
for sname, (BIN, CAT, NUM) in SETS.items():
    X = m[BIN + CAT + NUM].copy()
    for c in CAT: X[c] = X[c].astype(object)
    Xtr, Xte = X.iloc[idx_tr], X.iloc[idx_te]
    npar = pre(BIN, CAT, NUM).fit_transform(Xtr).shape[1]
    for mname, (clf, grid) in MODELS.items():
        gs = GridSearchCV(Pipeline([('pre', pre(BIN, CAT, NUM)), ('clf', clf)]), grid, scoring='roc_auc', cv=cv, n_jobs=-1).fit(Xtr, ytr)
        p = gs.predict_proba(Xte)[:, 1]; lo, hi = boot(p)
        r = dict(features=len(BIN + CAT + NUM), parameters=int(npar), cv_auc=float(gs.best_score_), test_auc=float(roc_auc_score(yte, p)), ci=(lo, hi),
                 pr_auc=float(average_precision_score(yte, p)), train_auc=float(roc_auc_score(ytr, gs.predict_proba(Xtr)[:, 1])),
                 params={k.replace('clf__', ''): v for k, v in gs.best_params_.items()})
        if 'L1' in mname:
            coef = gs.best_estimator_.named_steps['clf'].coef_[0]; r['nonzero'] = int((coef != 0).sum())
            names = gs.best_estimator_.named_steps['pre'].get_feature_names_out()
            r['top'] = [(str(n), float(c)) for n, c in sorted(zip(names, coef), key=lambda z: -abs(z[1]))[:15] if c != 0]
        res.setdefault(sname, {})[mname] = r; preds[(sname, mname)] = p
        print(f"{sname[:3]} {mname:44s} feat={r['features']:3d} par={npar:3d} cv={r['cv_auc']:.3f} test={r['test_auc']:.3f} [{lo:.3f},{hi:.3f}] pr={r['pr_auc']:.3f} train={r['train_auc']:.3f}", r.get('nonzero', ''))
# paired bootstrap: best expanded vs original, same model
for mname in MODELS:
    a = preds[('S0: original 14 features', mname)]
    for sname in list(SETS)[1:]:
        b = preds[(sname, mname)]; dif = [roc_auc_score(yte[i], b[i]) - roc_auc_score(yte[i], a[i]) for i in BOOT]
        res[sname][mname]['d_vs_S0'] = (float(np.mean(dif)), float(np.percentile(dif, 2.5)), float(np.percentile(dif, 97.5)))
# Riley criterion 1 for the number of parameters actually used
phi = y.mean(); lnL0 = phi * log(phi) + (1 - phi) * log(1 - phi); R2 = .15 * (1 - np.exp(2 * lnL0))
riley = {s: ceil(res[s]['Logistic regression (L2)']['parameters'] / ((.9 - 1) * log(1 - R2 / .9))) for s in SETS}
out = dict(results=res, riley_N=riley, n_train=int(len(ytr)), n_test=int(len(yte)))
json.dump(out, open(os.path.join(OUT, 'rq4_features.json'), 'w'), indent=1, default=float)
print('Riley N required by parameter count:', riley)
for s in list(SETS)[1:]:
    for mn in MODELS: print(s[:3], mn, 'diff vs S0', [round(v, 3) for v in res[s][mn]['d_vs_S0']])
print('L1 top (S3):', res[list(SETS)[3]]['Logistic regression (L1, selects features)'].get('top'))

# ---------------------------------------------------------------- precision / recall when flagging the top k% (test set)
def topk(p, frac):
    t = np.quantile(p, 1 - frac); yh = (p >= t).astype(int); TP = int(((yh == 1) & (yte == 1)).sum()); FP = int(((yh == 1) & (yte == 0)).sum())
    return dict(flagged=int(yh.sum()), TP=TP, FP=FP, precision=TP / max(yh.sum(), 1), recall=TP / int(yte.sum()), lift=TP / max(yh.sum(), 1) / yte.mean())
cmp_models = [('S0: original 14 features', 'Random Forest'), ('S0: original 14 features', 'Logistic regression (L2)'),
              ('S4: all, excluding possible reactions to fraud (strict)', 'Logistic regression (L1, selects features)'),
              ('S3: + financial situation, life events, support (all)', 'Logistic regression (L1, selects features)')]
topk_res = {f'{s[:2]} {mn}': {f'top {int(f*100)}%': topk(preds[(s, mn)], f) for f in [.1, .2, .33]} for s, mn in cmp_models}
out['topk'] = topk_res
s4 = 'S4: all, excluding possible reactions to fraud (strict)'
out['s4_l1_top'] = res[s4]['Logistic regression (L1, selects features)'].get('top')
json.dump(out, open(os.path.join(OUT, 'rq4_features.json'), 'w'), indent=1, default=float)
for k, v in topk_res.items():
    print(k); print(pd.DataFrame(v).T.round(3).to_string())
print('S4 L1 top:', out['s4_l1_top'])
