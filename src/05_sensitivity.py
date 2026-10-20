"""Interim report, part 3: RQ4 sensitivity check without ALERT_USE (possible reverse causation).
usage: python src/05_sensitivity.py  (optional args: <raw.csv> <outdir>)"""
import sys, os, json, numpy as np, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prep import load
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, 'data', 'raw', 'cfpb_nafbs-puf-labeled_2026-02.csv')
OUTD = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, 'outputs'); os.makedirs(OUTD, exist_ok=True)
d = load(RAW); SEED = 640
d['TYPE3'] = d.TYPE.where(d.TYPE.isin(['Bank', 'Credit Union', 'Online-only bank']))
d['COMPUTER_VF'] = np.where(d.DEVICE1.isna() | (d.DEVICE1 == 'SKIPPED ON WEB'), np.nan, (d.DEVICE1 == 'Very frequently').astype(float))
d['PHONE_VF'] = np.where(d.DEVICE2.isna() | (d.DEVICE2 == 'SKIPPED ON WEB'), np.nan, (d.DEVICE2 == 'Very frequently').astype(float))
m = d.dropna(subset=['FRAUD_EXP']).reset_index(drop=True)
out = {'alert_rate': {str(k): float(v) for k, v in m.groupby('ALERT_USE').FRAUD_EXP.mean().items()}}
BIN = ['MOBILE_USE', 'WEB_USE', 'P2P_USE', 'CHAT_USE', 'COMPUTER_VF', 'PHONE_VF']; CAT = ['TYPE3', 'AGE4', 'SEX', 'RACE4', 'EDUC4', 'INCOME4', 'URBAN3']
X = m[BIN + CAT].copy(); y = m.FRAUD_EXP.astype(int).values
for c in CAT: X[c] = X[c].astype(object)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, stratify=y, random_state=SEED)
pre = ColumnTransformer([('b', SimpleImputer(strategy='most_frequent'), BIN), ('c', Pipeline([('i', SimpleImputer(strategy='most_frequent')), ('o', OneHotEncoder(drop='first', handle_unknown='ignore'))]), CAT)])
cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
for name, clf, grid in [('Logistic regression (L2)', LogisticRegression(max_iter=2000, solver='liblinear'), {'clf__C': [.01, .03, .1, .3, 1, 3]}),
                        ('Random Forest', RandomForestClassifier(n_estimators=400, random_state=SEED, n_jobs=-1), {'clf__max_depth': [3, 5, 8], 'clf__min_samples_leaf': [10, 25, 50], 'clf__max_features': ['sqrt', .5]})]:
    gs = GridSearchCV(Pipeline([('pre', pre), ('clf', clf)]), grid, scoring='roc_auc', cv=cv, n_jobs=-1).fit(Xtr, ytr)
    out[name] = dict(cv_auc=float(gs.best_score_), auc=float(roc_auc_score(yte, gs.predict_proba(Xte)[:, 1])))
print(out); json.dump(out, open(os.path.join(OUTD, 'sensitivity.json'), 'w'), indent=1)
