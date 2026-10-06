"""Reproduces the sample-size calculations in the synopsis (Tables 4, 5 and A1).

RQ1-RQ3: power analysis for logistic regression with a binary covariate
         (Hsieh, Bloch & Larsen, 1998), with variance inflation adjustment.
RQ4:     minimum sample size for a prediction model with a binary outcome
         (Riley et al., 2019), with an events-per-variable cross-check.

Run from the repository root:
    python src/01_sample_size.py
Results are printed and saved to outputs/sample_size_results.json.
"""
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, 'data', 'raw', 'cfpb_nafbs-puf-labeled_2026-02.csv')
OUTJ = os.path.join(ROOT, 'outputs', 'sample_size_results.json')
import sys, json, numpy as np, pandas as pd
from math import sqrt, ceil, log, exp
from scipy.stats import norm, chi2_contingency
from scipy.optimize import brentq
import statsmodels.formula.api as smf
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from prep import load, CONTROLS
d = load(DATA); w = d.WEIGHT_FINAL; out = {}
za, zb = norm.ppf(.975), norm.ppf(.80)
def hsieh(P1, OR, B, rho2):
    o = P1/(1-P1); P2 = OR*o/(1+OR*o); Pb = (1-B)*P1 + B*P2
    A = za*sqrt(Pb*(1-Pb)/B); C = zb*sqrt(P1*(1-P1) + P2*(1-P2)*(1-B)/B)
    den = (P1-P2)**2*(1-B); n0 = (A+C)**2/den
    return dict(P1=P1, P2=P2, B=B, Pbar=Pb, termA=A, termB=C, denom=den, n_unadj=n0, rho2=rho2, n_adj=n0/(1-rho2), N=ceil(n0/(1-rho2)))
def rho2(x, sub, extra=''):
    r = smf.ols(f"{x} ~ {CONTROLS}{extra}", data=sub).fit(); return r.rsquared
def focal(y, x, sub, OR, extra=''):
    t = sub[[y, x]].dropna(); a = t[t[x]==1][y]; b = t[t[x]==0][y]
    r2 = rho2(x, sub.dropna(subset=[y, x]), extra)
    r2r = ceil(r2*20)/20  # round up to next .05
    h_up = hsieh(b.mean(), OR, len(a)/len(t), r2r); h_dn = hsieh(b.mean(), 1/OR, len(a)/len(t), r2r)
    h = h_up if h_up['n_adj'] >= h_dn['n_adj'] else h_dn
    h['direction'] = 'OR' if h is h_up else '1/OR'; h['N_up'] = h_up['N']; h['N_dn'] = h_dn['N']
    tab = pd.crosstab(t[x], t[y]); chi, p, _, _ = chi2_contingency(tab, correction=False)
    h.update(n=len(t), n_x1=len(a), n_x0=len(b), obs_p1=a.mean(), rho2_raw=r2, OR_target=OR,
             obs_OR=(a.mean()/(1-a.mean()))/(b.mean()/(1-b.mean())), chi2=chi, p=p, tab=tab.values.tolist())
    return h
# ---------- outcomes
oc = {}
for y in ['FRAUD_EXP','FIN_LOSS','UNREC_LOSS','FALSE_POS']:
    s = d[y].dropna(); oc[y] = dict(n=int(len(s)), events=int(s.sum()), unw=float(s.mean()), wtd=float(np.average(s, weights=w[s.index])))
out['outcomes'] = oc
out['raw'] = {c: d[c].fillna('(not asked)').value_counts().to_dict() for c in ['FRAUD1','FRAUD2','FRAUD8','MOBILE','WEB','TYPE','BIGBANK','AGE4','INCOME4','EDUC5','SEX','RACETHNICITY','TRUST1']}
out['deff'] = dict(full=float(1+(w.std(ddof=0)/w.mean())**2), neff=float(w.sum()**2/(w**2).sum()))
fr = d[d.FRAUD_EXP==1]; out['deff']['neff_fraud'] = float(fr.WEIGHT_FINAL.sum()**2/(fr.WEIGHT_FINAL**2).sum())
# ---------- RQ1
rq1 = d.dropna(subset=['FRAUD_EXP'])
out['RQ1'] = {x: focal('FRAUD_EXP', x, rq1, 1.5) for x in ['MOBILE_USE','WEB_USE','P2P_USE','CHAT_USE']}
# ---------- RQ2
rq2 = d.dropna(subset=['FIN_LOSS'])
out['RQ2'] = {x: focal('FIN_LOSS', x, rq2, 1.75) for x in ['DEBIT_INVOLVED','BANK_DETECTED','REPEAT_VICTIM','PHISHING','ONLINE_SHOP_SCAM','IMPOSTER','MOBILE_USE']}
for x in out['RQ2']:
    h = out['RQ2'][x]; f = lambda OR: max(hsieh(h['P1'], OR, h['B'], h['rho2'])['n_adj'], hsieh(h['P1'], 1/OR, h['B'], h['rho2'])['n_adj']) - 642
    h['MDE_642'] = brentq(f, 1.05, 6); h['N_at_1.5'] = max(hsieh(h['P1'], 1.5, h['B'], h['rho2'])['N'], hsieh(h['P1'], 1/1.5, h['B'], h['rho2'])['N'])
# ---------- RQ3
rq3 = d.dropna(subset=['FALSE_POS','FRAUD_EXP'])
out['RQ3'] = {x: focal('FALSE_POS', x, rq3, 1.5, ' + FRAUD_EXP') for x in ['MOBILE_USE','WEB_USE','P2P_USE','CHAT_USE','CREDIT_UNION','BIG_BANK']}
out['RQ3_n'] = dict(n_fp=int(d.FALSE_POS.notna().sum()), n_both=int(len(rq3)), events_both=int(rq3.FALSE_POS.sum()))
# ---------- RQ4 (Riley et al., 2019)
phi = oc['FRAUD_EXP']['unw']; lnL0 = phi*log(phi) + (1-phi)*log(1-phi); maxR2 = 1-exp(2*lnL0); R2 = 0.15*maxR2; S = 0.9; P = 24
c1 = P/((S-1)*log(1-R2/S)); S2 = R2/(R2+0.05*maxR2); c2 = P/((S2-1)*log(1-R2/S2)); c3 = (1.96/0.05)**2*phi*(1-phi)
out['RQ4'] = dict(phi=phi, lnL0=lnL0, maxR2=maxR2, R2cs=R2, S=S, P=P, c1=c1, S2=S2, c2=c2, c3=c3, N=ceil(max(c1,c2,c3)),
                  epv_N=ceil(10*P/phi), train_n=int(round(0.8*oc['FRAUD_EXP']['n'])), ln_term=log(1-R2/S))
# ---------- descriptives for data section
def dist(c, sub=None):
    t = (d if sub is None else sub); s = t[c].dropna(); ww = t.loc[s.index, 'WEIGHT_FINAL']
    return {k: dict(n=int((s==k).sum()), unw=float((s==k).mean()), wtd=float(ww[s==k].sum()/ww.sum())) for k in s.unique()}
out['desc'] = {c: dist(c) for c in ['AGE4','SEX','RACE4','EDUC4','INCOME4','TYPE','MOBILE','WEB','ACC2_6','ACC2_5','ALERT_1']}
out['mob_by_age'] = d.groupby('AGE4').MOBILE_USE.mean().round(4).to_dict()
# trust supporting evidence
t = d.dropna(subset=['TRUST_HIGH','FALSE_POS']); out['trust'] = dict(n=int(len(t)), fp0=float(t[t.FALSE_POS==0].TRUST_HIGH.mean()), fp1=float(t[t.FALSE_POS==1].TRUST_HIGH.mean()))
# missingness / validity for every retained variable
out['fraud_sub'] = dict(n=int(len(fr)), debit=int(fr.DEBIT_INVOLVED.sum()), bankdet=int(fr.BANK_DETECTED.sum()), repeat=int(fr.REPEAT_VICTIM.sum()))
feats = ['MOBILE_USE','WEB_USE','P2P_USE','CHAT_USE','ALERT_USE','TYPE','AGE4','SEX','RACE4','EDUC4','INCOME4','DEVICE1','DEVICE2','URBAN3']
bad = ['SKIPPED ON WEB',"DON'T KNOW",'REFUSED',"I'm not sure",'Unknown']
m = rq1[feats].copy()
for c in feats:
    if m[c].dtype == object: m.loc[m[c].isin(bad), c] = np.nan
out['RQ4_missing'] = dict(n=int(len(m)), complete=int(m.dropna().shape[0]), per_item={c:int(m[c].isna().sum()) for c in feats})
print('RQ4 missing', out['RQ4_missing'])
json.dump(out, open(OUTJ, 'w'), indent=1, default=float)
# ---------- print
for rq in ['RQ1','RQ2','RQ3']:
    print(rq)
    for x, h in out[rq].items():
        print(f"  {x:17s} n={h['n']} B={h['B']:.3f} P1={h['P1']:.3f} P2={h['P2']:.3f} Pbar={h['Pbar']:.4f} A={h['termA']:.4f} C={h['termB']:.4f} den={h['denom']:.6f} n0={h['n_unadj']:.1f} rho2={h['rho2_raw']:.3f}->{h['rho2']:.2f} N={h['N']} dir={h['direction']} up={h['N_up']} dn={h['N_dn']} | obsOR={h['obs_OR']:.2f} p={h['p']:.4f}" + (f" MDE@642={h['MDE_642']:.2f} N@1.5={h['N_at_1.5']}" if rq=='RQ2' else ''))
print('RQ4', {k:(round(v,4) if isinstance(v,float) else v) for k,v in out['RQ4'].items()})
print(out['outcomes']); print(out['deff']); print(out['RQ3_n']); print(out['trust']); print(out['fraud_sub']); print(out['mob_by_age'])
