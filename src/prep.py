"""Shared data preparation for the NAFBS capstone (derivations used everywhere)."""
import numpy as np, pandas as pd
YN = {'Yes': 1, 'No': 0}
TIMES = {'Never': 0, 'One time': 1, 'Two or three times': 1, 'More than three times': 1}
def load(path):
    d = pd.read_csv(path, low_memory=False)
    for c in d.columns:
        if d[c].dtype == object: d[c] = d[c].str.strip()
    # outcomes
    d['FRAUD_EXP'] = d.FRAUD1.map(TIMES)                      # "I'm not sure"/skipped -> NaN
    d['FALSE_POS'] = d.FRAUD8.map(TIMES)
    f2 = d.FRAUD2.fillna('')
    d['FIN_LOSS'] = np.where(f2.str.startswith('I lost money'), 1.0, np.where(f2.str.startswith('No money'), 0.0, np.nan))
    d['UNREC_LOSS'] = np.where(f2.str.contains('some of it|none of it'), 1.0,
                        np.where(f2.str.startswith('No money') | f2.str.contains('all of it'), 0.0, np.nan))
    # focal predictors
    d['MOBILE_USE'] = d.MOBILE.map(YN); d['WEB_USE'] = d.WEB.map(YN)
    d['P2P_USE'] = d.ACC2_6.map(YN); d['CHAT_USE'] = d.ACC2_5.map(YN); d['ALERT_USE'] = d.ALERT_1.map(YN)
    d['CREDIT_UNION'] = d.TYPE.map({'Credit Union': 1, 'Bank': 0, 'Online-only bank': 0})
    d['BIG_BANK'] = d.BIGBANK.map(YN)
    d['DEBIT_INVOLVED'] = d.FRAUD3_1.map(YN); d['BANK_DETECTED'] = d.FRAUD5_1.map(YN)
    d['REPEAT_VICTIM'] = d.FRAUD1.map({'One time': 0, 'Two or three times': 1, 'More than three times': 1})
    d['PHISHING'] = d.SCAM_3.map(YN); d['ONLINE_SHOP_SCAM'] = d.SCAM_7.map(YN); d['IMPOSTER'] = d.SCAM_2.map(YN)
    # controls
    d['AGE60'] = (d.AGE4 == '60+').astype(int)
    d['RACE4'] = d.RACETHNICITY.replace({'White, non-Hispanic': 'White NH', 'Black, non-Hispanic': 'Black NH', 'Hispanic': 'Hispanic'})
    d.loc[~d.RACE4.isin(['White NH', 'Black NH', 'Hispanic']), 'RACE4'] = 'Other NH'
    d['EDUC4'] = d.EDUC5.replace({'Less than HS': 'HS or less', 'HS graduate or equivalent': 'HS or less',
        'Some college/ associates degree': 'Some college', "Bachelor's degree": 'Bachelor', 'Post grad study/professional degree': 'Postgraduate'})
    ag = ['Strongly agree', 'Somewhat agree', 'Neither agree nor disagree', 'Somewhat disagree', 'Strongly disagree']
    d['TRUST_HIGH'] = np.where(d.TRUST1.isin(ag), (d.TRUST1 == 'Strongly agree').astype(float), np.nan)
    return d
CONTROLS = 'C(AGE4) + C(SEX) + C(RACE4) + C(EDUC4) + C(INCOME4)'
