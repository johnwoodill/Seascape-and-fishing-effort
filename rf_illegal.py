import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import glob
from sklearn.preprocessing import MinMaxScaler
from statsmodels.discrete.discrete_model import Logit
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics as metrics

# files = glob.glob('data/processed/*.feather')
# files
# list_ = []
# for file in files:
#     df = pd.read_feather(file)
#     list_.append(df)
#     mdat = pd.concat(list_, sort=False)


# mdat = mdat.reset_index(drop=True)

# dat = mdat

# Full processed data
dat = pd.read_feather('data/full_gfw_10d_effort_model_data_8DAY_2012-01-01_2016-12-26.feather')

dat = dat[dat.geartype == 'drifting_longlines']
dat = dat.sort_values('date')

# ~50% of obs are zero (remove?)
len(dat[dat.fishing_hours > 0])/len(dat)
dat = dat[dat.fishing_hours > 0]

# If illegally operating inside EEZ (!= ARG)
dat.loc[:, 'illegal'] = np.where(((dat['eez'] == True) & (dat['flag'] != 'ARG')), 1, 0)

# Convert true/false eez to 0/1
dat.loc[:, 'eez'] = dat.eez.astype('uint8')

sum(dat.illegal)/len(dat)

# Remove 2015 because incomplete (still processing)
#dat = dat[dat.year <= 2014]

# Linear model

# Get data frame of variables and dummy seascapes
moddat = dat[['illegal', 'fishing_hours', 'flag', 'year', 'month', 'seascape_class', 'sst', 'sst_grad', 'sst4', 'sst4_grad', 'chlor_a', 'lon1', 'lat1', 'tonnage', 'depth_m', 'coast_dist_km', 'port_dist_km', 'eez']].dropna().reset_index(drop=True)

moddat = dat[['illegal', 'fishing_hours', 'year', 'sst', 'sst_grad', 'sst4', 'sst4_grad', 'chlor_a', 'lon1', 'lat1', 'tonnage', 'depth_m', 'coast_dist_km', 'port_dist_km', 'eez']].dropna().reset_index(drop=True)

# Square sst
moddat['sst_sq'] = moddat['sst']**2

seascape_dummies = pd.get_dummies(moddat['seascape_class']).reset_index(drop=True)
month_dummies = pd.get_dummies(moddat['month']).reset_index(drop=True)
flag_dummies = pd.get_dummies(moddat['flag']).reset_index(drop=True)
moddat = pd.concat([moddat, month_dummies, flag_dummies], axis=1)


# Get X, y
y = moddat[['year', 'illegal']].reset_index(drop=True)

moddat = moddat.drop(columns = ['month', 'illegal', 'seascape_class', 'flag'])

moddat = moddat.drop(columns = ['illegal'])

X = moddat
X.columns
X.head()
y.head()


pred_score = []
roc_dat = pd.DataFrame()
for year in range(2013, 2017):
    X_train = moddat[moddat.year < year]
    y_train = y[y.year < year]
    X_test = moddat[moddat.year == year]
    y_test = y[y.year == year]

    y_train = y_train['illegal']
    y_test = y_test['illegal']
    
    # clf = LogisticRegression().fit(X_train, y_train)
    clf = RandomForestClassifier().fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    pred_score.append(score)

    probs = clf.predict_proba(X_test)
    preds = probs[:,1]
    fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
    roc_auc = metrics.auc(fpr, tpr)

    outdat = pd.DataFrame({'pred_year': year, 'fpr': fpr, 'tpr': tpr, 'thresh': threshold})
    roc_dat = pd.concat([roc_dat, outdat])

print(pred_score)

roc_dat = roc_dat.reset_index(drop=True)
roc_dat.to_feather('data/roc_auc_rf_illegal.feather')


probs = clf.predict_proba(X_test)
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
roc_auc = metrics.auc(fpr, tpr)

outdat = pd.DataFrame({'fpr': fpr, 'tpr': tpr, 'thresh': threshold})
outdat.to_feather('data/roc_auc_rf_illegal.feather')
roc_auc

# method I: plt
import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


clf = RandomForestClassifier().fit(X, y)
fea_import = pd.DataFrame({'variable': X.columns , 'importance': clf.feature_importances_})
fea_import.sort_values('importance', ascending=False)
fea_import.to_feather('data/feature_importance_rf_illegal.feather')
sns.barplot(fea_import.variable, fea_import.importance)
