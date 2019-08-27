import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

# Full processed data
dat = pd.read_feather('data/full_gfw_10d_effort_model_data_8DAY_2012-01-01_2016-12-26.feather')

#dat = dat[dat.geartype == 'drifting_longlines']
#dat = dat.sort_values('date')

# ~50% of obs are zero (remove?)
len(dat[dat.fishing_hours > 0])/len(dat)
dat = dat[dat.fishing_hours > 0]

# Get data frame of variables and dummy seascapes
moddat = dat[['fishing_hours', 'month', 'seascape_class', 'sst', 'chlor_a']].dropna().reset_index(drop=True)

# Square sst
moddat['sst_sq'] = moddat['sst']**2
moddat = moddat[['fishing_hours', 'month', 'seascape_class', 'sst', 'sst_sq', 'chlor_a']]

seascape_dummies = pd.get_dummies(moddat['seascape_class']).reset_index(drop=True)
month_dummies = pd.get_dummies(moddat['month']).reset_index(drop=True)
moddat = pd.concat([moddat, month_dummies], axis=1)

# Get X, y
y = moddat['fishing_hours'].reset_index(drop=True)
y = np.log(1 + y)
moddat = moddat.drop(columns = ['month', 'fishing_hours', 'seascape_class'])
X = moddat
X.head()
y.head()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

clf = RandomForestRegressor()
clf.fit(X_train, y_train)

# 10-Fold Cross validation
print( np.mean(cross_val_score(clf, X_test, y_test, cv=10)))

