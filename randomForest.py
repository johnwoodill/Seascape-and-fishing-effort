import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score, r2_score, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit, GridSearchCV

# Full processed data
dat = pd.read_feather('data/full_gfw_10d_effort_model_data_8DAY_2012-01-01_2016-12-26.feather')

#dat = dat[dat.geartype == 'drifting_longlines']
#dat = dat.sort_values('date')

# ~50% of obs are zero (remove?)
len(dat[dat.fishing_hours > 0])/len(dat)
dat = dat[dat.fishing_hours > 0]

# Get data frame of variables and dummy seascapes
moddat = dat[['fishing_hours', 'year', 'sst', 'sst_grad', 'sst4', 'sst4_grad', 'chlor_a', 'lon1', 'lat1', 'tonnage', 'seascape_class', 'month']].dropna().reset_index(drop=True)


# Square sst
#moddat['sst_sq'] = moddat['sst']**2
#moddat['sst4_sq'] = moddat['sst4']**2

seascape_dummies = pd.get_dummies(moddat['seascape_class'], dummy_na=True).reset_index(drop=True)
month_dummies = pd.get_dummies(moddat['month']).reset_index(drop=True)
moddat = pd.concat([moddat, month_dummies, seascape_dummies], axis=1)

# Get X, y
y = moddat[['fishing_hours', 'year']].reset_index(drop=True)
#y = np.log(1 + y)
moddat = moddat.drop(columns = ['month', 'fishing_hours', 'seascape_class'])
X = moddat
X.columns
X.head()
y.head()

X = X.set_index('year')
X

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]

# Number of features to consider at every split
max_features = ['auto', 'sqrt']

# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)

# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]

# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]

# Method of selecting samples for training each tree
bootstrap = [True, False]

# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
print(random_grid)

# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestRegressor()

# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
tscv = TimeSeriesSplit(n_splits=5).split(X)
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = tscv, verbose=2, random_state=42, n_jobs = -1)

# Fit the random search model
rf_random.fit(X, y.fishing_hours)

# /home/server/pi/homes/woodilla/.conda/envs/baseDS_env/lib/python3.7/site-packages/sklearn/base.py:420: FutureWarning: The default 
# value of multioutput (not exposed in score method) will change from 'variance_weighted' to 'uniform_average' in 0.23 to keep 
# consistent with 'metrics.r2_score'. To specify the default value manually and avoid the warning, please either call 'metrics.r2_score' 
# directly or make a custom scorer with 'metrics.make_scorer' (the built-in scorer 'r2' uses multioutput='uniform_average').
# "multioutput='uniform_average').", FutureWarning)

# Results
# RandomizedSearchCV(cv=<generator object TimeSeriesSplit.split at 0x7f31b8124f50>,
#                    error_score='raise-deprecating',
#                    estimator=RandomForestRegressor(bootstrap=True,
#                                                    criterion='mse',
#                                                    max_depth=None,
#                                                    max_features='auto',
#                                                    max_leaf_nodes=None,
#                                                    min_impurity_decrease=0.0,
#                                                    min_impurity_split=None,
#                                                    min_samples_leaf=1,
#                                                    min_samples_split=2,
#                                                    min_weight_fraction_leaf=0.0,
#                                                    n_est...
#                    param_distributions={'bootstrap': [True, False],
#                                         'max_depth': [10, 20, 30, 40, 50, 60,
#                                                       70, 80, 90, 100, 110,
#                                                       None],
#                                         'max_features': ['auto', 'sqrt'],
#                                         'min_samples_leaf': [1, 2, 4],
#                                         'min_samples_split': [2, 5, 10],
#                                         'n_estimators': [200, 400, 600, 800,
#                                                          1000, 1200, 1400, 1600,
#                                                          1800, 2000]},
#                    pre_dispatch='2*n_jobs', random_state=42, refit=True,
#                    return_train_score=False, scoring=None, verbose=2)

# Get best parameters from grid search
rf_random.best_params_

# {'n_estimators': 200, 'min_samples_split': 5, 'min_samples_leaf': 4, 'max_features': 'auto', 'max_depth': 10, 'bootstrap': True}


# Create the parameter grid based on the results of random search 
param_grid = {
    'bootstrap': [True],
    'max_depth': [80, 90, 100, 110],
    'max_features': [2, 3],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [100, 200, 300, 1000]
}

# Create a based model
rf = RandomForestRegressor()
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = tscv, n_jobs = -1, verbose = 2)

# Fit the grid search to the data
grid_search.fit(X, y)

# Get best parameters
grid_search.best_params_


mod = RandomForestRegressor(n_estimators=200, min_samples_split=5, min_samples_leaf=4, max_features='auto', max_depth=10, bootstrap=True)
cv_results = cross_validate(mod, X, y.fishing_hours, cv=tscv)
print(cv_results)

mod.fit(X_train, y_train)
mod.feature_importances_

y_pred = mod.predict(X_train)

explained_variance_score(y_train, y_pred)

mod.score(X_train, y_train)
mod.score(X_test, y_test)




tscv = TimeSeriesSplit(max_train_size=None, n_splits=5)
for train_index, test_index in tscv.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]


# Rolling-year Cross validate
pred_score = pd.DataFrame()
for year in range(2013, 2018):
    X_train = moddat[moddat.year < year]
    y_train = y[y.year < year]
    X_test = moddat[moddat.year == year]
    y_test = y[y.year == year]

    y_train = y_train['fishing_hours']
    y_test = y_test['fishing_hours']
    
    # Random Forest
    mod = RandomForestRegressor().fit(X_train,y_train)

    mod.feature_importances_

    y_train_pred = mod.predict(X_train)
    evar = explained_variance_score(y_train, y_train_pred)
    evar
        
    y_test_pred = mod.predict(X_test)
    
    train_mse = mean_squared_error(y_train, y_train_pred) 
    test_mse = mean_squared_error(y_test, y_test_pred) 

    train_mse; test_mse


    score = mod.score(y_test, y_test_pred)
    score
    score = accuracy_score(y_test, y_pred)
    pred_score.append(score)

    probs = clf.predict_proba(X_test)
    preds = probs[:,1]
    fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
    roc_auc = metrics.auc(fpr, tpr)

    outdat = pd.DataFrame({'pred_year': year, 'fpr': fpr, 'tpr': tpr, 'thresh': threshold})
    roc_dat = pd.concat([roc_dat, outdat])
    # OLS
    #mod = sm.OLS(y_train, X_train).fit()
    #y_train_pred = mod.predict(X_train)
    #y_test_pred = mod.predict(X_test)


    train_score = mean_squared_error(y_train, y_train_pred)
    test_score = mean_squared_error(y_test, y_test_pred)
    
    rdat = pd.DataFrame({'year': [year], 'train_score': [train_score], 'test_score': [test_score]})
    pred_score = pd.concat([pred_score, rdat])

print(pred_score)








fea_import = pd.DataFrame({'variable': X_train.columns , 'importance': mod.feature_importances_})
fea_import.sort_values('importance', ascending=False)

# 10-Fold Cross validation
#print( np.mean(cross_val_score(clf, X_test, y_test, cv=5)))

scores = cross_val_score(clf, X_test, y_test)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

X.columns
