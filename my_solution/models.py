#coding: utf-8
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
import lightgbm as lgb
import xgboost as xgb
import numpy as np
import pandas as pd

def display_importances(feature_importance_df_):
    cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False)[:40].index
    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]
    plt.figure(figsize=(8, 10))
    sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.show()

def lgb_model(train_df, test_df, X, X_submit):
	# k-cv
	N_FOLDS = 5
	y = train_df['信用分']
	kfold = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=2019)
	kf = kfold.split(X, y)

	# LightGBM: GBDT
	params = {
	    'learning_rate': 0.01,
	    'boosting_type': 'gbdt',
	    'objective': 'regression_l1',
	    'metric': 'mae',
	    'feature_fraction': 0.6,
	    'bagging_fraction': 0.8,
	    'bagging_freq': 2,
	    'num_leaves': 31,
	    'verbose': -1,
	    'max_depth': 5,
	    'lambda_l1': 0,
	    'lambda_l2': 5,
	    'nthread': 8
	}


	# process the k-cv
	cv_pred = np.zeros(test_df.shape[0])
	valid_best_l2_all = 0

	feature_importance_df = pd.DataFrame()
	count = 0
	for i, (train_idx, test_idx) in enumerate(kf):
	    print('fold: ',i, ' training')
	    X_train, X_test, y_train, y_test = X.iloc[train_idx, :], X.iloc[test_idx, :], y.iloc[train_idx], y.iloc[test_idx]
	#     X_train, X_test, y_train, y_test = X[train_idx, :], X[test_idx, :], y[train_idx], y[test_idx]
	    data_train = lgb.Dataset(X_train, y_train)
	    data_test = lgb.Dataset(X_test, y_test)
	    lgb_model = lgb.train(params, data_train, num_boost_round=10000, valid_sets=data_test, 
	                          verbose_eval=-1, early_stopping_rounds=50)
	    cv_pred += lgb_model.predict(X_submit, num_iteration=lgb_model.best_iteration)
	    valid_best_l2_all += lgb_model.best_score['valid_0']['l1']
	    
	#     fold_importance_df = pd.DataFrame()
	#     fold_importance_df["feature"] = list(unicode(X_train.columns))
	#     fold_importance_df["importance"] = lgb_model.feature_importance(importance_type='gain', iteration=lgb_model.best_iteration)
	#     fold_importance_df["fold"] = count + 1
	#     feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

	    count += 1
	    
	cv_pred /= N_FOLDS
	valid_best_l2_all /= N_FOLDS
	print('cv score for valid is: ', 1 / (1 + valid_best_l2_all))

	# show the importance of features
	# display_importances(feature_importance_df)

	return cv_pred, 1 / (1 + valid_best_l2_all)


def xgb_model():
	# XGBoost
	xgb_params={'eta': 0.005,
				'max_depth': 10,
				'subsample': 0.8,
				'colsample_bytree': 0.8,
				'objective': 'reg:linear',
				'eval_metric': 'mae',
				'silent': True,
				'nthread': 8}

	cv_pred_allxgb = 0
	en_amount = 3
	oof_xgb1 = np.zeros(len(train_data))
	prediction_xgb1 = np.zeros(len(test_data))
	for seed in range(en_amount):
		NFOLDS = 5
		train_label = train_data['信用分']
		kfold = KFold(n_splits=NFOLDS, shuffle=True, random_state=seed + 2019)
		kf = kfold.split(train_data, train_label)

		train_data_use = train_data.drop(['用户编码', '信用分'], axis=1)
		test_data_use = test_data.drop(['用户编码'], axis=1)

		cv_pred = np.zeros(test_data.shape[0])
		valid_best_l2_all = 0

		feature_importance_df = pd.DataFrame()
		count = 0

		for i, (train_fold, validate) in enumerate(kf):
			print('fold: ', i, ' training')
			X_train, X_validate, label_train, label_validate = train_data_use.iloc[train_fold, :], \
															   train_data_use.iloc[validate, :], \
															   train_label[train_fold], train_label[validate]
			dtrain = xgb.DMatrix(X_train, label_train)
			dvalid = xgb.DMatrix(X_validate, label_validate)
			watchlist = [(dtrain, 'train'), (dvalid, 'valid_data')]
			bst = xgb.train(dtrain=dtrain, num_boost_round=10000, evals=watchlist, early_stopping_rounds=100,
							verbose_eval=300, params=xgb_params)
			cv_pred += bst.predict(xgb.DMatrix(test_data_use), ntree_limit=bst.best_ntree_limit)
			oof_xgb1[validate] = bst.predict(xgb.DMatrix(X_validate), ntree_limit=bst.best_ntree_limit)
			prediction_xgb1 += bst.predict(xgb.DMatrix(test_data_use),
										   ntree_limit=bst.best_ntree_limit) / kfold.n_splits
			count += 1

		cv_pred /= NFOLDS
		cv_pred_allxgb += cv_pred
	cv_pred_allxgb /= en_amount



def base_sklearn_model(model, train_df, test_df, X, X_submit):
	# process the k-cv
	# k-cv
	N_FOLDS = 5
	y = train_df['信用分']
	kfold = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=2019)
	kf = kfold.split(X, y)

	feature_importance_df = pd.DataFrame()
	count = 0
	for i, (train_idx, test_idx) in enumerate(kf):
		print('fold: ', i, ' training')
		X_train, X_test, y_train, y_test = X.iloc[train_idx, :], X.iloc[test_idx, :], y.iloc[train_idx], y.iloc[
			test_idx]
		#     X_train, X_test, y_train, y_test = X[train_idx, :], X[test_idx, :], y[train_idx], y[test_idx]
		clf = model.fit(X_train, y_train)
		valid_best_l2_all += mean_absolute_error(clf.predict(X_test), y_test)
		cv_pred += clf.predict(X_submit)

	cv_pred /= N_FOLDS
	print('cv score for valid is: ', 1 / (1 + valid_best_l2_all))
	return cv_pred, 1 / (1 + valid_best_l2_all)

def lr_model(train_df, test_df, X, X_submit):
	# Logistical Regression 逻辑斯特回归，感觉不太合适吧，毕竟是做分类的

	base_sklearn_model(LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial'), train_df, test_df, X, X_submit)


def linear_regression_model(train_df, test_df, X, X_submit):
	base_sklearn_model(LinearRegression(), train_df, test_df, X, X_submit)