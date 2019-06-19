#coding: utf-8
# import packages
import pandas as pd
import seaborn as sns
import numpy as np

import matplotlib.pyplot as plt
from data_preprocessing import base_process
from feature_engineering import create_features
from models import lgb_model, linear_regression_model, lr_model
# pd.set_option('display.max_columns', 50)

import warnings
warnings.filterwarnings("ignore")

import time
import sys
stdout = sys.stdout
reload(sys)
sys.setdefaultencoding('utf-8')
sys.stdout = stdout

plt.rcParams['font.sans-serif'] = ['SimHei'] # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False # 用来正常显示负号



if __name__ == '__main__':
	# 0 -- load data
	data_dir = '../data/'
	train_df = pd.read_csv(data_dir + 'train_dataset.csv')
	test_df = pd.read_csv(data_dir + 'test_dataset.csv')


	# 1 -- data preprocessing
	train_df = base_process(train_df)
	test_df = base_process(test_df)


	# 2 -- feature engineering
	train_df = create_features(train_df)
	test_df = create_features(test_df)

	## drop useless features
	drop_cols = ['用户编码', '是否黑名单客户']
	X = train_df.drop(drop_cols + ['信用分'], axis=1)
	X_submit = test_df.drop(drop_cols, axis=1)


	# 3 -- train model
	start_time = time.time()
	cv_pred, model_score = lgb_model(train_df, test_df, X, X_submit)
	print 'training time: ' + str(time.time() - start_time) + 's'


	# 4 -- submit
	submit_df = test_df[['用户编码']]
	submit_df['score'] = cv_pred
	submit_df.columns = ['id', 'score']
	submit_df['score'] = submit_df['score'].apply(lambda x: int(np.round(x)))

	csv_name = './submission/baseline_' + str(time.strftime('%Y%m%d-%H:%M:%S')) + '_{}_'.format(model_score) + '.csv'
	print 'saving ' + csv_name + ' <|-.-|>'
	submit_df.to_csv(csv_name, index=False)

