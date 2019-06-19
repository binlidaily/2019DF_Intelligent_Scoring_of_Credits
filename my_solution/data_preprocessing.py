#coding: utf-8
#bin
import numpy as np

def base_process(data_df):
    transform_value_feats = ['用户年龄', '用户网龄（月）', '当月通话交往圈人数', '近三个月月均商场出现次数',
                            '当月网购类应用使用次数', '当月物流快递类应用使用次数', '当月金融理财类应用使用总次数', 
                             '当月视频播放类应用使用次数', '当月飞机类应用使用次数', '当月火车类应用使用次数', 
                             '当月旅游资讯类应用使用次数']
    bill_feats = ['缴费用户最近一次缴费金额（元）', '用户近6个月平均消费值（元）','用户账单当月总费用（元）', 
                   '用户当月账户余额（元）']
    log_feats = ['当月网购类应用使用次数', '当月金融理财类应用使用总次数', '当月视频播放类应用使用次数']
    
    # 处理极小或极大的离散点
    for col in transform_value_feats + bill_feats:
        up_limit = np.percentile(data_df[col].values, 99.9) # 99.9%分位数
        low_limit = np.percentile(data_df[col].values, 0.1) # 0.1%分位数
        data_df.loc[data_df[col] > up_limit, col] = up_limit
        data_df.loc[data_df[col] < low_limit, col] = low_limit
    
    # 解决正太分布左偏的情况，取对数
    for col in bill_feats + log_feats:
        data_df[col] = data_df[col].map(lambda x : np.log1p(x))
    
    return data_df

# run
# train_df = base_process(train_df)
# test_df = base_process(test_df)


def drop_cols(data_df, cols, axis=1, inplace=False):
	if inplace:
		data_df.drop(cols, axis=axis)
	else:
		new_df = data_df
		return new_df.drop(cols, axis=axis)
