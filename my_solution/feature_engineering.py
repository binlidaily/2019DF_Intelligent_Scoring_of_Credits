#coding: utf-8
def get_features(data_df):
    # 对年龄异常值取
    data_df['用户年龄'][data_df['用户年龄'] == 0] = data_df['用户年龄'].mode() # 线下测试，众数比平均数好
    ## 对年龄的另外一种处理方式，离散化，当然也可以直接用现成的分桶
    def map_age(age_x):
        if age_x <= 18:
            return 1
        elif x <= 20:
            return 2
        elif x <= 35:
            return 3
        elif x <= 45:
            return 4
        else:
            return 5

    # 构造一些特出特征
    data_df['缴费金额是否能覆盖当月账单'] = data_df['缴费用户最近一次缴费金额（元）'] - data_df['用户账单当月总费用（元）']
    data_df['最近一次缴费是否超过平均消费额'] = data_df['缴费用户最近一次缴费金额（元）'] - data_df['用户近6个月平均消费值（元）']
    data_df['当月账单是否超过平均消费额'] = data_df['用户账单当月总费用（元）'] - data_df['用户近6个月平均消费值（元）']

    # 对 bool 特征进行简单构造
    data_df['是否去过高档商场'] = data_df['当月是否到过福州山姆会员店'] + data_df['当月是否逛过福州仓山万达']
    ## 检查后发现结果为2的比较稀少，于是将1、2都归到1中
    data_df['是否去过高档商场'] = data_df['是否去过高档商场'].map(lambda x : 1 if x >= 1 else 0)

    data_df['是否_商场_电影'] = data_df['是否去过高档商场'] * data_df['当月是否看电影']
    data_df['是否_商场_旅游'] = data_df['是否去过高档商场'] * data_df['当月是否景点游览']
    data_df['是否_商场_体育馆'] = data_df['是否去过高档商场'] * data_df['当月是否体育场馆消费']
    data_df['是否_电影_体育馆'] = data_df['当月是否看电影'] * data_df['当月是否体育场馆消费']
    data_df['是否_电影_旅游'] = data_df['当月是否看电影'] * data_df['当月是否景点游览']
    data_df['是否_旅游_体育馆'] = data_df['当月是否景点游览'] * data_df['当月是否体育场馆消费']

    data_df['是否_商场_旅游_体育馆'] = data_df['是否去过高档商场'] * data_df['当月是否景点游览'] * data_df['当月是否体育场馆消费']
    data_df['是否_商场_电影_体育馆'] = data_df['是否去过高档商场'] * data_df['当月是否看电影'] * data_df['当月是否体育场馆消费']
    data_df['是否_商场_电影_旅游'] = data_df['是否去过高档商场'] * data_df['当月是否看电影'] * data_df['当月是否景点游览']
    data_df['是否_体育馆_电影_旅游'] = data_df['当月是否体育场馆消费'] * data_df['当月是否看电影'] * data_df['当月是否景点游览']

    data_df['是否_商场_体育馆_电影_旅游'] = data_df['是否去过高档商场'] * data_df['当月是否体育场馆消费'] * \
                                        data_df['当月是否看电影'] * data_df['当月是否景点游览']

    # APP 使用次数
    data_df['交通类应用使用次数'] = data_df['当月飞机类应用使用次数'] + data_df['当月火车类应用使用次数']
    discretize_feats = ['交通类应用使用次数', '当月物流快递类应用使用次数', '当月飞机类应用使用次数',
                           '当月火车类应用使用次数','当月旅游资讯类应用使用次数']

    def map_discreteze(feat_x):
        if feat_x == 0:
            return 0
        elif feat_x <= 5:
            return 1
        elif feat_x <= 15:
            return 2
        elif feat_x <= 50:
            return 3
        elif feat_x <= 100:
            return 4
        else:
            return 5

    for col in discretize_feats:
        data_df[col] = data_df[col].map(lambda x : map_discreteze(x))

    # 用户费用相关特征
    ## 不同的充值路径
    data_df['不同充值途径'] = 0
    data_df['不同充值途径'][(data_df['缴费用户最近一次缴费金额（元）'] % 10 == 0) &
                      (data_df['缴费用户最近一次缴费金额（元）'] != 0)] = 1
    ## 费用稳定性
    data_df['当前费用稳定性'] = data_df['用户账单当月总费用（元）'] / (data_df['用户近6个月平均消费值（元）'] + 1)
    ## 当月话费/当月账户余额
    data_df['用户余额比例'] = data_df['用户账单当月总费用（元）'] / (data_df['用户当月账户余额（元）'] + 1)
    ## new 6个月占比总费用
    data_df['6个月占比总费用'] = data_df['用户近6个月平均消费值（元）'] / (data_df['用户账单当月总费用（元）'] + 1)

    # 构造 ratio 比例特征

    return data_df

# =============================

# # Dimension Reduction 降维
# from sklearn.decomposition import PCA
# pca = PCA(n_components=600)
# pca.fit(X)
# X = pca.fit_transform(X)
# pca.fit(X_submit)
# X_submit = pca.fit_transform(X_submit)
# print X.shape, X_submit.shape


# my methods to create new features
def create_features(data_df):
    # 异常值处理
    ## 对年龄异常值取
    data_df.loc[data_df['用户年龄'] == 0, '用户年龄'] = data_df['用户年龄'].mode() # 线下测试，众数比平均数好
    ## 用户话费敏感度处理
    data_df.loc[data_df['用户话费敏感度'] == 0, '用户话费敏感度'] = data_df['用户话费敏感度'].mode()
    

    # 用户费用相关特征
    ## 不同的充值路径
    data_df['不同充值途径'] = 0
    data_df.loc[(data_df['缴费用户最近一次缴费金额（元）'] % 10 == 0) & 
                      (data_df['缴费用户最近一次缴费金额（元）'] != 0), '不同充值途径'] = 1
    ## 费用稳定性
    data_df['当前费用稳定性'] = data_df['用户账单当月总费用（元）'] / (data_df['用户近6个月平均消费值（元）'] + 1)
   
    
    # 构造 ratio 比例特征
    ## '缴费用户最近一次缴费金额（元）'/'用户当月账户余额（元）'
    data_df['充值_余额_比例'] = data_df['缴费用户最近一次缴费金额（元）'] / (data_df['用户当月账户余额（元）'] + 1)
    ## 用户账单当月总费用/当月账户余额
    data_df['月费_余额_比例'] = data_df['用户账单当月总费用（元）'] / (data_df['用户当月账户余额（元）'] + 1)
    # '用户账单当月总费用（元）'/ '缴费用户最近一次缴费金额（元）'
    data_df['月费_缴费_比例'] = data_df['用户账单当月总费用（元）'] / (data_df['缴费用户最近一次缴费金额（元）'] + 1)
    ## '用户近6个月平均消费值（元）'/ '缴费用户最近一次缴费金额（元）'
    data_df['均费_缴费_比例'] = data_df['用户近6个月平均消费值（元）'] / (data_df['缴费用户最近一次缴费金额（元）'] + 1)
    ## '用户近6个月平均消费值（元）' / 
    data_df['均费_月费_比例'] = data_df['用户近6个月平均消费值（元）'] / (data_df['用户账单当月总费用（元）'] + 1)
    
    ## 用户上网年龄
    data_df['用户上网年龄'] = data_df['用户年龄'] - data_df['用户网龄（月）']
    ## '用户网龄（月）'/'用户年龄', '用户年龄'/ '用户网龄（月）'不是很好算出来，毕竟是个大数
    data_df['网龄_年龄_比例'] = data_df['用户网龄（月）'] / (data_df['用户年龄'] + 1)
    
    
    # 构造加减特征
    data_df['缴费金额是否能覆盖当月账单'] = data_df['缴费用户最近一次缴费金额（元）'] - data_df['用户账单当月总费用（元）']
    data_df['最近一次缴费是否超过平均消费额'] = data_df['缴费用户最近一次缴费金额（元）'] - data_df['用户近6个月平均消费值（元）']
    data_df['当月账单是否超过平均消费额'] = data_df['用户账单当月总费用（元）'] - data_df['用户近6个月平均消费值（元）']
    
    # 对 bool 特征进行简单构造
    data_df['是否去过高档商场'] = data_df['当月是否到过福州山姆会员店'] + data_df['当月是否逛过福州仓山万达']
    ## 检查后发现结果为2的比较稀少，于是将1、2都归到1中
    data_df['是否去过高档商场'] = data_df['是否去过高档商场'].map(lambda x : 1 if x >= 1 else 0)
    
    
    data_df['是否_商场_电影'] = data_df['是否去过高档商场'] * data_df['当月是否看电影']
    data_df['是否_商场_旅游'] = data_df['是否去过高档商场'] * data_df['当月是否景点游览']
    data_df['是否_商场_体育馆'] = data_df['是否去过高档商场'] * data_df['当月是否体育场馆消费']
    data_df['是否_电影_体育馆'] = data_df['当月是否看电影'] * data_df['当月是否体育场馆消费']
    data_df['是否_电影_旅游'] = data_df['当月是否看电影'] * data_df['当月是否景点游览']
    data_df['是否_旅游_体育馆'] = data_df['当月是否景点游览'] * data_df['当月是否体育场馆消费']
    
    data_df['是否_商场_旅游_体育馆'] = data_df['是否去过高档商场'] * data_df['当月是否景点游览'] * data_df['当月是否体育场馆消费']
    data_df['是否_商场_电影_体育馆'] = data_df['是否去过高档商场'] * data_df['当月是否看电影'] * data_df['当月是否体育场馆消费']
    data_df['是否_商场_电影_旅游'] = data_df['是否去过高档商场'] * data_df['当月是否看电影'] * data_df['当月是否景点游览']
    data_df['是否_体育馆_电影_旅游'] = data_df['当月是否体育场馆消费'] * data_df['当月是否看电影'] * data_df['当月是否景点游览']
    
    data_df['是否_商场_体育馆_电影_旅游'] = data_df['是否去过高档商场'] * data_df['当月是否体育场馆消费'] * \
                                        data_df['当月是否看电影'] * data_df['当月是否景点游览']
    
    # 杰少特征参考
    data_df['次数'] = data_df['当月网购类应用使用次数'] +  data_df['当月物流快递类应用使用次数'] +  data_df['当月金融理财类应用使用总次数'] + \
                data_df['当月视频播放类应用使用次数'] + data_df['当月飞机类应用使用次数'] + data_df['当月火车类应用使用次数'] + \
                data_df['当月旅游资讯类应用使用次数']  + 1

    for col in ['当月金融理财类应用使用总次数', '当月旅游资讯类应用使用次数']: # 这两个比较积极向上一点
        data_df[col + '百分比'] = data_df[col] / data_df['次数'] 

    data_df['当月通话人均话费'] = data_df['用户账单当月总费用（元）'] / (data_df['当月通话交往圈人数'] + 1)

    data_df['上个月费用'] = data_df['用户当月账户余额（元）'] + data_df['用户账单当月总费用（元）']

    data_df['近似总消费'] = data_df['用户近6个月平均消费值（元）'] * data_df['用户网龄（月）'] 
    
    
    return data_df


# run
# train_df = create_features(train_df)
# test_df = create_features(test_df)