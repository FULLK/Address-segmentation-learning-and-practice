"""
根据语料库，训练初始状体概率、状态转移概率、发射概率
"""
import pandas as pd
import numpy as np
from common import cal_log, save_cache
import os
import config


# # 定义状态的前置状态
# PrevStatus = {
#     'pB': [],  # 省份的开始字
#     'pM': ['pB'],  # 省份的中间字
#     'pE': ['pB', 'pM'],  # 省份的结尾字
#     'cB': ['pE'],  # 市的开始字
#     'cM': ['cB'],  # 市的中间字
#     'cE': ['cB', 'cM'],  # 市的结尾字
#     'aB': ['cE'],  # 区的开始字
#     'aM': ['aB'],  # 区的中间字
#     'aE': ['aB', 'aM'],  # 区的结尾字
#     'dB': ['aE'],  # 详细地址的开始字
#     'dM': ['dB'],  # 详细地址的中间字
#     'dE': ['dB', 'dM'],  # 详细地址的结尾字
# }


# 生成隐马尔科夫模型的训练概率
def build_porb():
    start_p, trans_p, emit_p = cal_prob()
    save_cache(start_p, os.path.join(config.get_data_path(), 'start_p.p'))
    save_cache(trans_p, os.path.join(config.get_data_path(), 'trans_p.p'))
    save_cache(emit_p, os.path.join(config.get_data_path(), 'emit_p.p'))


# 计算状态转移概率
def cal_prob():
    # 初始化
    trans_p = {}  # 状态转移概率
    emit_p = {}  # 发射概率

    # 读取省市区的标准名称
    address_standard = pd.read_table(r'E:\project\poc\address_cut\data\dict3.txt',
                                     header=None, names=['name', 'num', 'type'], delim_whitespace=True)

    #header=None: 这个参数告诉Pandas这个文件没有标题行,所以需要自动生成列名。
    #names=['name', 'num', 'type']: 这个参数指定了要使用的列名。在这个例子中,列名分别是"name"、"num"和"type"
    # delim_whitespace=True: 这个参数告诉Pandas使用空白字符(空格和/或制表符)作为分隔符来分割各个列。
    # 读取一些切分地址后的样本
    address_sample = pd.read_excel(r'E:\project\poc\address_cut\data\df.xlsx')

    # 1、计算状态转移概率矩阵
    trans_p.setdefault('pB', {})['pE'],  trans_p.setdefault('pB', {})['pM'], \
        trans_p.setdefault('pM', {})['pM'], trans_p.setdefault('pM', {})['pE'] = \
        cal_trans_BE_BM_MM_ME(set(address_standard.loc[address_standard['type'] == 'prov', 'name'].values))
# trans_p.setdefault('pB', {})['pE']类似trans_p = {'pB': {'pE': <未指定的值>}}
    #从 address_standard DataFrame 中筛选出 'type' 列等于 'prov' 的行,取出 'name' 列的值。
    #将这些省份名称去重后,作为参数传递给 cal_trans_BE_BM_MM_ME 函数
    trans_p.setdefault('cB', {})['cE'], trans_p.setdefault('cB', {})['cM'], \
        trans_p.setdefault('cM', {})['cM'], trans_p.setdefault('cM', {})['cE'] = \
        cal_trans_BE_BM_MM_ME(set(address_standard.loc[address_standard['type'] == 'city', 'name'].values))

    trans_p.setdefault('aB', {})['aE'], trans_p.setdefault('aB', {})['aM'], \
        trans_p.setdefault('aM', {})['aM'], trans_p.setdefault('aM', {})['aE'] = \
        cal_trans_BE_BM_MM_ME(set(address_standard.loc[address_standard['type'] == 'dist', 'name'].values))

    detailed_address_sample = get_detailed_address(address_sample)

    # 详细地址样本库

    trans_p.setdefault('dB', {})['dE'], trans_p.setdefault('dB', {})['dM'], \
        trans_p.setdefault('dM', {})['dM'], trans_p.setdefault('dM', {})['dE'] = \
        cal_trans_BE_BM_MM_ME(set(detailed_address_sample))

    # 计算省市区详细地址四者之间的对应的隐藏状态转移矩阵
    cal_trans_p_c_a_d(address_sample, trans_p)

    # 2、计算初始概率矩阵
    start_p = cal_start_p(address_sample)  # 初始状态概率

    # 3、计算发射概率矩阵
    emit_p['pB'], emit_p['pM'], emit_p['pE'] = \
        cal_emit_p(set(address_standard.loc[address_standard['type'] == 'prov', 'name'].values))

    emit_p['cB'], emit_p['cM'], emit_p['cE'] = \
        cal_emit_p(set(address_standard.loc[address_standard['type'] == 'city', 'name'].values))

    emit_p['aB'], emit_p['aM'], emit_p['aE'] = \
        cal_emit_p(set(address_standard.loc[address_standard['type'] == 'dist', 'name'].values))

    emit_p['dB'], emit_p['dM'], emit_p['dE'] = \
        cal_emit_p(set(detailed_address_sample))

    return start_p, trans_p, emit_p


# 计算发射概率矩阵
def cal_emit_p(str_set): #省或市或区汉字
    str_list = list(str_set)
    stat_B = {}
    stat_M = {}
    stat_E = {}
    length = len(str_list)
    M_length = 0
    for str in str_list: # 字符串组
        str_len = len(str)
        for index, s in enumerate(str):# 字符串
            if index == 0:
                stat_B[s] = stat_B.get(s, 0) + 1
                # get得到字典中s的值，没用就返回0
            elif index < str_len - 1:
                stat_M[s] = stat_M.get(s, 0) + 1
                M_length += 1
                # 中间字符个数
            else:
                stat_E[s] = stat_E.get(s, 0) + 1
                # 末尾字符个数
    B = {key: cal_log(value / length) for key, value in stat_B.items()}
    # 开头出现某个汉字在整个省或市或区汉字汉字字符串组中的比例的对数
    M = {key: cal_log(value / M_length) for key, value in stat_M.items()}
    # 中间出现某个汉字在整个省或市或区汉字汉字字符串组中的比例的对数
    E = {key: cal_log(value / length) for key, value in stat_E.items()}
    # 末尾出现某个汉字在整个省或市或区汉字汉字字符串组中的比例的对数
    #cal_log 是指某个对数函数
    return B, M, E


# 根据地址样本，省市区详细地址之间的转移矩阵
def cal_trans_p_c_a_d(address_df, t_p):
    df_prov = address_df[address_df['prov'].isnull().values == False]
    # prov列不为空的行
    df_prov_no_city = df_prov[df_prov['city'].isnull().values == True]
    # prov列不为空的并且city列为空的行
    df_prov_no_city_no_area = df_prov_no_city[df_prov_no_city['dist'].isnull().values == True]
    # prov列不为空的并且city列为空并且dist列为空的行
    t_p.setdefault('pE', {})['cB'] = cal_log(1 - len(df_prov_no_city) / len(df_prov))
    # 对于1-省后面不是城市开始的概率
    # 省后面是城市开始
    t_p.setdefault('pE', {})['aB'] = cal_log((len(df_prov_no_city) - len(df_prov_no_city_no_area)) / len(df_prov))
    # 省后面直接是地区开始 对于身后面不是城市的概率减去省后面既不是城市也不是地区的概率即省后面是地区的概率
    t_p.setdefault('pE', {})['dB'] = cal_log(len(df_prov_no_city_no_area) / len(df_prov))
    # 省后面直接是详细地址开始  省后面既不是城市也不是地区开始的概率即省后面是详细地址开始的概率
    df_city = address_df[address_df['city'].isnull().values == False]
    # 城市列不为空的行
    df_city_no_area = df_city[df_city['dist'].isnull().values == True]
    #  城市列不为空且地区列为空的行
    t_p.setdefault('cE', {})['aB'] = cal_log(1 - len(df_city_no_area) / len(df_city))
    # 城市后面是是区域开始
    t_p.setdefault('cE', {})['dB'] = cal_log(len(df_city_no_area) / len(df_city))
    # 城市后面是详细地址开始
    t_p.setdefault('aE', {})['dB'] = cal_log(1.0)
    # 区域后面是详细地址开始  百分之百有可能是规定吧

# 根据地址样本， 计算初始概率矩阵
def cal_start_p(address_df):
    length = len(address_df)
    df_prov_nan = address_df[address_df['prov'].isnull().values == True]
    # 省列为空的行
    length_pB = length - len(df_prov_nan)
    # 省列不为空的行 即起始状态为 pb
    df_city_nan = df_prov_nan[df_prov_nan['city'].isnull().values == True]
    # 省列为空且市列为空的行
    length_cB = len(df_prov_nan) - len(df_city_nan)
    # 省列为空并且市列不为空的个数 即起始状态为 cb
    df_area_nan = df_city_nan[df_city_nan['dist'].isnull().values == True]
    # 省列为空且市列为空且地区列为空的行
    length_aB = len(df_city_nan) - len(df_area_nan)
    # 省列为空并且市列为空但地区列不为空的个数 即起始状态为 ab
    length_dB = len(df_area_nan)
    # 省列为空且市列为空且地区列为空且详细地址不为空（详细地址列一定不为空规定吧）
    s_p = {'pB': cal_log(length_pB / length),  # 省份的开始字
           'pM': -3.14e+100,  # 省份的中间字
           'pE': -3.14e+100,  # 省份的结尾字
           'cB': cal_log(length_cB / length),  # 市的开始字
           'cM': -3.14e+100,  # 市的中间字
           'cE': -3.14e+100,  # 市的结尾字
           'aB': cal_log(length_aB / length),  # 区的开始字
           'aM': -3.14e+100,  # 区的中间字
           'aE': -3.14e+100,  # 区的结尾字
           'dB': cal_log(length_dB / length),  # 详细地址的开始字
           'dM': -3.14e+100,  # 详细地址的中间字
           'dE': -3.14e+100,  # 详细地址的结尾字
           }# -3.14e+100表示不可能 因为起始状态只有可能是 pb cb ab db
    return s_p


# 获取样本数据中的详细地址
def get_detailed_address(address_df):
    detailed_address = []
    for index, row in address_df.iterrows():
        tmp = row['address_'].strip().strip('\ufeff')
        if row['prov'] is not None and row['prov'] is not np.nan:
            tmp = tmp.replace(row['prov'], '', 1)
            # prov 列不为空且不为 NaN,则使用 replace() 方法将 tmp 中的省份信息删除
        if row['city'] is not None and row['city'] is not np.nan:
            tmp = tmp.replace(row['city'], '', 1)
        if row['dist'] is not None and row['dist'] is not np.nan:
            tmp = tmp.replace(row['dist'], '', 1)
        detailed_address.append(tmp)
        # 最后只剩详细地址部分了
    return detailed_address


# 计算字符串数组中“开始字-->结束字”、“开始字-->中间字”、“中间字-->中间字”和“中间字-->结束字”的转移概率
def cal_trans_BE_BM_MM_ME(str_set): # 根据各个样本的字符长度来判断对应的隐藏状态的个数从而计算概率
    str_list = list(str_set)
    length = len(str_list)
    if length == 0:
        raise Exception('输入的集合为空')

    str_len_ori = np.array([len(str) for str in str_list])
    # 各个字数长度的数组
    # 筛选出字数大于1的地名,进行计算
    str_len = str_len_ori[np.where(str_len_ori > 1)]
    # if sum(str_len < 2):
    #     raise Exception('含有单个字的省、市、区名称!')

    # “开始字-->结束字“的概率
    # 两个字就是直接开始字和结束字
    p_BE = sum(str_len == 2) / length
    # 除了开始字到中间字就是开始字到结束字了
    # “开始字-->中间字”的概率
    p_BM = 1 - p_BE
    # “中间字 -->结束字”的概率
    # 字数大于2说明有中间字，并且只存在一个中间字到结束字
    # 减2是代表当前存在中间字到中间字和中间字到结束字，如果只有中间字到 结束字那么就只有1，如果是存在一个中间字到中间字那么就是2
    p_ME = sum(str_len > 2) / sum(str_len - 2)  # ？？？ 负数  ？？？？？
    # “中间字-->中间字”的概率
    # 除了中间字到中间字就是中间字到结束字了
    p_MM = 1 - p_ME

    return cal_log(p_BE), cal_log(p_BM), cal_log(p_MM), cal_log(p_ME)


if __name__ == '__main__':
    build_porb()
    pass

