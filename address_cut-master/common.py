import numpy as np
import pickle

# 极小值
EPS = 1e-100
MIN_FLOAT = -3.14e100


# 安全计算log
def cal_log(x):
    return np.log(x + EPS)


# 加载二进制文件
def load_cache(cache_file_name):
    with open(cache_file_name, 'rb') as f:
        ds = pickle.load(f)  # 从二进制数据读取和还原数据
        return ds


# 转换为二进制文件存储
def save_cache(cache, cache_file_name):
    with open(cache_file_name, 'wb') as f:
        pickle.dump(cache, f)