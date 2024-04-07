"""
全局静态配置文件，方便统一管理和修改。
"""
import os


class GlobalVar:
    # data_path = r'E:\project\poc\address_cut\data'
    data_path = os.path.join(os.getcwd(), 'data')  # 即当前路径


def get_data_path():
    return GlobalVar.data_path #即返回数据所在路径


