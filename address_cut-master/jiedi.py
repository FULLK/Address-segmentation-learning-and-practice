"""
根据隐马尔科夫模型，进行切分地址
其中隐层状态定义为：
    'pB':  # 省份的开始字
    'pM':  # 省份的中间字
    'pE':  # 省份的结尾字
    'cB':  # 市的开始字
    'cM':  # 市的中间字
    'cE':  # 市的结尾字
    'aB':  # 区的开始字
    'aM':  # 区的中间字
    'aE':  # 区的结尾字
    'dB':  # 详细地址的开始字
    'dM':  # 详细地址的中间字
    'dE':  # 详细地址的结尾字
"""
from common import load_cache, MIN_FLOAT  # 导入了common.py中的load_cache函数和MIN_FLOAT变量
import config  # 导入了config.py
import os
import pandas as pd
import numpy as np
import datetime


# 分词器
class Tokenizer(object):
    def __init__(self):
        try:
            self.start_p = load_cache(os.path.join(config.get_data_path(), 'start_p.p'))
            self.trans_p = load_cache(os.path.join(config.get_data_path(), 'trans_p.p'))
            self.emit_p = load_cache(os.path.join(config.get_data_path(), 'emit_p.p'))
            self.mini_d_emit_p = self.get_mini_emit_p('d')
            standard_address_library = pd.read_excel(os.path.join(config.get_data_path(), 'adress_area.xlsx'))
            self.standard_address_library = standard_address_library.fillna('')  # 用于将 DataFrame 中的所有缺失值(NaN)替换为空字符串 ''
            self.time = datetime.datetime.now()
            self.time_takes = {}  # 空的字典(dictionary)
        except Exception:
            raise  # 直接将异常再次抛出

    # 维特比算法求大概率路径
    def viterbi(self, address):
        length = len(address)
        V = []  # 存储中间结果
        path = {}  # 存储最优路径
        temp_pro = {}  # 字典将用于存储中间计算的概率值
        for hidden_state, prop in self.start_p.items():  # 所有键值对 隐藏状态的初始概率分布
            temp_pro[hidden_state] = self.start_p[hidden_state] + self.get_emit_p(hidden_state,
                                                                                  address[0])  # 各个初始状态对应的此时
            path[hidden_state] = [hidden_state]  # 各个初始状态状态此时对应的隐藏
        V.append(temp_pro)  # 将各个初始状态对应的初始状态概率和发射概率和的字典加入V

        for i_c, character in enumerate(address[1:]):  # 从地址字符串的第二个字符开始遍历。
            temp_pro = {}
            new_path = {}
            for hidden_state, _ in self.start_p.items():  # 尝试各个隐藏状态并保存各个隐藏状态对应的最大概率
                pre_hidden_state_pro = {pre_hidden_state: (pre_pro
                                                           + self.get_trans_p(pre_hidden_state, hidden_state)
                                                           + self.get_emit_p(hidden_state, character))
                                        for pre_hidden_state, pre_pro in V[i_c].items()}
                # 字典 键为前一个各种隐藏状态  值为前一个隐藏状态的概率+前一个隐藏状态的概率到当前隐藏状态的概率+当前隐藏状态的概率到观测状态的概率

                max_pre_hidden_state, max_pro = max(pre_hidden_state_pro.items(), key=lambda x: x[1])
                # 比较的依据是每个键值对中的值(x[1])
                temp_pro[hidden_state] = max_pro  # 当前的某个隐藏状态的最大可能概率
                new_path[hidden_state] = path[max_pre_hidden_state] + [hidden_state]  # 更新各个隐藏状态对应的最大路径
                #  path 字典中 max_pre_hidden_state 对应的路径, 加上当前的 hidden_state, 作为新的路径存储到 new_path 字典中
            V.append(temp_pro)  # 加入各个隐藏状态和其最大概率的列表
            path = new_path  # 替换

        # 解析最大概率路径, 只从可能的最后一个字符状态进行解析
        (prob, state) = max((V[length - 1][y], y) for y, _ in self.start_p.items())
        # V的第一个键是时刻，第二个键为各个隐藏状态 就是最后一个时刻的各个隐藏状态对应各个概率的最大值
        self.note_time_takes('viterbi_time_takes', self.get_time_stamp())
        return prob, path[state]

    # prob对应V[length - 1][y]即最优概率   state 对应y即最优状态下的最后一个时刻对应的隐藏状态，此时对应path为当前这条路径

    # 获取隐含状态到可见状态的发射概率
    def get_emit_p(self, hidden_state, visible_state):

        if 'd' in hidden_state:  # 有详细地址部分
            # 对省、市、县等关键字进行过滤，防止出现在详细地址中
            if '省' in visible_state or '市' in visible_state or '县' in visible_state:
                return self.emit_p.get(hidden_state, {}).get(visible_state, MIN_FLOAT)
            # 详细地址部分出现省、市、县的概率极小，如果确实存在发射概率就返回，不存在就返回一个很小的值
            else:
                return self.emit_p.get(hidden_state, {}).get(visible_state, self.mini_d_emit_p)
            # 正常详细地址部分不出现省、市、县的情况，此时如果没有找到出现的就以隐藏状态的最小发射概率为准
        # 无详细地址部分
        else:
            return self.emit_p.get(hidden_state, {}).get(visible_state, MIN_FLOAT)
        # 正常返回，没有就返回一个很小的值
        pass

    # 获取详细地址最小的发射概率
    def get_mini_emit_p(self, h_state_feature):
        mini_p = -MIN_FLOAT  # 大值
        for h_state, v_states_pro in self.emit_p.items():
            if h_state_feature in h_state:  # 隐藏状态存在d
                for v_state, pro in v_states_pro.items():  # 该隐藏状态对应的观测状态和发射概率
                    mini_p = min(mini_p, pro)  # 找一个更小的
        return mini_p  # 即所有隐藏状态转换到所有观测状态中最小的概率

    # 获取前一隐含状态到下一隐含状态的转移概率
    def get_trans_p(self, pre_h_state, h_state):
        return self.trans_p.get(pre_h_state, {}).get(h_state, MIN_FLOAT)

    # 修正市区详细地址
    def revise_address_cut(self, pro, city, area, detailed):
        # 1、修正省市区地址
        list_addr = [pro, city, area, detailed]  # 将各个字符合并成一个列表
        col_name = ['pro', 'city', 'area']  # 定义三个列名
        revise_addr_list = ['', '', '', '']  # 修正后的地址信息
        i = 0
        k = 0
        filter_df = self.standard_address_library
        while i < len(col_name) and k < len(col_name):
            # 三列都判断过一遍后或者已经判好（判好表示存在或者为空）到第四列出去（K）
            add = list_addr[k]  # 得到的可观测序列的列表
            if add == '':  # 只观测前三个  k < len(col_name)
                k += 1  # 前三个出现空的时候k+1，表示这个已经存在并判断完了，不然会判断到详细地址去
                continue
            while i < len(col_name):
                # 避免重复判断字符串是否被包含，优化匹配效率
                area_set = set(filter_df[col_name[i]].values)
                # 用于获取 DataFrame 中某一列的唯一值集合 当前列(省、市或区)在标准地址库中的所有唯一值
                match_area_set = {a for a in area_set if add in a}
                # 当前可观测状态出现在列中的所有值的集合
                # 遍历 area_set 中的每个元素 a(也就是每个区域),并且只有当 add 这个变量存在于 a 中时,才会将 a 包含在新的集合中
                filter_temp = filter_df.loc[filter_df[col_name[i]].isin(match_area_set), :]
                # 检查 filter_df[col_name[i]] 中的每个值是否存在于 match_area_set 集合中
                # 用这个布尔型 Series 作为行索引,选择满足条件的行,并返回一个新的 DataFrame filter_temp。
                if len(filter_temp) > 0:
                    revise_addr_list[i] = add  # 存在就保持不变
                    filter_df = filter_temp  # 过滤后的赋值data
                    i += 1  # 下一个
                    k += 1  #
                    break
                else:  # 不存在 就看下一个属性（省市区）
                    i += 1  # k没有+1说明不存在一个属性
                    continue
        # 将剩余的值全作为详细地址
        revise_addr_list[3] = ''.join(list_addr[k:len(list_addr)])
        # 不存在的属性部分当作详细地址
        self.note_time_takes('revise_address_0_time_takes', self.get_time_stamp())

        # 2、补全省市区地址
        effective_index_arr = np.where([s != '' for s in revise_addr_list[0:3]])[0]
        # np.where（数组）数组如果是多维的，返回值也是多维数组，所以用到了[0]，这里一维字符串数组返回值也只有一个数组
        # 返回所有大于5的数组元素的索引所构成数组
        max_effective_index = 0
        if len(effective_index_arr) > 0:
            max_effective_index = effective_index_arr[-1]
            # 最后一个不为空的索引值
        if len(filter_df) > 0:  # 非空的都是存在于样本中的
            for index, addr in enumerate(revise_addr_list):
                if addr == '' and index < max_effective_index:  # 空的在不空以下
                    revise_addr_list[index] = filter_df.iloc[0, :][col_name[index]]
                    # [0]: 这表示访问第 0 行(也就是第一行)的数据。在 Pandas 中,行索引从 0 开始。
                    # [:]: 这表示访问该行的所有列。冒号 : 表示选择该行的所有列。
                    # 补全为剩余的filter_df第零行的对应属性列的值

        self.note_time_takes('revise_address_1_time_takes', self.get_time_stamp())
        return revise_addr_list[0], revise_addr_list[1], revise_addr_list[2], revise_addr_list[3]

    # 初始化耗时初始时刻和耗时记录
    def time_init(self):
        self.time = datetime.datetime.now()
        self.time_takes = {}

    # 计算初始时刻至今的耗时
    def get_time_stamp(self):
        time_temp = datetime.datetime.now()
        time_stamp = (time_temp - self.time).microseconds / 1000000
        self.time = time_temp
        return time_stamp

    # 记录各个时间段名称和耗时时间字典
    def note_time_takes(self, key, time_takes):
        self.time_takes[key] = time_takes


dt = Tokenizer()


# 对输入的地址进行切分
def cut(address):
    # 带切分地址必须大于一个字符
    if address is None or len(address) < 2:
        return '', '', '', '', 0, [], {}
    # address 是 None，或者它是一个空字符串或只包含一个字
    dt.time_init()
    p, max_path = dt.viterbi(address)  # max_path是一个隐藏状态字符串
    pro = ''
    city = ''
    area = ''
    detailed = ''
    for i_s, state in enumerate(max_path):  # 相当于遍历隐藏状态字符串
        # enumerate() 函数用于将一个可迭代对象(如列表、字符串等)转换为一个枚举对象(enumerate object)
        character = address[i_s]  # 可观测序列与隐藏状态序列相同下标下一一对比
        if 'p' in state:
            pro += character
        elif 'c' in state:
            city += character
        elif 'a' in state:
            area += character
        else:
            detailed += character  # 当前对应的隐藏状态是啥就往对应分布的地方加上当前隐藏状态对应的可观测状态的字符

    # 通过字典修正输出
    r_pro, r_city, r_area, r_detailed = dt.revise_address_cut(pro, city, area, detailed)
    return r_pro, r_city, r_area, r_detailed, p, max_path, dt.time_takes


if __name__ == '__main__':
    # 读取execel批量测试
    # 读取一些切分地址后的样本
    address_sample = pd.read_excel(r'.\data\df_test.xlsx')
    address_sample['pro_hmm'] = ' '
    address_sample['city_hmm'] = ' '
    address_sample['area_hmm'] = ' '
    address_sample['detailed_hmm'] = ' '
    address_sample['route_state_hmm'] = ' '
    # 创建了一个新的列 'pro_hmm' city_hmm  area_hmm detailed_hmm route_state_hmm新列的初始值被设置为空格 ' '
    s_time = datetime.datetime.now()
    time_takes_total = {}
    for index, row in address_sample.iterrows():
        # .iterrows(): 这是 DataFrame 对象的一个方法,它返回一个迭代器,可以用于遍历 DataFrame 中的每一行
        # index 变量表示当前行的索引值,row 变量是一个 Pandas Series 对象,包含了该行的所有数据
        addr = row['address_'].strip().strip('\ufeff')
        # 'address_' 是这行数据中对应的地址列的列名
        # strip() 方法用于删除字符串两端的空白字符,包括空格、制表符、换行符等
        # \ufeff 是一个特殊的Unicode字符,称为"字节顺序标记"(Byte Order Mark, BOM)。
        # 有时候,从某些数据源导入的字符串数据可能会包含这个字符,需要将其删除。
        pro, city, area, detailed, *route_state, time_takes = cut(addr)
        # 切分
        address_sample.loc[index, 'pro_hmm'] = pro
        address_sample.loc[index, 'city_hmm'] = city
        address_sample.loc[index, 'area_hmm'] = area
        address_sample.loc[index, 'detailed_hmm'] = detailed
        address_sample.loc[index, 'route_state_hmm'] = str(route_state)  # 隐藏路径
        # 将index行的各个cut后对应的值的列填值

        time_takes_total = {key: (time_takes_total.get(key, 0) + value) for key, value in time_takes.items()}
        # 如果 key 存在于字典 time_takes_total 中,则返回该 key 对应的值。
        # 如果 key 不存在于字典中,则返回默认值 0。
        # 每个事件段的所花的时间
    e_time = datetime.datetime.now()
    times_total = (e_time - s_time).seconds
    print('总共{}条数据，共耗时:{}秒，平均每条{}秒。'.format(index + 1, times_total, times_total / (index + 1)))
    print({key: value for key, value in time_takes_total.items()})
    # 每条数据各个步骤的时间
    address_sample.to_excel(r'.\data\df_test_hmm.xlsx')
    # 结果转换为xlsx形式存储到表中

    # adr = '青岛路6号  一楼厂房'
    # pro, city, area, detailed,  *_ = cut(adr)
    # print(pro)
    # print(city)
    # print(area)
    # print(detailed)
