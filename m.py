import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("data/train.csv")

cate_cols = [
    # "地理区域",
    # "是否双频",
    # "是否翻新机",
    "手机网络功能",
    # "婚姻状况",
    # "信息库匹配",
    # "信用卡指示器",
    # "新手机用户",
    # "信用等级代码",
    "账户消费限额",
]



numeric_cols = [
    "当前手机价格",
    "家庭成人人数",
    "预计收入",
    "当前设备使用天数",
    "在职总月数",
    "家庭中唯一订阅者的数量",
    "家庭活跃用户数",
    "平均月费用",
    "每月平均使用分钟数",
    "平均超额使用分钟数",
    "平均超额费用",
    "平均语音费用",
    "数据超载的平均费用",
    "平均漫游呼叫数",
    "当月使用分钟数与前三个月平均值的百分比变化",
    "当月费用与前三个月平均值的百分比变化",
    "平均掉线语音呼叫数",
    "平均丢弃数据呼叫数",
    "平均占线语音呼叫数",
    "平均占线数据调用次数",
    "平均未接语音呼叫数",
    "未应答数据呼叫的平均次数",
    "尝试拨打的平均语音呼叫次数",
    "尝试数据调用的平均数",
    "平均接听语音电话数",
    "平均完成的语音呼叫数",
    "完成数据调用的平均数",
    "平均客户服务电话次数",
    "使用客户服务电话的平均分钟数",
    "一分钟内的平均呼入电话数",
    "平均三通电话数",
    "已完成语音通话的平均使用分钟数",
    "平均呼入和呼出高峰语音呼叫数",
    "平均峰值数据调用次数",
    "使用高峰语音通话的平均不完整分钟数",
    "平均非高峰语音呼叫数",
    "非高峰数据呼叫的平均数量",
    "平均掉线或占线呼叫数",
    "平均尝试调用次数",
    "平均已完成呼叫数",
    "平均呼叫转移呼叫数",
    "平均呼叫等待呼叫数",
    "客户生命周期内的总通话次数",
    "客户生命周期内的总使用分钟数",
    "客户生命周期内的总费用",
    "计费调整后的总费用",
    "计费调整后的总分钟数",
    "计费调整后的呼叫总数",
    "客户生命周期内平均月费用",
    "客户生命周期内的平均每月使用分钟数",
    "客户整个生命周期内的平均每月通话次数",
    "过去三个月的平均每月使用分钟数",
    "过去三个月的平均每月通话次数",
    "过去三个月的平均月费用",
    "过去六个月的平均每月使用分钟数",
    "过去六个月的平均每月通话次数",
    "过去六个月的平均月费用",
]
sns.set_style("darkgrid")
plt.style.use(['dark_background'])
numeric_cols = ["在职总月数",
# "家庭中唯一订阅者的数量",
"每月平均使用分钟数",
# "平均占线语音呼叫数",
"平均未接语音呼叫数", ]


print(cate_cols)
df_cate = pd.get_dummies(df[cate_cols].astype("str"))
# df_cate = df[cate_cols]
df_nume = df[numeric_cols]
df_label = df[["是否流失"]]
df = pd.concat([df_label, df_cate, df_nume], axis=1)
df = pd.concat([df_label, df_nume], axis=1)
print(df.shape)


# for c in cate_cols:
#     df_ = pd.DataFrame()
#     df_["1"] = df[df["是否流失"] == 1][c].value_counts()
#     df_["0"] = df[df["是否流失"] == 0][c].value_counts()
#     print(c)
#     sns.kdeplot(df_["1"], label="1", color="#B2A293", linewidth=2.5)
#     sns.kdeplot(df_["0"], label="0", color="#3C5840", linewidth=2.5)
#     plt.show()



y = df["是否流失"]
X = df.drop(columns=["是否流失"])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10056)
df_train = pd.concat([X_train, y_train], axis=1)
df_test = pd.concat([X_test, y_test], axis=1)
df_train.to_csv("dx_train_3cols.csv", index=False)
df_test.to_csv("dx_test_3cols.csv", index=False)
for i in df_train.columns:
    print(i)