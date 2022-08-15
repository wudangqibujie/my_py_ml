# from sklearn.feature_extraction.text import TfidfVectorizer
# tv = TfidfVectorizer(min_df=100, smooth_idf=True)
# import os
# import jieba
# # train = ["Chinese Beijing Chinese", "Chinese Chinese Shanghai", "Chinese Macao", "Tokyo Japan Chinese"]
# # tv_fit = tv.fit_transform(train)
# # print(tv.transform(["Chinese Beijing Chinese"]).toarray())
#
# base_dir = "../data/cnews"
#
# with open(os.path.join(base_dir, "stopwords.txt"), encoding="utf-8") as f:
#     staopwords = [i.strip() for i in f.readlines()]
#
#
# def corpus_inter(file):
#     stop_flag = 0
#     with open(file, encoding="utf-8") as f:
#         for i in f:
#             stop_flag += 1
#             if stop_flag >= 500: break
#             text = i.strip().split("\t")[1]
#             texts = [i for i in jieba.cut(text) if i not in staopwords]
#             text = " ".join(texts)
#             yield text
#
# corpus = corpus_inter("../data/cnews/cnews.train.txt")
# tv.fit(corpus)
#
# t = ["新浪 体育讯 北京 时间 11 月 日 消息 休斯敦 火箭 主场", "主场"]
# trans = tv.transform(t)
# print(trans)
# print(trans.toarray().shape)
# print(tv.get_feature_names())
# print(len(tv.get_feature_names()))
#

import pandas as pd
df = pd.read_csv("data/train.csv", encoding="utf-8")
print(df.columns)


