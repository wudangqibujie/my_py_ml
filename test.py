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

# import pandas as pd
# df = pd.read_csv("data/train.csv", encoding="utf-8")
# print(df.columns)
#
# import tensorflow as tf
# a = tf.constant([10, 4, 3, 6, 7, 1, 2, 5])
# a = tf.as_string(a)
# out = tf.string_to_hash_bucket_fast(a, num_buckets=4)
# sess = tf.Session()
# print(sess.run(out))

# f = open("data/train.csv", encoding="utf-8")
# for i in f:
#     print(i.strip())

# from dataset.dx_dataset import DXDataset
# dxDataset = DXDataset("data/train.csv")
# print(dxDataset.columns_name)
#
# for features, label in dxDataset.read_iterator():
#     print(features)
#     print(label)

import pandas as pd
from sklearn.preprocessing import LabelEncoder
# #
df = pd.read_csv("data/train.csv", encoding="utf-8")
from dataset.dx_dataset import CATE_COLS, NUMERIC_COLS
# #
enc = LabelEncoder()
#
for ix in range(len(CATE_COLS)):
    df[CATE_COLS[ix]] = enc.fit_transform(df[CATE_COLS[ix]])
df_max_id = df[CATE_COLS].max(axis=0)
print(df_max_id)
df_max_id_columns = df_max_id.index
df_max_id_values = df_max_id.values
ls_val = df_max_id_values[0]
print()
for ix, i in enumerate(df_max_id_columns):
    if ix == 0: continue
    df[i] = df[i] + ls_val + 1
    ls_val = df[i].max()
print(df[CATE_COLS].max(axis=0))

for n in NUMERIC_COLS + CATE_COLS:
    df[n] = (df[n] - df[n].mean()) / df[n].std()
print(df[CATE_COLS[0]].values)

# df.to_csv("../data/dx_train.csv", index=False)

df_cate = df[CATE_COLS].values
df_numeric = df[NUMERIC_COLS].values
df_label = df["是否流失"].values
print(df_label)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
model = LogisticRegression(max_iter=500, solver="sag")
model.fit(df[CATE_COLS+NUMERIC_COLS].values, df_label)

# import random
# import collections
# import tensorflow as tf
#
#
# def _create_int_feature(value):
#     return tf.train.Feature(int64_list=tf.train.Int64List(value=list(value)))
#
#
# def _create_float_feature(value):
#     return tf.train.Feature(float_list=tf.train.FloatList(value=list(value)))
#
#
# def create_features(cate_feats, num_feats, label):
#     features = collections.OrderedDict()
#     features["num_features"] = _create_float_feature(num_feats)
#     features["cate_features"] = _create_int_feature(cate_feats)
#     features["label"] = _create_int_feature([label])
#     return tf.train.Example(features=tf.train.Features(feature=features))
#
#
# train_writer = tf.python_io.TFRecordWriter("../data/dx_train.tf_record")
# val_writer = tf.python_io.TFRecordWriter("../data/dx_val.tf_record")
#
# for ix, i in enumerate(df_cate):
#     tf_axample = create_features(i, df_numeric[ix], df_label[ix])
#     # print(tf_axample)
#     if random.random() < 0.8:
#         train_writer.write(tf_axample.SerializeToString())
#     else:
#         val_writer.write(tf_axample.SerializeToString())
# import pandas as pd
# df = pd.read_csv("../data/dx_train.csv")
# for c in CATE_COLS:
#     print(df[c].max())


# import tensorflow as tf
#
#
# def parse(record):
#     name_to_features = {
#         "num_features": tf.FixedLenFeature([len(NUMERIC_COLS)], tf.float32),
#         "cate_features": tf.FixedLenFeature([len(CATE_COLS)], tf.int64),
#         "label": tf.FixedLenFeature([1], tf.int64)
#     }
#     features = tf.parse_single_example(record, name_to_features)
#     return features
#
#
# def readDataset(tf_record_file, batch_size):
#     dataset = tf.data.TFRecordDataset(tf_record_file)
#     dataset = dataset.map(parse).shuffle(100).batch(batch_size)
#     dataset = dataset.make_one_shot_iterator().get_next()
#     return dataset
#
#
# from logistic_regression.logistic_regression import LogisticRegressionV2
#
# train_dataset = readDataset("../data/dx_train.tf_record", 64)
# val_dataset = readDataset("../data/dx_val.tf_record", 64)
#
# model = LogisticRegressionV2(len(CATE_COLS), len(NUMERIC_COLS), 100)
# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
#
# for epoch in range(100):
#     while True:
#         try:
#             read_data = sess.run(train_dataset)
#             input_num, input_cate, label = read_data["num_features"], read_data["cate_features"], read_data["label"]
#             # print(input_num.shape)
#             out, emb, _ = sess.run([model.cat_out, model.num_out, model.train_op], feed_dict={model.input_num_ph: input_num, model.input_cate_pd: input_cate, model.label_ph: label})
#             # sess.run(model.train_op, feed_dict={model.input_num_ph: input_num, model.input_cate_pd: input_cate, model.label_ph: label})
#         except tf.errors.OutOfRangeError:
#             break
#     print(epoch)
#     print(out)
#     print(emb)
    # print(emb)