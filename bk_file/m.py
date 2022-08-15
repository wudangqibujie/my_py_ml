import numpy as np
import tensorflow as tf
from dataset.dataset import DataSet


    
embedding_dim = 1
embedding_size = 91
dataSet = DataSet("DY.tf_dataset")
print(dataSet.cate_fea_num, dataSet.numeric_fea_num)
lr = 0.1
numeric_feature_len = dataSet.numeric_fea_num
cate_features_len = dataSet.cate_fea_num
numeric_feature_ph = tf.placeholder(dtype=tf.float32, shape=[None, numeric_feature_len])
cate_features_pd = tf.placeholder(dtype=tf.int32, shape=[None, cate_features_len])

embedding_table = tf.get_variable(name="embedding", shape=[embedding_size, embedding_dim], dtype=tf.float32, initializer=tf.truncated_normal_initializer(0.02))

numeric_feature_out = tf.layers.BatchNormalization()(numeric_feature_ph)

# cate_out = tf.nn.embedding_lookup(embedding_table, tf.expand_dims(cate_features_pd, axis=-1))
cate_out = tf.nn.embedding_lookup(embedding_table, cate_features_pd)

x_feature = tf.concat([numeric_feature_out, tf.squeeze(cate_out)], axis=-1)
label_ph = tf.placeholder(dtype=tf.float32, shape=[None, 1])
weight = tf.Variable(np.random.random([numeric_feature_len + cate_features_len, 1]), dtype=tf.float32)

bias = tf.Variable(0, dtype=tf.float32)
yhat = tf.sigmoid(tf.matmul(x_feature, weight) + bias)
loss = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(yhat, tf.squeeze(label_ph))) + tf.nn.l2_loss(weight) + tf.nn.l2_loss(embedding_table)
train_op = tf.train.AdamOptimizer(lr).minimize(loss)


sess = tf.Session()
init_op = tf.global_variables_initializer()
sess.run(init_op)
loss_history = []

for epoch in range(100):
    tf_dataset = dataSet.read_TFRecord("trainDY.tf_dataset")
    tnt_loss = 0
    tnt_size = 0
    while True:
        try:
            rs = sess.run(tf_dataset)
            numeric_feature = rs["numeric_feature"]
            label = rs["label"]
            batch_size = numeric_feature.shape[0]
            cate_features = rs["cate_features"]
            feed_dict = {
                numeric_feature_ph: numeric_feature,
                cate_features_pd: cate_features,
                label_ph: label
            }
            # print(numeric_feature)
            sess.run(train_op, feed_dict=feed_dict)
            loss_val = sess.run(loss, feed_dict=feed_dict)
            # loss, weight, _ = sess.run([loss, numeric_weight, train_op], feed_dict=feed_dict)
            # print("********* 2 ********")
            # print(loss_val)
            # print(sess.run(numeric_weight))
            tnt_loss += (loss_val * batch_size)
            tnt_size += batch_size
        except tf.errors.OutOfRangeError:
            print("over")
            break

    print(f"epoch: {epoch}, {tnt_loss / tnt_size}")
    loss_history.append(tnt_loss / tnt_size)


import matplotlib.pyplot as plt
plt.plot(loss_history)
plt.show()