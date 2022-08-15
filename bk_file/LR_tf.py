import numpy as np
import tensorflow as tf
from dataset.dataset import DataSet


class LR:
    def __init__(self, numeric_feature_len, cate_features_len, lr=0.1):
        self.numeric_feature_ph = tf.placeholder(dtype=tf.float32, shape=[None, numeric_feature_len])
        self.cate_features_pd = tf.placeholder(dtype=tf.int64, shape=[None, cate_features_len])

        self.label_ph = tf.placeholder(dtype=tf.float32, shape=[None, 1])

        self.numeric_weight = tf.Variable(np.ones([numeric_feature_len, 1]), dtype=tf.float32)
        self.bias = tf.Variable(0, dtype=tf.float32)
        self.yhat = tf.sigmoid(tf.matmul(self.numeric_feature_ph, self.numeric_weight) + self.bias)
        self.loss = tf.reduce_mean(tf.square(self.yhat - tf.squeeze(self.label_ph))) + tf.nn.l2_loss(self.bias)
        self.train_op = tf.train.GradientDescentOptimizer(lr).minimize(self.loss)


dataSet = DataSet("DY.tf_dataset")
tf_dataset = dataSet.read_TFRecord("trainDY.tf_dataset")
print(dataSet.cate_fea_num, dataSet.numeric_fea_num)
sess = tf.Session()
model = LR(dataSet.numeric_fea_num, dataSet.cate_fea_num)
init_op = tf.global_variables_initializer()
sess.run(init_op)
for epoch in range(100):
    total_batch = 0

    while True:
        try:
            rs = sess.run(tf_dataset)
            numeric_feature = rs["numeric_feature"]
            label = rs["label"]
            cate_features = rs["cate_features"]
            feed_dict = {
                model.numeric_feature_ph: numeric_feature,
                model.cate_features_pd: cate_features,
                model.label_ph: label
            }
            # print(numeric_feature)
            sess.run(model.train_op, feed_dict=feed_dict)
            loss, weight = sess.run([model.loss, model.bias], feed_dict=feed_dict)
            # loss, weight, _ = sess.run([model.loss, model.numeric_weight, model.train_op], feed_dict=feed_dict)
            print(loss)
            print(weight)
        except tf.errors.OutOfRangeError:
            break

