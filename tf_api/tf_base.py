import tensorflow as tf
import numpy as np
from neuralprophet import NeuralProphet

# 1. add_to_collection
# """ tf.add_to_collection() 把变量都放在一个list里面， get_collection 可以获取变量列表
#  add_to_collectio为Graph的一个方法，可以简单地认为Graph下维护了一个字典，key为name,value为list，
#  而add_to_collection就是把变量添加到对应key下的list中 add_to_collection(name,value)"""
# v1 = tf.get_variable(name="v1", shape=[1], initializer=tf.constant_initializer(30.0))
# v2 = tf.get_variable(name="v2", shape=[1], initializer=tf.constant_initializer(2.0))
# tf.add_to_collection("loss", v1)
# tf.add_to_collection("loss", v2)
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     print(sess.run(v1))
#     print(sess.run(v2))
#     print(tf.get_collection("loss"))
#     print(sess.run(tf.add_n(tf.get_collection("loss"))))


# # 2.argmax argmin
# a = np.random.random(size=(10, 5, 4))
# # b = tf.argmax(a)
# b = tf.argmin(a, axis=1)
# print(a)
# with tf.Session() as sess:
#     print(sess.run(b))
#     print(sess.run(b).shape)


# # 3. assert_variables_initialized
# w = tf.Variable(np.ones(shape=[10, 10]))
# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
# sess.run(tf.assert_variables_initialized([w]))


# 4. assign
# Variable使用assign方法更新变量后, 依然还是 Variable, 但如果使用类似+, 就会变成一个Tensor, 而不再是变量.
# Variable用于存储网络中的权重矩阵等变量，而Tensor更多的是中间结果等。
# a = tf.Variable(tf.constant(20))
# update = tf.assign(a, 40)
# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
# print(sess.run(a))
# print(sess.run(update))
# print(sess.run(a))

# a = tf.Variable(tf.constant(20))
# b = tf.Variable(tf.constant(50))
# update = tf.assign(a, 60)
# with tf.control_dependencies([update]):
#     c = tf.add(a, b)
# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
# print(sess.run(c))


# # 5. assign_add
# a = tf.Variable(tf.constant(10))
# b = tf.Variable(tf.constant(20))
# update = tf.assign_add(a, 10)
# with tf.control_dependencies([update]):
#     c = a + b
# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
# print(sess.run(c))


# # 6.tf.boolean_mask()
# # tensor = [0, 1, 2, 3]
# tensor = [[1, 2], [3, 4], [5, 6]]
# # mask = np.array([True, False, True, False])
# mask = np.array([True, False, True])
# out = tf.boolean_mask(tensor, mask)
# sess = tf.Session()
# print(sess.run(out))

# # 7. tf.clip_by_norm() tf.clip_by_value()
# """这里的clip_by_norm是指对梯度进行裁剪，通过控制梯度的最大范式，防止梯度爆炸的问题，是一种比较常用的梯度规约的方式"""
# weight = tf.Variable(tf.constant(3.), dtype=tf.float32)
# loss = 3. * tf.pow(weight, 2.) + tf.multiply(weight, 5.) + 10.
# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
# grads = optimizer.compute_gradients(loss)
# for i, (g, v) in enumerate(grads):
#     grads[i] = (tf.clip_by_norm(g, 5), v)
# train_op = optimizer.apply_gradients(grads)
# print(sess.run(loss))
# print(sess.run(grads))
# sess.run(train_op)
# print(sess.run(loss))
# print(sess.run(grads))


# # 8.tf.cond
# x = tf.constant(2)
# y = tf.constant(5)
# def f1(): return tf.multiply(x, 17)
# def f2(): return tf.add(y, 23)
# r_fn = tf.cond(tf.less(x, y), f1, f2)
# print(r_fn)
# sess = tf.Session()
# print(sess.run(r_fn))


# # 9.convert_to_tensor
# label_list = [[1, 2, 3, 4, 5], [6, 7, 1, 2, 3], [3, 2, 1, 5, 6]]
# out = tf.convert_to_tensor(label_list, tf.int32)
# sess = tf.Session()
# print(sess.run(out))


# # 10.fill
# a = tf.fill([3, 3], 100)
# print(a)
# sess = tf.Session()
# print(sess.run(a))


# # 11.gather
# weight = tf.Variable(np.ones(shape=[10]))
# out = tf.gather(weight, [3, 2, 1])
# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
# print(sess.run(out))

