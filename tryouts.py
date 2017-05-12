
# from copy import deepcopy
# class A:
#     def __init__(self):
#         self.a = 5
#
# class B:
#     def __init__(self, z):
#         self.z = z
#     def foo(self):
#         self.z.a += 1
#
# AA = A()
# BB = B(AA)
# CC = deepcopy(BB)
# CC.z.a += 1
# print(BB.z.a, CC.z.a)
# BB.foo()
# print(AA.a, BB.z.a)

# import numpy as np
# def foo(b):
#     b+=1
#     print(b)
#     return b
# a = np.asarray([1,2,3,4,5])
# foo(a)
# print(a)

# import pandas as pd
# a = {("1", 0.2): 1, ("1", 0.5): 2, ("1", 1): 3,
#      ("2", 0.2): 1, ("2", 0.5): 2, ("2", 1): 3}
# b = pd.DataFrame.from_dict(a)

# import time
# loctime = time.localtime(time.time())


import numpy as np
import tensorflow as tf
import NNbuilds

# for _ in range(2):
#     np.random.seed(0)
#     print(np.random.rand(3))
for _ in range(2):
    tf.set_random_seed(0)
    a = tf.Variable(tf.random_normal([1, 3]), name="hjk")
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    print(a.eval(sess))

r = []
for _ in range(2):
    g = tf.Graph()
    with g.as_default():
        tf.set_random_seed(0)
        with tf.variable_scope("D{}".format(_)) as scope:
            a = tf.get_variable(name="a", shape=[1,3], initializer=tf.random_normal_initializer())
        sess = tf.Session(graph=g)
        sess.run(tf.global_variables_initializer())
        print(a.eval(sess))
    # tf.reset_default_graph()
    r.append((g, sess))
# for _ in range(2):
#     g = tf.Graph()
#     with g.as_default():
#         tf.set_random_seed(0)
#         a = tf.Variable(tf.random_normal([1, 3]), name="hjk")
#         sess = tf.Session()     # sess = tf.Session(graph=g)
#         sess.run(tf.global_variables_initializer())
#         print(a.eval(sess))
print("END")


    # sess.run(a)
    # print(a.eval(sess))


