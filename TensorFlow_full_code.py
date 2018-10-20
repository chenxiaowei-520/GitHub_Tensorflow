"""
Please note, this code is only for python 3+. If you are using python 2+, please modify the code accordingly.
"""
from __future__ import print_function
import tensorflow as tf
import numpy as np ##科学计算模块，需要下载安装##

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

# create data
x_data = np.random.rand(100).astype(np.float32) ##自己创造编译的data##
y_data = x_data*0.1 + 0.3  ##预测0.1接近0.3##

### create tensorflow structure start ###
Weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0)) ##Weights 为矩阵形式##
biases = tf.Variable(tf.zeros([1]))

y = Weights*x_data + biases

loss = tf.reduce_mean(tf.square(y-y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss) 
### create tensorflow structure end ###

sess = tf.Session()
# tf.initialize_all_variables() no long valid from
# 2018-03-02 if using tensorflow >= 0.12
if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
        init = tf.initialize_all_variables()
else:
    init = tf.global_variables_initializer()
sess.run(init)

for step in range(201):
    sess.run(train)
    if step % 10 == 0:
       print(step, sess.run(Weights), sess.run(biases))
