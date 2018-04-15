import numpy as np

import tensorflow as tf

tf.set_random_seed(777)

x_data = [[73., 80., 75.],
          [93., 88., 93.],
          [89., 91., 90.],
          [96., 98., 100.],
          [73., 66., 70.]]
y_data = [[152.],
          [185.],
          [180.],
          [196.],
          [142.]]

class regression_model():
    def __init__(self):

        input_x = tf.placeholder(tf.float32,shape=[None,3],name='input')

        output_y=tf.placeholder(tf.float32,shape=[None,1])

        self.placeholder={'input':input_x,'output':output_y}

        weights= tf.get_variable('weights',shape=[3,1],dtype=tf.float32,initializer=tf.random_uniform_initializer(-0.01,0.01))

        bias = tf.get_variable('bias',shape=[1],dtype=tf.float32,initializer=tf.random_uniform_initializer(-0.01,0.01))



        result=tf.add(tf.matmul(input_x,weights),bias,name='result')

        cost=tf.square(result-output_y)

        loss=tf.reduce_mean(cost)




        train=tf.train.GradientDescentOptimizer(learning_rate=1e-5).minimize(loss)



        self.out ={'result':result,'loss':loss,'train':train}


def exe_func(model):
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for step in range(2001):
            out=sess.run(model.out,feed_dict={model.placeholder['input']:x_data,model.placeholder['output']:y_data})
            print("loss", out['loss'], "prediction", out['result'])

        saver.save(sess, '/Users/exepaul/Desktop/')


if __name__=='__main__':

    model=regression_model()

    exe_func(model)


















