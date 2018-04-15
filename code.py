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

        input_x = tf.placeholder(tf.float32,shape=[None,3])

        output_y=tf.placeholder(tf.float32,shape=[None,1])

        self.placeholder={'input':input_x,'output':output_y}

        # weights= tf.get_variable('weights',shape=[3,1],dtype=tf.float32,initializer=tf.random_uniform_initializer(-0.01,0.01))
        #
        # bias = tf.get_variable('bias',shape=[1],dtype=tf.float32,initializer=tf.random_uniform_initializer(-0.01,0.01))

        weights = tf.Variable(tf.random_normal([3, 1]), name='weight')
        bias = tf.Variable(tf.random_normal([1]), name='bias')

        result=tf.matmul(input_x,weights) + bias

        cost=tf.square(result-output_y)

        loss=tf.reduce_mean(cost)




        train=tf.train.GradientDescentOptimizer(learning_rate=1e-5).minimize(loss)



        self.out ={'result':result,'loss':loss,'train':train}


def exe_func(model):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for step in range(2001):
            out=sess.run(model.out,feed_dict={model.placeholder['input']:x_data,model.placeholder['output']:y_data})
            print("loss", out['loss'], "prediction", out['result'])



if __name__=='__main__':

    model=regression_model()

    exe_func(model)




#when using tf.Variable
#weights = tf.Variable(tf.random_normal([3, 1]), name='weight')
#bias = tf.Variable(tf.random_normal([1]), name='bias')

output:
loss 22655.951 prediction [[22.048063]
 [21.619787]
 [24.096693]
 [22.293005]
 [18.633902]]
loss 7105.457 prediction [[80.82241]
 [92.26364]
 [93.70251]
 [98.09218]
 [72.51759]]
 
...........
...........
...........

loss 3.1811848 prediction [[154.36069]
 [182.95021]
 [181.85094]
 [194.35553]
 [142.03456]]
loss 3.1804314 prediction [[154.36021]
 [182.95052]
 [181.8508 ]
 [194.35547]
 [142.03491]]
loss 3.179638 prediction [[154.35976]
 [182.95085]
 [181.85066]
 [194.35545]
 [142.0353 ]]
loss 3.1788766 prediction [[154.3593 ]
 [182.95117]
 [181.85052]
 [194.35541]
 [142.03566]]

Process finished with exit code 0





# Now when using with tf.get_variable()

# weights= tf.get_variable('weights',shape=[3,1],dtype=tf.float32,initializer=tf.random_uniform_initializer(-0.01,0.01))
# bias = tf.get_variable('bias',shape=[1],dtype=tf.float32,initializer=tf.random_uniform_initializer(-0.01,0.01))


loss 29856.605 prediction [[-0.5352072 ]
 [-0.5790558 ]
 [-0.6041024 ]
 [-0.65316194]
 [-0.42943165]]
loss 9359.545 prediction [[66.94317 ]
 [80.52572 ]
 [79.3094  ]
 [86.37067 ]
 [61.433266]]
loss 2934.8047 prediction [[104.72164]
 [125.93343]
 [124.04998]
 [135.09207]
 [ 96.06809]]
 
 ...........
 ...........
 ...........
 
 
 loss 0.73736966 prediction [[152.12337]
 [184.1447 ]
 [180.76556]
 [196.88792]
 [140.74876]]
loss 0.7371322 prediction [[152.12311]
 [184.14487]
 [180.76549]
 [196.88786]
 [140.74901]]
loss 0.73689765 prediction [[152.12286]
 [184.14502]
 [180.76541]
 [196.88777]
 [140.74924]]
loss 0.7366613 prediction [[152.12263]
 [184.1452 ]
 [180.76535]
 [196.88771]
 [140.74948]]

Process finished with exit code 0













