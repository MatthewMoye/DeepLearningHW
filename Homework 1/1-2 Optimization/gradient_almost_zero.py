import time, math
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
np.random.seed(1)

# Get each model info
def get_model_info(X,Y):
    x = tf.placeholder(tf.float32, shape=[2000,1], name="x")
    y_true = tf.placeholder(tf.float32, shape=[2000,1], name="y_true")
    y_true_cls = tf.argmax(y_true, dimension=1)

    # Model
    h1 = tf.layers.dense(x, 30, activation=tf.nn.relu)
    h2 = tf.layers.dense(x, 20, activation=tf.nn.relu)
    out = tf.layers.dense(inputs=h2, name="out", units=1, activation=None)

    y_pred = tf.nn.softmax(logits=out)
    y_pred_cls = tf.argmax(y_pred)

    loss = tf.losses.mean_squared_error(y_true, out)
    optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss)
    correct_prediction = tf.equal(y_pred_cls, y_true_cls)

    sess.run(tf.global_variables_initializer())

    # Get weight, function from https://github.com/zhangdan8962/DeepLearning
    def get_weights_variable(layer_name):
        with tf.variable_scope(layer_name, reuse=True):
            variable = tf.get_variable('kernel')
        return variable
    weight_layer1 = []

    # Set up gradient norm loss
    gradient_list = [0]
    grad_norm = tf.Variable(1.0)
    grad_norm_run = tf.assign(grad_norm, gradient_list[-1])
    optimizer2 = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(tf.convert_to_tensor(grad_norm))

    sess.run(tf.global_variables_initializer())

    gradient_list = []

    grads = tf.gradients(loss, get_weights_variable(layer_name='out'))[0]
    hessian = tf.reduce_sum(tf.hessians(loss, get_weights_variable(layer_name='out'))[0], axis = 2)

    # Do one run of data before replacing loss with grad_norm
    feed = {x: X, y_true: Y}
    sess.run(optimizer, feed_dict=feed)
    los, grads_vals = sess.run([loss, grads], feed_dict=feed)

    # Train
    min_grad = 10
    hess_vals = 0
    loss_hess = 1
    feed = {x: X, y_true: Y}
    for i in range(2000):
        sess.run(optimizer2, feed_dict=feed)
        sess.run(grad_norm_run)
        grads_vals = sess.run([grads], feed_dict=feed)
        gradient_list.append(np.linalg.norm(np.asarray(grads_vals)))
        if gradient_list[-1] < min_grad:
            min_grad = gradient_list[-1]
            hess_vals = sess.run([hessian], feed_dict=feed)
            loss_hess = los
    val = np.linalg.eig(hess_vals)[0]
    min_ratio = len(val[val>0])/(val.shape[1])
    return loss_hess, min_ratio

# Data
X = np.expand_dims(np.arange(0.0, 2.0, 0.001), 1)
Y = np.sinc(X)

# Eval model
min_ratio_list = []
grad_list = []
start = time.time()
for i in range(100):
    tf.reset_default_graph()
    with tf.Session() as sess:
        grad, min_ratio = get_model_info(X,Y)
        grad_list.append(grad)
        min_ratio_list.append(min_ratio)
print("Runtime:", time.time() - start)

fig, axs = plt.subplots(1, 1, figsize=(10, 6))
# Plot Loss
axs.scatter(min_ratio_list,grad_list)
axs.set(title="Loss to Minimum Ratio 100 Trainings", ylabel="loss", xlabel="min ratio")
plt.savefig("loss_to_min_ratio.png")