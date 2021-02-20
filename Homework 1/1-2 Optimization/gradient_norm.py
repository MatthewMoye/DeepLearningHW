import time, math
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
np.random.seed(1)
tf.set_random_seed(1)

data = input_data.read_data_sets("data/MNIST/", one_hot=True)
data.test.cls = np.argmax(data.test.labels, axis=1)
img_size = 28

# Get each model info
def get_model_info():
    x = tf.placeholder(tf.float32, shape=[None, img_size * img_size], name="x")
    y_true = tf.placeholder(tf.float32, shape=[None, 10], name="y_true")
    y_true_cls = tf.argmax(y_true, dimension=1)

    # Model
    h1 = tf.layers.dense(inputs=x, name="h1", units=128, activation=tf.nn.relu)
    h2 = tf.layers.dense(inputs=h1, name="h2", units=64, activation=tf.nn.relu)
    h3 = tf.layers.dense(inputs=h2, name="h3", units=32, activation=tf.nn.relu)
    out = tf.layers.dense(inputs=h3, name="out", units=10, activation=None)

    y_pred = tf.nn.softmax(logits=out)
    y_pred_cls = tf.argmax(y_pred, dimension=1)

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=out)
    loss = tf.reduce_mean(cross_entropy)
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)
    correct_prediction = tf.equal(y_pred_cls, y_true_cls)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    sess.run(tf.global_variables_initializer())

    # Get weight, function from https://github.com/zhangdan8962/DeepLearning
    def get_weights_variable(layer_name):
        with tf.variable_scope(layer_name, reuse=True):
            variable = tf.get_variable('kernel')
        return variable
    weights_out = get_weights_variable(layer_name='out')

    loss_list = []
    gradient_list = []

    train_batch_size = 64
    grads = tf.gradients(loss, weights_out)[0]
    total_iterations = 0
    for i in range(20000):
        x_batch, y_true_batch = data.train.next_batch(train_batch_size)
        feed = {x: x_batch,y_true: y_true_batch}
        sess.run(optimizer, feed_dict=feed)
        
        los, grads_vals = sess.run([loss, grads], feed_dict=feed)
        gradient_list.append(np.linalg.norm(np.asarray(grads_vals), ord=2))
        loss_list.append(los)
    
    fig, axs = plt.subplots(2, 1, figsize=(10, 12))
    # Plot Loss
    axs[0].plot(gradient_list)
    # Plot grad
    axs[1].plot(loss_list)
    axs[0].set(title="Gradient norm to iterations", ylabel="gradient", xlabel="iteration")
    axs[1].set(title="Loss to iterations", ylabel="Loss", xlabel="iteration")
    plt.savefig("gradient_norm_vs_loss.png")


# Evaluate Model
start = time.time()
tf.reset_default_graph()
with tf.Session() as sess:
    tf.set_random_seed(1)
    get_model_info()
print("Runtime:", time.time() - start)
