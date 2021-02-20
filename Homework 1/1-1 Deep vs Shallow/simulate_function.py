import time, math
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

tf.logging.set_verbosity(tf.logging.ERROR)
np.random.seed(1)


# Counter Parameters for model, function from https://github.com/zhangdan8962/DeepLearning
def parameter_counter():
    total_parameters = 0
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    print(total_parameters)


# Get each model info
def get_model_info(model_num, X, Y):
    # 601 params
    if model_num == 2:
        h1 = tf.layers.dense(x, 6, activation=tf.nn.relu)
        h2 = tf.layers.dense(h1, 9, activation=tf.nn.relu)
        h3 = tf.layers.dense(h2, 10, activation=tf.nn.relu)
        h4 = tf.layers.dense(h3, 12, activation=tf.nn.relu)
        h5 = tf.layers.dense(h4, 10, activation=tf.nn.relu)
        h6 = tf.layers.dense(h5, 9, activation=tf.nn.relu)
        h7 = tf.layers.dense(h6, 6, activation=tf.nn.relu)
        out = tf.layers.dense(h7, 1)
    # 601 params
    elif model_num == 1:
        h1 = tf.layers.dense(x, 10, activation=tf.nn.relu)
        h2 = tf.layers.dense(h1, 18, activation=tf.nn.relu)
        h3 = tf.layers.dense(h2, 13, activation=tf.nn.relu)
        h4 = tf.layers.dense(h3, 9, activation=tf.nn.relu)
        out = tf.layers.dense(h4, 1)
    # 603 params
    else:
        h1 = tf.layers.dense(x, 200, activation=tf.nn.relu)
        out = tf.layers.dense(h1, 1)
    parameter_counter()

    loss = tf.losses.mean_squared_error(y, out)
    optimizer = tf.train.AdamOptimizer(learning_rate=0.0005).minimize(loss)

    loss_list = []
    sess.run(tf.global_variables_initializer())
    for i in range(0, 20000):
        _, loss_val = sess.run([optimizer, loss], feed_dict={x: X, y: Y})
        loss_list.append(loss_val)
    YP = sess.run(out, feed_dict={x: X})

    return loss_list, YP


# Plot loss and grouth truth
def plot(model_count, loss_lists, YP_lists, X, Y, function, filename):
    fig, axs = plt.subplots(2, 1, figsize=(10, 12))
    for i in model_count:
        name = "model" + str(i)
        # Plot loss
        axs[0].plot(loss_lists[i], label=name)
        # Plot ground truth and predictions
        axs[1].plot(X, YP_lists[i], label=name)
    axs[0].set(title="Model loss", ylabel="Loss", xlabel="Epoch", yscale="log")
    axs[0].legend(loc="upper right")
    axs[1].set(title="Ground Truth and Predictions", ylabel="Y", xlabel="X")
    axs[1].plot(X, Y, label=function, color="y")
    axs[1].legend(loc="upper right")
    plt.savefig(filename)
    plt.clf()


# Data
X = np.expand_dims(np.arange(0.0, 2.0, 0.01), 1)
Y1 = np.sinc(X)
Y2 = np.sign(np.sin(5*X))

# Evaluate Models
start = time.time()
for Y in [Y1, Y2]:
    loss_lists = []
    YP_lists = []
    model_count = range(3)
    for i in model_count:
        tf.reset_default_graph()
        with tf.Session() as sess:
            tf.set_random_seed(1)
            x = tf.placeholder(tf.float64, [200, 1], name="x")
            y = tf.placeholder(tf.float64, [200, 1], name="y")
            loss_list, YP = get_model_info(i, X, Y)
            YP_lists.append(YP)
            loss_lists.append(loss_list)
    if np.array_equal(Y,Y1):
        plot(model_count, loss_lists, YP_lists, X, Y, "sin(πx)/(πx)", "dnn_sinc.png")
    else:
        plot(model_count, loss_lists, YP_lists, X, Y, "sgn(sin(5πx))", "dnn_sgn_sin.png")
print("Runtime:", time.time() - start)
