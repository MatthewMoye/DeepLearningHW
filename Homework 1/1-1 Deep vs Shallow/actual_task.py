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
def get_model_info(model_num):
    x = tf.placeholder(tf.float32, shape=[None, img_size * img_size], name="x")
    x_image = tf.reshape(x, [-1, img_size, img_size, 1])
    y_true = tf.placeholder(tf.float32, shape=[None, 10], name="y_true")
    y_true_cls = tf.argmax(y_true, dimension=1)

    # Models
    if model_num == 2:
        net = tf.layers.conv2d(
            inputs=x_image,
            name="conv1",
            padding="same",
            filters=16,
            kernel_size=5,
            activation=tf.nn.relu,
        )
        net = tf.layers.max_pooling2d(inputs=net, pool_size=2, strides=2)
        net = tf.layers.conv2d(
            inputs=x_image,
            name="conv2",
            padding="same",
            filters=36,
            kernel_size=5,
            activation=tf.nn.relu,
        )
        net = tf.layers.max_pooling2d(inputs=net, pool_size=2, strides=2)
        net = tf.layers.flatten(net)
        net = tf.layers.dense(inputs=net, name="fc1", units=128, activation=tf.nn.relu)
        logits = tf.layers.dense(inputs=net, name="fc_out", units=10, activation=None)
    elif model_num == 1:
        net = tf.layers.conv2d(
            inputs=x_image,
            name="conv1",
            padding="same",
            filters=36,
            kernel_size=5,
            activation=tf.nn.relu,
        )
        net = tf.layers.max_pooling2d(inputs=net, pool_size=2, strides=2)
        net = tf.layers.flatten(net)
        net = tf.layers.dense(inputs=net, name="fc1", units=128, activation=tf.nn.relu)
        logits = tf.layers.dense(inputs=net, name="fc_out", units=10, activation=None)
    else:
        net = tf.layers.conv2d(
            inputs=x_image,
            name="conv1",
            padding="same",
            filters=16,
            kernel_size=5,
            activation=tf.nn.relu,
        )
        net = tf.layers.max_pooling2d(inputs=net, pool_size=2, strides=2)
        net = tf.layers.flatten(net)
        net = tf.layers.dense(inputs=net, name="fc1", units=128, activation=tf.nn.relu)
        logits = tf.layers.dense(inputs=net, name="fc_out", units=10, activation=None)

    y_pred = tf.nn.softmax(logits=logits)
    y_pred_cls = tf.argmax(y_pred, dimension=1)

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
        labels=y_true, logits=logits
    )
    loss = tf.reduce_mean(cross_entropy)
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)
    correct_prediction = tf.equal(y_pred_cls, y_true_cls)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    sess.run(tf.global_variables_initializer())

    loss_list = []
    train_acc_list = []
    test_acc_list = []

    # Calculate test accuracy, function from https://github.com/zhangdan8962/DeepLearning
    def test_accuracy():
        test_batch_size = 256
        num_test = len(data.test.images)
        cls_pred = np.zeros(shape=num_test, dtype=np.int64)
        i = 0
        while i < num_test:
            j = min(i + test_batch_size, num_test)
            cls_pred[i:j] = sess.run(
                y_pred_cls,
                feed_dict={
                    x: data.test.images[i:j, :],
                    y_true: data.test.labels[i:j, :],
                },
            )
            i = j
        cls_true = data.test.cls
        correct = cls_true == cls_pred
        correct_sum = correct.sum()
        acc = float(correct_sum) / num_test
        return acc

    # Train
    train_batch_size = 64
    for i in range(200):
        x_batch, y_true_batch = data.train.next_batch(train_batch_size)
        feed = {x: x_batch, y_true: y_true_batch}
        sess.run(optimizer, feed_dict=feed)
        if i % 5 == 0:
            los, acc = sess.run([loss, accuracy], feed_dict=feed)
            loss_list.append(los)
            train_acc_list.append(acc)
            test_acc_list.append(test_accuracy())

    return loss_list, train_acc_list, test_acc_list


# Plot loss and accuracy
def plot(num_models, loss_lists, train_acc_lists, test_acc_lists, filename):
    fig, axs = plt.subplots(2, 1, figsize=(10, 12))
    for i in range(num_models):
        name = "model" + str(i)
        # Plot Loss
        axs[0].plot(loss_lists[i], label=name)
        # Plot Accuracy
        axs[1].plot(train_acc_lists[i], label=name+'_train')
        axs[1].plot(test_acc_lists[i], label=name+'_test')
    axs[0].set(title="Model loss", ylabel="Loss", xlabel="Per 5 iterations")
    axs[0].legend(loc="upper right")
    axs[1].set(title="Ground Truth and Predictions", ylabel="Accuracy", xlabel="Per 5 iterations")
    axs[1].legend(loc="lower right")
    plt.savefig(filename)
    plt.clf()


# Evaluate Models
loss_lists = []
train_acc_lists = []
test_acc_lists = []
start = time.time()
model_count = range(3)
for i in model_count:
    tf.reset_default_graph()
    with tf.Session() as sess:
        tf.set_random_seed(1)
        loss_list, train_acc_list, test_acc_list = get_model_info(i)
        loss_lists.append(loss_list)
        train_acc_lists.append(train_acc_list)
        test_acc_lists.append(test_acc_list)
print("Runtime:", time.time() - start)

plot(len(model_count), loss_lists, train_acc_lists, test_acc_lists, "MNIST_models.png")