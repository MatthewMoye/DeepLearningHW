import time, math, random
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
def get_model_info(epochs):
    x = tf.placeholder(tf.float32, shape=[None, img_size * img_size], name="x")
    y_true = tf.placeholder(tf.float32, shape=[None, 10], name="y_true")
    y_true_cls = tf.argmax(y_true, dimension=1)

    # Model, Parameters - 8474
    h1 = tf.layers.dense(inputs=x, name="h1", units=10, activation=tf.nn.relu)
    h2 = tf.layers.dense(inputs=h1, name="h2", units=24, activation=tf.nn.relu)
    h3 = tf.layers.dense(inputs=h2, name="h3", units=10, activation=tf.nn.relu)
    out = tf.layers.dense(inputs=h3, name="out", units=10, activation=None)

    y_pred = tf.nn.softmax(logits=out)
    y_pred_cls = tf.argmax(y_pred, dimension=1)

    # train loss
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=out)
    loss = tf.reduce_mean(cross_entropy)

    # test loss
    cross_entropy2 = tf.nn.softmax_cross_entropy_with_logits(labels=y_pred, logits=out)
    tst_loss = tf.reduce_mean(cross_entropy2)

    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)
    correct_prediction = tf.equal(y_pred_cls, y_true_cls)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    sess.run(tf.global_variables_initializer())

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

    # Train model
    train_los_list = []
    test_los_list = []
    train_acc_list = []
    test_acc_list = []

    # Shuffle
    random.shuffle(data.train.labels)
    for i in range(epochs):
        feed = {x: data.train.images,y_true: data.train.labels}
        sess.run(optimizer, feed_dict=feed)
        los, acc, tst_los = sess.run([loss, accuracy, tst_loss], feed_dict=feed)
        train_los_list.append(los)
        test_los_list.append(tst_los)
        train_acc_list.append(acc)
        test_acc_list.append(test_accuracy())
    return train_los_list, test_los_list, train_acc_list, test_acc_list

epochs = 4000
start = time.time()
tf.reset_default_graph()
with tf.Session() as sess:
    tf.set_random_seed(1)
    los, tst_los, tr_acc, tst_acc = get_model_info(epochs)
print("Runtime:", time.time() - start)

fig, axs = plt.subplots(2, 1, figsize=(10, 12))
# Plot Loss
axs[0].scatter(range(epochs),los,label="Train Loss")
axs[0].scatter(range(epochs),tst_los,label="Test Loss")
# Plot accuracy
axs[1].scatter(range(epochs),tr_acc,label="Train Accuracy")
axs[1].scatter(range(epochs),tst_acc,label="Test Accuacy")

axs[0].set(title="Model Loss", ylabel="Loss", xlabel="Epoch")
axs[0].legend(loc="upper right")
axs[1].set(title="Model Accuracy", ylabel="Accuracy", xlabel="Epoch")
axs[1].legend(loc="lower right")
plt.savefig("fit_random_labels.png")