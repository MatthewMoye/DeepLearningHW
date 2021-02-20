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


# Counter Parameters for model, function from https://github.com/zhangdan8962/DeepLearning
def parameter_counter():
    total_parameters = 0
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    return total_parameters


# Get each model info
def get_model_info(i):
    x = tf.placeholder(tf.float32, shape=[None, img_size * img_size], name="x")
    y_true = tf.placeholder(tf.float32, shape=[None, 10], name="y_true")
    y_true_cls = tf.argmax(y_true, dimension=1)

    # Model
    h1 = tf.layers.dense(inputs=x, name="h1", units=i, activation=tf.nn.relu)
    h2 = tf.layers.dense(inputs=h1, name="h2", units=i*2, activation=tf.nn.relu)
    h3 = tf.layers.dense(inputs=h2, name="h3", units=i, activation=tf.nn.relu)
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
    train_batch_size = 64
    for i in range(100):
        x_batch, y_true_batch = data.train.next_batch(train_batch_size)
        feed = {x: x_batch,y_true: y_true_batch}
        sess.run(optimizer, feed_dict=feed)
    los, acc, tst_los = sess.run([loss, accuracy, tst_loss], feed_dict=feed)
    test_acc = test_accuracy()
    return los, tst_los, acc, test_acc, parameter_counter()


loss_list = []
tst_loss_list = []
train_acc_list = []
test_acc_list = []
parameter_list = []

# Eval Models
for i in range(1,30):
    start = time.time()
    tf.reset_default_graph()
    with tf.Session() as sess:
        tf.set_random_seed(1)
        los, tst_los, tr_acc, tst_acc, param = get_model_info(i*10)
        loss_list.append(los)
        tst_loss_list.append(tst_los)
        train_acc_list.append(tr_acc)
        test_acc_list.append(tst_acc)
        parameter_list.append(param)
    print("Runtime:", time.time() - start)

fig, axs = plt.subplots(2, 1, figsize=(10, 12))
# Plot Loss
axs[0].scatter(parameter_list,loss_list,label="Train Loss")
axs[0].scatter(parameter_list,tst_loss_list,label="Test Loss")
# Plot accuracy
axs[1].scatter(parameter_list,train_acc_list,label="Train Accuracy")
axs[1].scatter(parameter_list,test_acc_list,label="Test Accuacy")

axs[0].set(title="Model Loss", ylabel="Loss", xlabel="Parameters")
axs[0].legend(loc="upper right")
axs[1].set(title="Model Accuracy", ylabel="Accuracy", xlabel="Parameters")
axs[1].legend(loc="lower right")
plt.savefig("params_vs_general.png")
