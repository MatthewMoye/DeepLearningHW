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
def get_model_info(model_num, batch_size=64):
    x = tf.placeholder(tf.float32, shape=[None, img_size * img_size], name="x")
    y_true = tf.placeholder(tf.float32, shape=[None, 10], name="y_true")
    y_true_cls = tf.argmax(y_true, dimension=1)

    # Model
    if model_num == 5:
        h1 = tf.layers.dense(x, 128, name="h1", activation=tf.nn.relu)
        h2 = tf.layers.dense(h1, 64, activation=tf.nn.relu)
        h3 = tf.layers.dense(h2, 64, activation=tf.nn.relu)
        h4 = tf.layers.dense(h3, 32, activation=tf.nn.relu)
        h5 = tf.layers.dense(h4, 32, activation=tf.nn.relu)
        h6 = tf.layers.dense(h5, 16, activation=tf.nn.relu)
        h7 = tf.layers.dense(h6, 16, activation=tf.nn.relu)
        out = tf.layers.dense(h7, 10, name="out")
    elif model_num == 4:
        h1 = tf.layers.dense(x, 128, name="h1", activation=tf.nn.relu)
        h2 = tf.layers.dense(h1, 64, activation=tf.nn.relu)
        h3 = tf.layers.dense(h2, 32, activation=tf.nn.relu)
        h4 = tf.layers.dense(h3, 32, activation=tf.nn.relu)
        h5 = tf.layers.dense(h4, 16, activation=tf.nn.relu)
        out = tf.layers.dense(h5, 10, name="out")
    elif model_num == 3:
        h1 = tf.layers.dense(x, 128, name="h1", activation=tf.nn.relu)
        h2 = tf.layers.dense(h1, 64, activation=tf.nn.relu)
        h3 = tf.layers.dense(h2, 32, activation=tf.nn.relu)
        h4 = tf.layers.dense(h3, 16, activation=tf.nn.relu)
        out = tf.layers.dense(h4, 10, name="out")
    elif model_num == 2:
        h1 = tf.layers.dense(x, 128, name="h1", activation=tf.nn.relu)
        h2 = tf.layers.dense(h1, 32, activation=tf.nn.relu)
        h3 = tf.layers.dense(h2, 32, activation=tf.nn.relu)
        h4 = tf.layers.dense(h3, 16, activation=tf.nn.relu)
        out = tf.layers.dense(h4, 10, name="out")
    elif model_num == 1:
        h1 = tf.layers.dense(x, 128, name="h1", activation=tf.nn.relu)
        h2 = tf.layers.dense(h1, 64, activation=tf.nn.relu)
        h3 = tf.layers.dense(h2, 32, activation=tf.nn.relu)
        out = tf.layers.dense(h3, 10, name="out")
    else:
        h1 = tf.layers.dense(x, 200, name="h1", activation=tf.nn.relu)
        out = tf.layers.dense(h1, 10, name="out")

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

    # Get weight, function from https://github.com/zhangdan8962/DeepLearning
    def get_weights_variable(layer_name):
        with tf.variable_scope(layer_name, reuse=True):
            variable = tf.get_variable('kernel')
        return variable
    weights_l1 = get_weights_variable(layer_name='h1')

    grads = tf.gradients(loss, weights_l1)[0]
    
    # Train model
    for i in range(1000):
        x_batch, y_true_batch = data.train.next_batch(batch_size)
        feed = {x: x_batch,y_true: y_true_batch}
        sess.run(optimizer, feed_dict=feed)
    los, acc, tst_los, grads_vals = sess.run([loss, accuracy, tst_loss, grads], feed_dict=feed)
    sensitivity = np.linalg.norm(np.asarray(grads_vals), ord="fro")
    test_acc = test_accuracy()
    
    return los, tst_los, acc, test_acc, sensitivity


def plots(batch_sizes, loss_list, tst_loss_list, train_acc_list, test_acc_list, sensitivity_list, i):
    fig, axs = plt.subplots(3, 1, figsize=(10, 18))
    # Plot Loss
    axs[0].plot(batch_sizes,loss_list,label="Train Loss")
    axs[0].plot(batch_sizes,tst_loss_list,label="Test Loss")
    # Plot accuracy
    axs[1].plot(batch_sizes,train_acc_list,label="Train Accuracy")
    axs[1].plot(batch_sizes,test_acc_list,label="Test Accuacy")
    # Plot Sensitivity
    axs[2].plot(batch_sizes,sensitivity_list)
    # Plot settings
    axs[0].set(title="Loss vs Batch Size", ylabel="Loss", xlabel="Batch Size", xscale="log")
    axs[0].legend(loc="upper right")
    axs[1].set(title="Accuracy vs Batch Size", ylabel="Accuracy", xlabel="Batch Size", xscale="log")
    axs[1].legend(loc="upper right")
    axs[2].set(title="Sensitivity vs Batch Size", ylabel="Sensitivity", xlabel="Batch Size", xscale="log")

    plt.savefig("flat_vs_gen_model"+str(i)+".png")
    plt.clf()

# Eval Models
batch_sizes = [32,64,128,256,512,1024,2048,4096,8192]
for i in range(6):
    start = time.time()
    loss_list = []
    tst_loss_list = []
    train_acc_list = []
    test_acc_list = []
    sensitivity_list = []
    for j in batch_sizes:
        tf.reset_default_graph()
        with tf.Session() as sess:
            tf.set_random_seed(1)
            los, tst_los, tr_acc, tst_acc, sens = get_model_info(i)
            loss_list.append(los)
            tst_loss_list.append(tst_los)
            train_acc_list.append(tr_acc)
            test_acc_list.append(tst_acc)
            sensitivity_list.append(sens)
    plots(batch_sizes, loss_list, tst_loss_list, train_acc_list, test_acc_list, sensitivity_list, i)
    print("Runtime per model:", time.time() - start)
