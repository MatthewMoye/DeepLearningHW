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


# Get weights from two models
def get_model_weights(batch_size, rate):
    x = tf.placeholder(tf.float32, shape=[None, img_size * img_size], name="x")
    y_true = tf.placeholder(tf.float32, shape=[None, 10], name="y_true")
    y_true_cls = tf.argmax(y_true, dimension=1)
    
    # Model
    h1 = tf.layers.dense(x, 128, name="h1", activation=tf.nn.relu)
    out = tf.layers.dense(h1, 10, name="out")

    y_pred = tf.nn.softmax(logits=out)
    y_pred_cls = tf.argmax(y_pred, dimension=1)

    # train loss
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=out)
    loss = tf.reduce_mean(cross_entropy)

    optimizer = tf.train.AdamOptimizer(learning_rate=rate).minimize(loss)
    correct_prediction = tf.equal(y_pred_cls, y_true_cls)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    sess.run(tf.global_variables_initializer())

    # Get weight, function from https://github.com/zhangdan8962/DeepLearning
    def get_weights_variable(layer_name):
        with tf.variable_scope(layer_name, reuse=True):
            variable = tf.get_variable('kernel')
        return variable
    weights_l1 = get_weights_variable(layer_name='h1')
    weight_out = get_weights_variable(layer_name='out')
    
    # Train model
    for i in range(100):
        x_batch, y_true_batch = data.train.next_batch(batch_size)
        feed = {x: x_batch,y_true: y_true_batch}
        sess.run(optimizer, feed_dict=feed)
    wl1, wout = sess.run([weights_l1,weight_out], feed_dict=feed)
    return wl1, wout


# Get acc and loss for model with predefined weights
def get_model_info(batch_size, rate, weight1, weightout):
    x = tf.placeholder(tf.float32, shape=[None, img_size * img_size], name="x")
    y_true = tf.placeholder(tf.float32, shape=[None, 10], name="y_true")
    y_true_cls = tf.argmax(y_true, dimension=1)

    # Model
    h1_w = tf.constant_initializer(weight1)
    out_w = tf.constant_initializer(weightout)
    print(h1_w)
    h1 = tf.layers.dense(x, 128, name="h1", activation=tf.nn.relu, kernel_initializer=h1_w)
    out = tf.layers.dense(h1, 10, name="out", kernel_initializer=out_w)

    #######^^^^ Change model here

    y_pred = tf.nn.softmax(logits=out)
    y_pred_cls = tf.argmax(y_pred, dimension=1)

    # train loss
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=out)
    loss = tf.reduce_mean(cross_entropy)

    # test loss
    cross_entropy2 = tf.nn.softmax_cross_entropy_with_logits(labels=y_pred, logits=out)
    tst_loss = tf.reduce_mean(cross_entropy2)

    optimizer = tf.train.AdamOptimizer(learning_rate=rate).minimize(loss)
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
            feed = {x: data.test.images[i:j, :],y_true: data.test.labels[i:j, :]}
            cls_pred[i:j] = sess.run(y_pred_cls,feed_dict=feed)
            i = j
        cls_true = data.test.cls
        correct = cls_true == cls_pred
        correct_sum = correct.sum()
        acc = float(correct_sum) / num_test
        return acc
    
    # Train model
    for i in range(100):
        x_batch, y_true_batch = data.train.next_batch(batch_size)
        feed = {x: x_batch,y_true: y_true_batch}
        sess.run(optimizer, feed_dict=feed)
    los, acc, tst_los = sess.run([loss, accuracy, tst_loss], feed_dict=feed)
    test_acc = test_accuracy()
    
    return los, tst_los, acc, test_acc


# Plots
def plots(alpha, loss_list, tst_loss_list, train_acc_list, test_acc_list, size, rate):
    fig, axs = plt.subplots(2, 1, figsize=(10, 18))
    # Plot Loss
    axs[0].plot(alpha,loss_list,label="Train Loss")
    axs[0].plot(alpha,tst_loss_list,label="Test Loss")
    # Plot accuracy
    axs[1].plot(alpha,train_acc_list,label="Train Accuracy")
    axs[1].plot(alpha,test_acc_list,label="Test Accuacy")
    # Plot settings
    axs[0].set(title="Batch Size "+str(size)+", Rate "+str(rate), ylabel="Loss", xlabel="alpha")
    axs[0].legend(loc="upper right")
    axs[1].set(ylabel="Accuracy", xlabel="alpha")
    axs[1].legend(loc="upper right")
    if rate == 1e-3:
        filename = "combine_model_batch"+str(batch)+"_rate1e-3.png"
    else:
        filename = "combine_model_batch"+str(batch)+"_rate1e-2.png"
    plt.savefig(filename)
    plt.clf()

# Eval Models
batch_sizes = [64,1024]
learning_rates = [1e-3,1e-2]
alpha = np.linspace(-1,2,15)
start = time.time()
for batch in batch_sizes:
    for i in range(2):
        loss_list = []
        tst_loss_list = []
        train_acc_list = []
        test_acc_list = []
        for a in alpha:
            tf.reset_default_graph()
            with tf.Session() as sess:
                model1_w1, model1_wout = get_model_weights(batch, learning_rates[i])
            tf.reset_default_graph()
            with tf.Session() as sess:
                model2_w1, model2_wout = get_model_weights(batch, learning_rates[i])
            model3_w1 = (1-a)*model1_w1 + a*model2_w1
            model3_wout = (1-a)*model1_wout + a*model2_wout
            tf.reset_default_graph()
            with tf.Session() as sess:
                los, tst_los, acc, test_acc = get_model_info(batch, learning_rates[i], model3_w1, model3_wout)
            loss_list.append(los)
            tst_loss_list.append(tst_los)
            train_acc_list.append(acc)
            test_acc_list.append(test_acc)
        plots(alpha, loss_list, tst_loss_list, train_acc_list, test_acc_list, batch, learning_rates[i])
print("Runtime:", time.time() - start)

