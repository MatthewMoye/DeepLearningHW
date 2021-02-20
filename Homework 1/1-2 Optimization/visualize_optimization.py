import time, math
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
#np.random.seed(1)
#tf.set_random_seed(1)

data = input_data.read_data_sets("data/MNIST/", one_hot=True)
data.test.cls = np.argmax(data.test.labels, axis=1)
img_size = 28


# Get each model info
def get_model_info():
    x = tf.placeholder(tf.float32, shape=[None, img_size * img_size], name="x")
    y_true = tf.placeholder(tf.float32, shape=[None, 10], name="y_true")
    y_true_cls = tf.argmax(y_true, dimension=1)

    # Model
    h1 = tf.layers.dense(inputs=x, name="h1", units=32, activation=tf.nn.relu)
    h2 = tf.layers.dense(inputs=x, name="h2", units=16, activation=tf.nn.relu)
    out = tf.layers.dense(inputs=h2, name="out", units=10, activation=None)

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
    weights_h1 = get_weights_variable(layer_name='h1')
    weights_out = get_weights_variable(layer_name='out')

    loss_list = []
    train_acc_list = []
    weight_layer1 = []
    weight_model = []

    for i in range(10):
        feed = {x: data.train.images, y_true: data.train.labels}
        sess.run(optimizer, feed_dict=feed)
        if i % 3 == 0:
            los, acc = sess.run([loss, accuracy], feed_dict=feed)
            w1, wm = sess.run([weights_h1, weights_out],feed)
            weight_layer1.append(w1)
            weight_model.append(wm)
            train_acc_list.append(acc)
    
    weight_layer1 = np.vstack(weight_layer1)
    weight_model = np.vstack(weight_model)
    #weight_layer1 = np.array(weight_layer1).reshape(10,784*128)
    #weight_model = np.array(weight_model).reshape(10,64*10)

    return weight_layer1, weight_model

# Run Models
weight1 = []
weight_model = []
start = time.time()
for i in range(8):
    tf.reset_default_graph()
    with tf.Session() as sess:
        w_layer1, w_model = get_model_info()
        weight1.append(w_layer1)
        weight_model.append(w_model)
print("Runtime:", time.time() - start)


# Plot weights, function from https://github.com/zhangdan8962/DeepLearning
from sklearn.decomposition import PCA
def plot_fc_weights(weights_list,filename):
    pca = PCA(n_components=2)
    
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('Principal Component 1', fontsize = 15)
    ax.set_ylabel('Principal Component 2', fontsize = 15)
    ax.set_title('2 component PCA', fontsize = 20)
    for i in range(len(weights_list)):
        principalComponents = pca.fit_transform(weights_list[i])
        ax.scatter(principalComponents[:,0], principalComponents[:,1], label=str("Train_"+str(i)), alpha=0.5)
    ax.legend()
    plt.savefig(filename)
plot_fc_weights(weight1,"w1_plot.png")
plot_fc_weights(weight_model,"wfinal_plot.png")
