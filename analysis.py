"""
Adopted from
https://github.com/aymericdamien/TensorFlow-Examples/

"""
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import gzip

actives = "dude/dud38/ace/actives_final.sdf.gz"
decoys = "dude/dud38/ace/decoys_final.sdf.gz"

def load(fn):
    inf = gzip.open(fn)
    gzsuppl = Chem.ForwardSDMolSupplier(inf)
    mols = [x for x in gzsuppl if x is not None]
    return mols

def fingerprints(mols):

    # generate fingeprints: Morgan fingerprint with radius 2
    fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2) for m in mols]

    # convert the RDKit explicit vectors into numpy arrays
    np_fps = []
    for fp in fps:
        arr = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(fp, arr)
        np_fps.append(arr)
    return np_fps

def categories(fingerprints, index):
    mask = np.zeros((len(fingerprints), 2))
    mask[:,index] = 1.0
    return zip(fingerprints, mask)

actives = categories(fingerprints(load(actives)), 0)
decoys = categories(fingerprints(load(decoys)), 1)

print(len(actives), len(decoys))

molecules = np.vstack((actives, decoys))
np.random.shuffle(molecules)
training, testing = molecules[::2], molecules[1::2]

print(len(training), len(testing))




import tensorflow as tf

# Parameters
learning_rate = 0.001
training_iters = len(training) // 2
batch_size = 2 # 128
display_step = 10


# Network Parameters
n_input = 2048 # Length of default Morgan fingerprint
n_classes = 2  # Only 2 classes: actives and decoys
dropout = 0.75 # Dropout, probability to keep units

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)


# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


# Create model
def conv_net(x, weights, biases, dropout):
    # Reshape input picture
    # JD treating this like an image, even though I shouldn't
    x = tf.reshape(x, shape=[-1, 64, 32, 1])

    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=1)

    # Convolution Layer
    #conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    #conv2 = maxpool2d(conv2, k=1)
    print(conv1.get_shape())

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv1, [-1, weights['wd1'].get_shape().as_list()[0]], name="lolreshape")
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out

# Store layers weight & bias
weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([4, 2, 1, 16])),
    # 5x5 conv, 32 inputs, 64 outputs
    #'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([64*32*16, 16])),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([16, n_classes]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([16])),
    #'bc2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([16])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
pred = conv_net(x, weights, biases, keep_prob)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.initialize_all_variables()


# Launch the graph
with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        batch_x, batch_y = zip(*training[step*batch_size-batch_size:step*batch_size])
        assert(len(batch_x) == batch_size)

        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: np.array(batch_x), y: np.array(batch_y),
                                       keep_prob: dropout})
        if step % display_step == 0:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,
                                                              y: batch_y,
                                                              keep_prob: 1.})
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        step += 1
    print("Optimization Finished!")

    # Calculate accuracy for 256 mnist test images
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={x: zip(*testing)[0],
                                      y: zip(*testing)[1],
                                      keep_prob: 1.}))
