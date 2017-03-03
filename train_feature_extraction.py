import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split
from alexnet import AlexNet
from sklearn.utils import shuffle
import time

# TODO: Load traffic signs data.
file = 'train.p'
nb_classes = 43
epochs = 10
batch_size = 128

with open(file, mode='rb') as f:
    data = pickle.load(f)
    
# TODO: Split data into training and validation sets.
# Examening data
print(type(data))
print(len(data))
print(data.keys())
#print(data)

X_train, X_val, y_train, y_val  = train_test_split(data['features'],data['labels'], test_size=0.33, random_state=0)

# TODO: Define placeholders and resize operation.
# Examening data
print(len(X_train))
print(X_train.shape)
print(len(X_val))
print(X_val.shape)
print(len(y_train))
print(y_train.shape)
print(len(y_val))
print(y_val.shape)

x = tf.placeholder(tf.float32, (None, 32,32,3))
y = tf.placeholder(tf.int64, None)
resized = tf.image.resize_images(x, (227,227))

# TODO: pass placeholder as first argument to `AlexNet`.
fc7 = AlexNet(resized, feature_extract=True)
# NOTE: `tf.stop_gradient` prevents the gradient from flowing backwards
# past this point, keeping the weights before and up to `fc7` frozen.
# This also makes training faster, less work to do!
fc7 = tf.stop_gradient(fc7)

# TODO: Add the final layer for traffic sign classification.
shape = (fc7.get_shape().as_list()[-1], nb_classes)
fc8W = tf.Variable(tf.truncated_normal(shape, stddev=1e-2))
fc8b = tf.Variable(tf.zeros(nb_classes))
logits = tf.nn.xw_plus_b(fc7, fc8W, fc8b)

# TODO: Define loss, training, accuracy operations.
# HINT: Look back at your traffic signs project solution, you may
# be able to reuse some the code.
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, y)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer()
training_operation = optimizer.minimize(loss_operation, var_list=[fc8W,fc8b])
print(training_operation)
init = tf.global_variables_initializer()

prediction = tf.arg_max(logits, 1)
accurac_operation = tf.reduce_mean(tf.cast(tf.equal(prediction,y), tf.float32))

# TODO: Train and evaluate the feature extraction model.

def evaluate(X_data, y_data):
    total_accuracy = 0
    total_loss = 0
    for offset in range(0, len(X_train),batch_size):
        end = offset + batch_size
        batch_X = X_data[offset:end]
        batch_y = y_data[offset:end]
        loss = sess.run([loss_operation, accurac_operation])
        accuracy = sess.run(accurac_operation, feed_dict={x: batch_X, y:batch_y})
        total_accuracy += (accuracy*batch_X.shape[0])
        total_loss = (loss*batch_X.shape[0])
    return total_accuracy/X_data.shape[0], total_loss/X_data.shape[0]

with tf.Session() as sess:
    sess.run(init)
    print('Training...')
    
    for i in range(epochs):
        X_train,y_train = shuffle(X_train,y_train)
        t0 = time.time()
        for offset in range(0, len(X_train), batch_size):
            end = offset + batch_size
            batch_x = X_train[offset:end]
            batch_y = y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})
        
        val_loss, vall_acc = evaluate(X_val,y_val)
        print('Epoch', i+1)
        print('Time: %.3f seconds' % (time.time()-t0))
        print('Validation loss = ', val_loss)
        print('Validation accu = ', vall_acc)
        print('')