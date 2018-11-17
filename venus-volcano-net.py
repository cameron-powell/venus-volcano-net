import numpy as np
import tensorflow as tf

def predictions(W, b, X_set):
    m = X_set.shape[1]
    Y_predictions = np.zeros((1, m))

    Z_param = np.dot(W, X_set) + b
    A_param = 1 / (1 + np.exp(-1 * Z_param))

    for i in range(m):
        Y_predictions[0, i] = 1 if A_param[0, i] > 0.5 else 0

    return Y_predictions

# Load in the pre-flattened images
train_x = np.loadtxt('train_images.csv', delimiter=',').T
train_y = np.loadtxt('train_labels.csv', delimiter=',', skiprows=1, usecols=0).reshape(7000, 1).T
print('train_x: (%s, %s)' % (train_x.shape[0], train_x.shape[1]))
print('train_y: (%s, %s)' % (train_y.shape[0], train_y.shape[1]))
test_x = np.loadtxt('test_images.csv', delimiter=',').T
test_y = np.loadtxt('test_labels.csv', delimiter=',', skiprows=1, usecols=0).reshape(2734, 1).T
print('text_x: (%s, %s)' % (test_x.shape[0], test_x.shape[1]))
print('test_y: (%s, %s)' % (test_y.shape[0], test_y.shape[1]))

# Standardize the datasets
train_x = train_x/255.0
test_x = test_x/255.0

# Setup features and labels
Y = tf.placeholder(tf.float32, shape=(1, 7000))
X = tf.placeholder(tf.float32, shape=(12100, 7000))

# Setup forward propagation
W = tf.get_variable("W", [1, 12100], initializer=tf.contrib.layers.xavier_initializer(seed=1))                          # REMOVE SEED
b = tf.get_variable("b", [1, 1], initializer=tf.zeros_initializer())
Z = tf.add(tf.matmul(W, X), b)

# Setup calculation of the cost
cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=tf.transpose(Z), labels=tf.transpose(Y)))

# Setup backward propagation
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(cost)

# Train
init = tf.global_variables_initializer()
with tf.Session() as session:
    # Initialize the variables
    session.run(init)
    # Perform gradient descent
    for i in range(10):
        _, cost_result = session.run([optimizer, cost], feed_dict={X:train_x, Y:train_y})
        print('Cost after iteration %s: %s' % (i+1, cost_result))
    # Save the parameters
    W = session.run(W)
    b = session.run(b)
    # Calculate prediction accuracy
    train_predictions = predictions(W, b, train_x)
    print('Training Accuracy: ', (100 - np.mean(np.abs(train_predictions - train_y))*100), '%')
    test_predictions = predictions(W, b, test_x)
    print('Testing Accuracy: ', (100 - np.mean(np.abs(test_predictions - test_y))*100), '%')