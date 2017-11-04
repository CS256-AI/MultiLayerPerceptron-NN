import tensorflow as tf
import sticky_snippet_generator as gen
import sys

# Configurable parameters
mini_batch_size = 1000
inp_feature_size = 40
op_classes = 6
epochs = 75
learning_rate = 0.05

# Neuron count
h1_neuron = 500
h2_neuron = 250
h3_neuron = 100

# Weight and Baias Initializers
w_init_min, w_init_max = 0.1,0.5
b_init_min, b_init_max = 0.1,0.5


def perceptron(input, weight, bias, activation=None):
    z = tf.add(tf.matmul(input, weight), bias)
    a = z
    if activation:
        a = activation(z)
    return a

def leaky_relu(x, alpha=0.01):
    return tf.nn.relu(x) - alpha * tf.nn.relu(-x)

# input/output placeholder
inp = tf.placeholder(tf.float32, shape=[None,inp_feature_size])
op = tf.placeholder(tf.float32, shape=[None, op_classes])

# Defining weight variables
h1_w = tf.Variable(tf.truncated_normal([inp_feature_size, h1_neuron], stddev=0.1))
h2_w = tf.Variable(tf.truncated_normal([h1_neuron, h2_neuron], stddev=0.1))
h3_w = tf.Variable(tf.truncated_normal([h2_neuron, h3_neuron], stddev=0.1))
op_w = tf.Variable(tf.truncated_normal([h3_neuron, op_classes], stddev=0.1))

# defining bias variables
#tf.random_normal([h1_neuron], mean=0, stddev=0.001)
# h1_b = tf.Variable(tf.truncated_normal([h1_neuron], stddev=0.001))
# h2_b = tf.Variable(tf.truncated_normal([h2_neuron], stddev=0.001))
# h3_b = tf.Variable(tf.truncated_normal([h3_neuron], stddev=0.001))
h1_b = tf.constant(0.1, shape=[h1_neuron])
h2_b = tf.constant(0.1, shape=[h2_neuron])
h3_b = tf.constant(0.1, shape=[h3_neuron])
op_b = tf.Variable(tf.truncated_normal([op_classes], stddev=0.001))

h1_activations = perceptron(inp, h1_w, h1_b, tf.nn.relu)
h2_activations = perceptron(h1_activations, h2_w, h2_b, tf.nn.relu)
h3_activations = perceptron(h2_activations, h3_w, h3_b, tf.nn.relu)

op_logist = perceptron(h3_activations, op_w, op_b)
op_soft_max = tf.nn.softmax(op_logist)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=op_logist, labels=op))
model_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

# Calculate accuracy
correct_prediction = tf.equal(tf.argmax(op_soft_max, 1), tf.argmax(op, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
confusion_matrix = tf.confusion_matrix(tf.argmax(op, 1), tf.argmax(op_soft_max, 1), num_classes=op_classes)

init = tf.global_variables_initializer()

def get_random_batches():
    return [None,None]

def train(model_file, data_folder):
    data_gen = gen.DataUtil()
    data = data_gen.load_data(data_folder)

    with tf.Session() as session:
        session.run(init)
        # read number of examples using the data utility
        for e in range(epochs):
            processed = 0
            print("Processing epoch {} of {}".format(e+1, epochs))
            batched_data_list = data.get_epoch_data(mini_batch_size)
            for batch_x, batch_y in batched_data_list:
                o,l,a = session.run([model_optimizer, loss, accuracy], feed_dict={inp: batch_x, op:batch_y})
                processed += mini_batch_size
                print("Processed {} training data. Batch Loss : {}. Batch Accuracy : {}".format(processed, l, a))
        print("Training complete. Final Loss: {}".format(l))

        saver = tf.train.Saver()
        if not model_file.endswith(".ckpt"): model_file+= ".ckpt"
        save_path = saver.save(session, model_file)
        print("Model saved in file: ", save_path)


def five_fold_train(model_file, data_file):
    with tf.Session() as session:
        session.run(init)
        num_examples = 0
        x, y = None, None # Read complete data from utility
        fold = num_examples/5
        test_fold_start, test_fold_end = 0, fold
        for f in range(5):
            print("Training data by leaving out fold {}".format(f+1))
            trimmed_x = x[:test_fold_start] + x[test_fold_end:]
            trimmed_y = y[:test_fold_start] + y[test_fold_end:]
            # read number of examples using the data utility
            for e in range(epochs):
                processed = 0
                print("Processing epoch {} of {}".format(e+1, epochs))
                batch_iters = len(trimmed_x)/mini_batch_size
                for batch_x, batch_y in get_random_batches(trimmed_x, trimmed_y, mini_batch_size):
                    o,l = session.run([model_optimizer, loss], feed_dict={inp: batch_x, op: batch_y})
                    processed += batch_iters
                    print("Processed {} training data. Current Loss : {}".format(processed, l))
            print("Training complete by leaving out fold {}. Final Loss: {}".format(f+1, l))
            #print(session.run([h1_w,h2_w,h3_w,op_w]))
            test_fold_start += fold
            test_fold_end += fold

        saver = tf.train.Saver()
        if not model_file.endswith(".ckpt"): model_file+= ".ckpt"
        save_path = saver.save(session, model_file)
        print("Model saved in file: ", save_path)


def test(model_file, data_file):
    saver = tf.train.Saver()
    data_gen = gen.DataUtil()
    data = data_gen.load_data(data_folder)

    with tf.Session() as session:
        # Restore variables from disk.
        if not model_file.endswith(".ckpt"): model_file += ".ckpt"
        saver.restore(session, model_file)
        print("Model restored.")
        # Check the values of the variables
        # read number of examples using the data utility
        test_x, test_y = data.get_test_data()
        a, cm = session.run([accuracy, confusion_matrix], feed_dict={inp:test_x, op:test_y})
        print("Model Accuracy : ", a)
        print("Confusion Matrix :\n", cm)


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print(
            "Insufficient number of arguments.\nPattern : python sticky_snippet_net.py mode model_file data_folder")
        sys.exit()
    else:
        mode, model_file, data_folder = sys.argv[1:]
        if mode.upper() == "TRAIN": train(model_file, data_folder)
        elif mode.upper() == "TEST": test(model_file, data_folder)
        elif mode.upper() == "5FOLD": five_fold_train(model_file, data_folder)


