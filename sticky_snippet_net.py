import tensorflow as tf
import sticky_snippet_generator as gen
import sys

# Configurable parameters
mini_batch_size = 1000
inp_feature_size = 40
op_classes = 6
epochs = 100
learning_rate = 0.01

# Neuron count
h1_neuron = 80
h2_neuron = 80
h3_neuron = 80

# Weight and Baias Initializers
w_init_min, w_init_max = 0.1,0.5
b_init_min, b_init_max = 0.1,0.5

def perceptron(input, weight, bias, activation=None):
    z = tf.add(tf.matmul(input, weight), bias)
    a = z
    if activation:
        a = activation(z)
    return a

# input/output placeholder
inp = tf.placeholder(tf.float32, shape=[None,inp_feature_size])
op = tf.placeholder(tf.float32, shape=[None, op_classes])

# Defining weight variables
h1_w = tf.Variable(tf.random_uniform([inp_feature_size, h1_neuron], minval=w_init_min, maxval=w_init_max, dtype=tf.float32))
h2_w = tf.Variable(tf.random_uniform([h1_neuron, h2_neuron], minval=w_init_min, maxval=w_init_max, dtype=tf.float32))
h3_w = tf.Variable(tf.random_uniform([h2_neuron, h3_neuron], minval=w_init_min, maxval=w_init_max, dtype=tf.float32))
op_w = tf.Variable(tf.random_uniform([h3_neuron, op_classes], minval=w_init_min, maxval=w_init_max, dtype=tf.float32))

# defining bias variables
h1_b = tf.Variable(tf.random_uniform([h1_neuron],minval=b_init_min, maxval= b_init_max, dtype = tf.float32))
h2_b = tf.Variable(tf.random_uniform([h2_neuron],minval=b_init_min, maxval= b_init_max, dtype = tf.float32))
h3_b = tf.Variable(tf.random_uniform([h3_neuron],minval=b_init_min, maxval= b_init_max, dtype = tf.float32))
op_b = tf.Variable(tf.random_uniform([op_classes], minval=b_init_min, maxval= b_init_max, dtype = tf.float32))

h1_activations = perceptron(inp, h1_w, h1_b, tf.nn.relu)
h2_activations = perceptron(h1_activations, h2_w, h2_b, tf.nn.relu)
h3_activations = perceptron(h2_activations, h3_w, h3_b, tf.nn.relu)

op_logist = perceptron(h3_activations, op_w, op_b)
op_soft_max = tf.nn.softmax(op_logist)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=op_logist, labels=op))
model_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

init = tf.global_variables_initializer()


def get_random_batches(batch_count):
    return None,None


def train(model_file, data_folder):
    data_gen = gen.DataUtil()
    data = data_gen.load_data(data_folder)

    with tf.Session() as session:
        session.run(init)
        # read number of examples using the data utility
        for e in range(epochs):
            processed = 0
            print("Processing epoch {} of {}".format(e+1, epochs))
            for batch_x, batch_y in data.get_epoch_data(mini_batch_size):
                o,l = session.run([model_optimizer, loss], feed_dict={inp: batch_x, op:batch_y})
                processed += mini_batch_size
                print("Processed {} training data. Current Loss : {}".format(processed, l))
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
        saver.restore(session, model_file)
        print("Model restored.")
        # Check the values of the variables
        # read number of examples using the data utility
        test_x, test_y = data.get_epoch_data(500)[0]

        # Comparing the model result with label
        correct_prediction = tf.equal(tf.argmax(op_soft_max, 1), tf.argmax(test_y, 1))

        # Calculate accuracy
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print(session.run(accuracy, feed_dict={inp:test_x, op:test_y}))


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


