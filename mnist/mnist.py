from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import argparse
import sys
import os

def train():
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
    global_step = tf.Variable(0, trainable=False)
    learning_rate = FLAGS.learning_rate
    if FLAGS.practice == 3:
        learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, global_step,
                                                   100, 0.96, staircase=True)

    sess = tf.InteractiveSession()
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, shape=[None, 784])
        y_ = tf.placeholder(tf.float32, shape=[None, 10])

    def weight_variable(shape):
        """Create a weight variable with appropriate initialization."""
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(shape):
        """Create a bias variable with appropriate initialization."""
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def variable_summaries(var):
        """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            #tf.summary.histogram('histogram', var)

    def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
        """Reusable code for making a simple neural net layer.
        It does a matrix multiply, bias add, and then uses relu to nonlinearize.
        It also sets up name scoping so that the resultant graph is easy to read,
        and adds a number of summary ops.
        """
        # Adding a name scope ensures logical grouping of the layers in the graph.
        with tf.name_scope(layer_name):
            # This Variable will hold the state of the weights for the layer
            with tf.name_scope('weights'):
                weights = weight_variable([input_dim, output_dim])
                variable_summaries(weights)
            with tf.name_scope('biases'):
                biases = bias_variable([output_dim])
                variable_summaries(biases)
            with tf.name_scope('Wx_plus_b'):
                preactivate = tf.matmul(input_tensor, weights) + biases
                tf.summary.histogram('pre_activations', preactivate)
            activations = act(preactivate, name='activation')
            #tf.summary.histogram('activations', activations)
            return activations

    def conv2d(x, W, strides=[1, 2, 2, 1]):
        return tf.nn.conv2d(x, W, strides=strides, padding='SAME')

    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1], padding='SAME')

    keep_prob = tf.placeholder(tf.float32)
    if FLAGS.practice == 1:
        y = nn_layer(x, 784, 10, 'layer1', tf.identity)
    elif FLAGS.practice == 2:
        hidden1 = nn_layer(x, 784, 1024, 'layer1')
        y = nn_layer(hidden1, 1024, 10, 'layer2', tf.identity)
    elif FLAGS.practice == 3:
        hidden1 = nn_layer(x, 784, 1024, 'layer1')
        hidden2 = nn_layer(hidden1, 1024, 256, 'layer2')
        y = nn_layer(hidden2, 256, 10, 'layer3', tf.identity)
    elif FLAGS.practice == 4:
        x_image = tf.reshape(x, [-1,28,28,1])

        W_conv1 = weight_variable([5, 5, 1, 16])
        b_conv1 = bias_variable([16])
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

        W_conv2 = weight_variable([5, 5, 16, 16])
        b_conv2 = bias_variable([16])
        h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2)

        W_fc1 = weight_variable([7 * 7 * 16, 1024])
        b_fc1 = bias_variable([1024])

        h_conv2_flat = tf.reshape(h_conv2, [-1, 7*7*16])
        h_fc1 = tf.nn.relu(tf.matmul(h_conv2_flat, W_fc1) + b_fc1)

        W_fc2 = weight_variable([1024, 10])
        b_fc2 = bias_variable([10])
        y = tf.matmul(h_fc1, W_fc2) + b_fc2
    elif FLAGS.practice == 5:
        x_image = tf.reshape(x, [-1,28,28,1])

        W_conv1 = weight_variable([5, 5, 1, 16])
        b_conv1 = bias_variable([16])
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1, [1, 1, 1, 1]) + b_conv1)
        h_pool1 = max_pool_2x2(h_conv1)

        W_conv2 = weight_variable([5, 5, 16, 16])
        b_conv2 = bias_variable([16])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, [1, 1, 1, 1]) + b_conv2)
        h_pool2 = max_pool_2x2(h_conv2)

        W_fc1 = weight_variable([7 * 7 * 16, 1024])
        b_fc1 = bias_variable([1024])

        h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*16])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        W_fc2 = weight_variable([1024, 10])
        b_fc2 = bias_variable([10])
        y = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    with tf.name_scope('cross_entropy'):
        diff = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)
        with tf.name_scope('total'):
            cross_entropy = tf.reduce_mean(diff)
    tf.summary.scalar('cross_entropy', cross_entropy)

    with tf.name_scope('train'):
        if FLAGS.practice < 4:
            train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(
                cross_entropy, global_step=global_step)
        else:
            train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy, global_step=global_step)

    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
    val_writer = tf.summary.FileWriter(FLAGS.log_dir + '/val')
    tf.global_variables_initializer().run()
    saver = tf.train.Saver(tf.global_variables())
    practice_dir = FLAGS.model_dir+'/practice'+str(FLAGS.practice)
    if tf.gfile.Exists(practice_dir):
        tf.gfile.DeleteRecursively(practice_dir)
    tf.gfile.MakeDirs(practice_dir)

    def feed_dict(train):
        if train:
            if FLAGS.full_batch:
                xs, ys = mnist.train.images, mnist.train.labels
            else:
                xs, ys = mnist.train.next_batch(100)
            k = FLAGS.dropout
        else:
            xs, ys = mnist.validation.images, mnist.validation.labels
            k = 1.0
        return {x: xs, y_: ys, keep_prob: k}

    for i in range(FLAGS.max_steps):
        if i % FLAGS.steps_per_checkpoint == 0:
            checkpoint_path = os.path.join(practice_dir, 'model')
            saver.save(sess, checkpoint_path, global_step=global_step)
        if i % 10 == 0:  # Record summaries and test-set accuracy
            summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(False))
            val_writer.add_summary(summary, i)
            print('Accuracy at step %s: %s' % (i, acc))
        if i % 100 == 99:  # Record execution stats
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            summary, _ = sess.run([merged, train_step],
                                  feed_dict=feed_dict(True),
                                  options=run_options,
                                  run_metadata=run_metadata)
            train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
            train_writer.add_summary(summary, i)
            print('Adding run metadata for', i)
        else:  # Record a summary
            summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True))
            train_writer.add_summary(summary, i)
    print("test accuracy: %g" % accuracy.eval(feed_dict={
        x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
    train_writer.close()
    val_writer.close()

def main(_):
    if tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)
    if not tf.gfile.Exists(FLAGS.model_dir):
        tf.gfile.MakeDirs(FLAGS.model_dir)
    train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--practice', type=int, default=1,
                        help='Which practice to run.')
    parser.add_argument('--max_steps', type=int, default=2000,
                        help='Number of steps to run trainer.')
    parser.add_argument('--full_batch', nargs='?', const=True, type=bool, default=False,
                        help='If true, use full-batch for gradient descent.')
    parser.add_argument('--learning_rate', type=float, default=0.3,
                        help='Initial learning rate')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Keep probability for training dropout.')
    parser.add_argument('--model_dir', type=str, default='./models',
                        help='Directory for storing models')
    parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                        help='Directory for storing input data')
    parser.add_argument('--log_dir', type=str, default='/tmp/tensorflow/mnist/logs/mnist_with_summaries',
                        help='Summaries log directory')
    parser.add_argument('--steps_per_checkpoint', type=int, default=100,
                        help='How many training steps to do per checkpoint.')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
