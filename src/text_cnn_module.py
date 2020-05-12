# -*- coding: UTF-8 -*-
'''
Created on Mon May 4 09:23:12 2020

@author: Damon Li
'''

from tensorflow.contrib import learn
import tensorflow as tf
import numpy as np
import datetime
import time
import os


class TextCNN(object):

    def __init__(self,
                 embedding_dir, # 预训练模型路径
                 sequence_length, #
                 num_classes,
                 filter_sizes,
                 num_filters,
                 train_data_dir='',
                 test_data_dir='',
                 model_save_dir='',
                 vocab_processor_dir='',
                 l2_reg_lambda=0.0,
                 batch_size=64,
                 num_epochs=10,
                 num_checkpoints=5,
                 evaluate_every=100,
                 dev_sample_percentage=0.1,
                 dropout_prob=0.5,
                 is_inference=False,
                 device_id='/gpu:1'):

        self.embedding     = np.load(embedding_dir)
        self.embedding_dim = np.shape(self.embedding)[-1]

        self.device              = device_id
        self.num_epochs          = num_epochs
        self.batch_size          = batch_size
        self.num_classes         = num_classes
        self.num_filters         = num_filters
        self.is_inference        = is_inference
        self.dropout_prob        = dropout_prob
        self.filter_sizes        = filter_sizes
        self.l2_reg_lambda       = l2_reg_lambda
        self.test_data_dir       = test_data_dir
        self.model_save_dir      = model_save_dir
        self.train_data_dir      = train_data_dir
        self.evaluate_every      = evaluate_every
        self.num_checkpoints     = num_checkpoints
        self.sequence_length     = sequence_length
        self.val_sample_ratio    = dev_sample_percentage
        self.vocab_processor_dir = vocab_processor_dir



    def load_raw_data(self, filepath):
        """
        Loads MR polarity data from files, splits the data into words and generates labels.
        Returns split sentences and labels.
        """
        # Load data from files
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            train_datas = f.readlines()
        one_hot_labels = []
        x_datas = []
        for line in train_datas:
            parts = line.encode('utf-8').decode('utf-8-sig').strip().split(' ', 1)
            if len(parts) < 2 or (len(parts[1].strip()) == 0):
                continue
            x_datas.append(parts[1])
            one_hot_label = [0] * self.num_classes
            label = int(parts[0])
            one_hot_label[label] = 1
            one_hot_labels.append(one_hot_label)
        print('[INFO] train data size = ', len(train_datas))
        return [x_datas, np.array(one_hot_labels)]


    def data_processing(self, train_data_dir):
        print("[INFO] Loading train dataset...")
        x_text, y = self.load_raw_data(train_data_dir)

        # Build vocabulary
        max_document_length = max([len(x.split(" ")) for x in x_text])
        vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
        x = np.array(list(vocab_processor.fit_transform(x_text)))

        # Randomly shuffle data
        np.random.seed(10)
        shuffle_indices = np.random.permutation(np.arange(len(y)))
        x_shuffled = x[shuffle_indices]
        y_shuffled = y[shuffle_indices]

        # Split train/test set
        # TODO: This is very crude, should use cross-validation
        dev_sample_index = -1 * int(self.val_sample_ratio * float(len(y)))
        x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
        y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]

        del x, y, x_shuffled, y_shuffled

        print("[INFO] Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
        print("[INFO] Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))

        return x_train, y_train, vocab_processor, x_dev, y_dev


    def batch_iter(self, data, batch_size, num_epochs, shuffle=True):
        """
        Generates a batch iterator for a dataset.
        """
        data = np.array(data)
        data_size = len(data)
        num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
        for epoch in range(num_epochs):
            # Shuffle the data at each epoch
            if shuffle:
                shuffle_indices = np.random.permutation(np.arange(data_size))
                shuffled_data = data[shuffle_indices]
            else:
                shuffled_data = data
            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, data_size)
                yield shuffled_data[start_index:end_index]


    def build_model(self):
        if self.is_inference:
            if os.path.join(self.vocab_processor_dir):
                vocab_processor = learn.preprocessing.VocabularyProcessor.restore(self.vocab_processor_dir)
                self.vocab_size = len(vocab_processor.vocabulary_)
            else:
                raise IOError('{}文件不存在！'.format(self.vocab_processor_dir))

        # Placeholders for input, output and dropout
        self.input_x = tf.compat.v1.placeholder(tf.int32, [None, self.sequence_length], name="input_x")
        self.input_y = tf.compat.v1.placeholder(tf.float32, [None, self.num_classes], name="input_y")
        self.dropout_keep_prob = tf.compat.v1.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.device(self.device), tf.name_scope("embedding"):
            # self.W = tf.Variable(
            #     tf.random.uniform([self.vocab_size, self.embedding_dim], -1.0, 1.0),
            #     name="W") # 不用预训练embedding
            self.W = tf.Variable(initial_value=self.embedding, trainable=True, name="W") # 使用预训练embedding模型
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(self.filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, self.embedding_dim, 1, self.num_filters]
                W = tf.Variable(tf.random.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[self.num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool2d(
                    h,
                    ksize=[1, self.sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = self.num_filters * len(self.filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.compat.v1.get_variable(
                "W",
                shape=[num_filters_total, self.num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[self.num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.compat.v1.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + self.l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")


    def train(self):
        x_train, y_train, vocab_processor, x_dev, y_dev = self.data_processing(self.train_data_dir)
        self.vocab_size = len(vocab_processor.vocabulary_)

        with tf.Graph().as_default():
            session_conf = tf.ConfigProto(allow_soft_placement=True,
                                          log_device_placement=False)
            sess = tf.compat.v1.Session(config=session_conf)
            with sess.as_default():
                self.build_model() # 创建模型的函数一定要放在Graph内部，不然Graph认为没有变量进行操作，出现'No variables to optimize'的报错

                # Define Training procedure
                global_step = tf.Variable(0, name="global_step", trainable=False)
                optimizer = tf.compat.v1.train.AdamOptimizer(1e-3)
                grads_and_vars = optimizer.compute_gradients(self.loss)
                train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

                # Keep track of gradient values and sparsity (optional)
                grad_summaries = []
                for g, v in grads_and_vars:
                    if g is not None:
                        grad_hist_summary = tf.compat.v1.summary.histogram("{}/grad/hist".format(v.name), g)
                        sparsity_summary = tf.compat.v1.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                        grad_summaries.append(grad_hist_summary)
                        grad_summaries.append(sparsity_summary)
                grad_summaries_merged = tf.compat.v1.summary.merge(grad_summaries)

                # Output directory for models and summaries
                timestamp = str(int(time.time()))
                out_dir = os.path.abspath(os.path.join(self.model_save_dir, timestamp))
                print("[INFO] Writing to {}\n".format(out_dir))

                # Summaries for loss and accuracy
                loss_summary = tf.compat.v1.summary.scalar("loss", self.loss)
                acc_summary = tf.compat.v1.summary.scalar("accuracy", self.accuracy)

                # Train Summaries
                train_summary_op = tf.compat.v1.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
                train_summary_dir = os.path.join(out_dir, "summaries", "train")
                train_summary_writer = tf.compat.v1.summary.FileWriter(train_summary_dir, sess.graph)

                # Dev summaries
                dev_summary_op = tf.compat.v1.summary.merge([loss_summary, acc_summary])
                dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
                dev_summary_writer = tf.compat.v1.summary.FileWriter(dev_summary_dir, sess.graph)

                # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
                checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
                checkpoint_prefix = os.path.join(checkpoint_dir, "model")
                if not os.path.exists(checkpoint_dir):
                    os.makedirs(checkpoint_dir)
                saver = tf.compat.v1.train.Saver(tf.global_variables(), max_to_keep=self.num_checkpoints)

                # Write vocabulary
                vocab_processor.save(os.path.join(out_dir, "vocab"))

                # Initialize all variables
                sess.run(tf.global_variables_initializer())

                def train_step(x_batch, y_batch):
                    """
                    A single training step
                    """
                    feed_dict = {
                        self.input_x: x_batch,
                        self.input_y: y_batch,
                        self.dropout_keep_prob: self.dropout_prob
                    }
                    _, step, summaries, loss, accuracy = sess.run(
                        [train_op, global_step, train_summary_op, self.loss, self.accuracy],
                        feed_dict)
                    time_str = datetime.datetime.now().isoformat()
                    print("[INFO] {}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                    train_summary_writer.add_summary(summaries, step)

                def dev_step(x_batch, y_batch, writer=None):
                    """
                    Evaluates model on a dev set
                    """
                    feed_dict = {
                        self.input_x: x_batch,
                        self.input_y: y_batch,
                        self.dropout_keep_prob: 1.0
                    }
                    step, summaries, loss, accuracy = sess.run(
                        [global_step, dev_summary_op, self.loss, self.accuracy],
                        feed_dict)
                    time_str = datetime.datetime.now().isoformat()
                    print("[INFO] {}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                    if writer:
                        writer.add_summary(summaries, step)

                # Generate batches
                batches = self.batch_iter(list(zip(x_train, y_train)), self.batch_size, self.num_epochs)
                # Training loop. For each batch...
                for batch in batches:
                    x_batch, y_batch = zip(*batch)
                    train_step(x_batch, y_batch)
                    current_step = tf.train.global_step(sess, global_step)
                    if current_step % self.evaluate_every == 0:
                        print("\n[INFO] Evaluation:")
                        dev_step(x_dev, y_dev, writer=dev_summary_writer)
                        print("")
                    if current_step % self.evaluate_every == 0:
                        path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                        print("[INFO] Saved model checkpoint to {}\n".format(path))



if __name__ == '__main__':
    import settings
    setting = settings.setting()

    test_obj = TextCNN(embedding_dir=setting.text_cls_embedding_dir,
                       sequence_length=setting.text_cls_sentence_len,
                       num_classes=setting.num_classes,
                       filter_sizes=list(map(int, setting.filter_sizes.split(" "))),
                       num_filters=setting.num_filters,
                       train_data_dir=setting.text_cls_train_dir,
                       test_data_dir='',
                       model_save_dir=setting.text_cnn_model_save_dir,
                       vocab_processor_dir='',
                       l2_reg_lambda=setting.l2_lambda,
                       batch_size=300, #64
                       num_epochs=200,
                       num_checkpoints=setting.text_cls_num_checkpoints,
                       evaluate_every=100,
                       dev_sample_percentage=0.1,
                       dropout_prob=setting.text_cls_dropout_prob,
                       is_inference=False,
                       device_id=setting.cpu_id)

    test_obj.train()
