import tensorflow as tf
from hah_classification.models.nn_factory import init_embedding, muti_class_attention, muti_layer_conv, fc


class ClassificationModel:
    def __init__(self,
                 class_first,
                 class_second,
                 filter_num,
                 kernel_sizes,
                 skip,
                 layer_num,
                 vocab_size,
                 embedding_size,
                 class_num,
                 learning_rate,
                 keep_drop_prob=1.0,
                 is_train=False,
                 embedding_path=None):

        self.class_first = class_first
        self.class_second = class_second
        self.filter_num = filter_num
        self.kernel_sizes = kernel_sizes
        self.skip = skip
        self.layer_num = layer_num
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.class_num = class_num
        self.learning_rate = learning_rate
        self.keep_drop_prob = keep_drop_prob if is_train else 1.0
        self.is_train = is_train
        self.embedding_path = embedding_path
        self.build()
        self.name = self.__class__.__name__

    def build(self):
        self._build_placeholder()
        self._build_embedding(self.vocab_size, self.embedding_size, self.embedding_path)
        input_sequences_emb = tf.nn.embedding_lookup(self.embedding, self.input_sequences)
        input_sequences_emb = tf.layers.dropout(input_sequences_emb,
                                                rate=1 - self.keep_drop_prob,
                                                training=self.is_train)
        conv_out = muti_layer_conv(input_sequences_emb, self.filter_num, self.skip, self.layer_num)
        first_outs = muti_class_attention(conv_out, self.class_first)
        outputs = []
        for first_out, class_num in zip(first_outs, self.class_second):
            seconds_outs = muti_class_attention(conv_out, class_num, concat_inputs=first_out)
            for second_out in seconds_outs:
                outputs.append(fc(second_out, self.class_num))

        losses = []
        class_pros = []
        class_labels = []
        labels = tf.unstack(self.input_labels, axis=1)
        for output, label in zip(outputs, labels):
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output, labels=label))
            class_pro = tf.nn.softmax(output, axis=1)
            class_label = tf.expand_dims(tf.argmax(class_pro, axis=1, output_type=tf.int32), axis=1)
            index = tf.stack([tf.range(tf.shape(class_label)[0])[:, None], class_label], axis=2)
            class_pro = tf.gather_nd(class_pro, index)
            losses.append(loss)
            class_pros.append(class_pro)
            class_labels.append(class_label)

        self.loss = tf.reduce_sum(losses)
        self.class_pros = tf.concat(class_pros, axis=1)
        self.class_labels = tf.concat(class_labels, axis=1)

        correct_pred = tf.equal(self.class_labels, self.input_labels)
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


        if self.is_train:
            self._build_optimize(self.loss, self.learning_rate)

    def inference(self, sequences, lengths):
        labels, class_pro = self.session.run([self.class_labels, self.class_pros],
                                             feed_dict={self.input_sequences: sequences,
                                                        self.input_lengths: lengths})

        return labels, class_pro

    def to_inference(self, model_path):
        ckpt = tf.train.get_checkpoint_state(model_path)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.session = tf.Session(config=config)
        saver = tf.train.Saver(max_to_keep=1)
        saver.restore(self.session, ckpt.model_checkpoint_path)
        print('use_inference')

    def _build_placeholder(self, batch_size=None, sequence_length=None):
        """
        构建placeholder
        :param batch_size: 默认为None,即动态batch_size
        :param sequence_length: 序列长度
        """
        with tf.name_scope('input_placeholder'):
            self.input_sequences = tf.placeholder(tf.int32, [batch_size, sequence_length], 'input_sequences')
            self.input_labels = tf.placeholder(tf.int32, [batch_size, sum(self.class_second)])
            self.input_lengths = tf.placeholder(tf.int32, [batch_size], 'input_lengths')

    def _build_embedding(self, vocab_size, embedding_size, embedding_path):
        with tf.device('/cpu:0'):
            self.embedding = init_embedding(vocab_size, embedding_size, embedding_path)

    def _build_optimize(self, loss, learning_rate, optimizer='adam'):
        self.global_step = tf.Variable(0, trainable=False, name="Global_Step")
        if optimizer.lower() == 'adam':
            Optimizer = tf.train.AdamOptimizer
        else:
            Optimizer = tf.train.GradientDescentOptimizer
        self.optimize = Optimizer(learning_rate=learning_rate).minimize(loss, global_step=self.global_step)

    def print_parms(self):
        print('\n', '-' * 20)
        print('%s : parms' % self.name)
        for var in tf.trainable_variables():
            print(var.name, var.shape)
        print('-' * 20, '\n')
