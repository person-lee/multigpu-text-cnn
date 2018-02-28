#coding=utf-8
import tensorflow as tf

class CNN(object):
    def __init__(self, batch_size, sequence_len, embeddings, embedding_size, filter_sizes, num_filters, num_classes, l2_reg_lambda=0.0, adjust_weight=False,label_weight=[]):
        """
        batch_size: the size of each batch
        sequence_len:sequence length
        embeddings:embeddings of all words
        embedding_size:the dim of embedding
        filter_sizes:filter_sizes,eg:[1,2,3,4,5]
        num_filters:how many filters in convolution, eg:128
        num_classes:how many label in this experment
        """
        # define input variable
        self.dropout = tf.placeholder(tf.float32, name="dropout")
        self.batch_size = batch_size
        self.sequence_len = sequence_len
        self.embeddings = embeddings
        self.embedding_size = embedding_size
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.num_classes = num_classes
        self.l2_reg_lambda = l2_reg_lambda
        self.adjust_weight = adjust_weight
        self.label_weight = label_weight

    def inference(self, input_x):
        # define embedding
        l2_loss = tf.constant(0.0)
        out_pools = []
        with tf.name_scope("embedding-layer"):
            self.embs = tf.Variable(tf.to_float(self.embeddings), trainable=True, name="embs")
            x_with_embedding = tf.nn.embedding_lookup(self.embs, input_x, name="x_embeddings")

            # reshape x_with_embeddings (batch_size, sequence_len, embedding_size, in_channels)
            self.x_with_embeddings = tf.expand_dims(x_with_embedding, [-1])

        with tf.name_scope("convolution"):
            for idx, filter_size in enumerate(self.filter_sizes):
                with tf.name_scope("conv-maxpool-%s"%(filter_size)):
                    # filter_weight (height, width, in_channels, out_channels)
                    filter_weight = tf.get_variable("filter_weight-%s"%filter_size, [filter_size, self.embedding_size, 1, self.num_filters], initializer=tf.truncated_normal_initializer(stddev=0.1))
                    filter_bias = tf.get_variable("filter_bias-%s"%filter_size, [self.num_filters], initializer=tf.constant_initializer(0.1))

                    if self.l2_reg_lambda > 0:
                        l2_loss += tf.nn.l2_loss(filter_weight)
                        l2_loss += tf.nn.l2_loss(filter_bias)

                    # convolution (batch_size, sequence_len - filter_size + 1, 1, num_filters)
                    x_conv = tf.nn.conv2d(self.x_with_embeddings, filter_weight, strides=[1,1,1,1], padding="VALID", name="x_conv")

                    # nonlinear 
                    x_conv_relu = tf.nn.relu(tf.nn.bias_add(x_conv, filter_bias), name="relu")

                    # maxpool
                    x_conv_pool = tf.nn.max_pool(x_conv_relu, ksize=[1, self.sequence_len - filter_size + 1, 1, 1], strides=[1,1,1,1], padding="VALID", name="maxpool")

                    out_pools.append(x_conv_pool)

            # concatenate output of all filters (batch_size, len(filter_sizes) * num_filters)
            output = tf.squeeze(tf.concat(3, out_pools), [1, 2], name="output")

        # dropout
        with tf.name_scope("dropout"):
            output = tf.nn.dropout(output, self.dropout, name="dropout")

        # calculate logit
        with tf.name_scope("output"):
            W = tf.get_variable("W", [len(self.filter_sizes) * self.num_filters, self.num_classes], initializer=tf.truncated_normal_initializer(stddev=0.1))
            b = tf.get_variable("b", [self.num_classes], initializer=tf.constant_initializer(0.1))

            # regulation
            if self.l2_reg_lambda > 0:
                l2_loss += tf.nn.l2_loss(W)
                l2_loss += tf.nn.l2_loss(b)
                weight_decay = tf.mul(self.l2_reg_lambda, l2_loss, name="l2_loss")

                tf.add_to_collection("losses", weight_decay)

            logits = tf.nn.xw_plus_b(output, W, b, name="logit")
        return logits

    # calculate loss
    def loss(self, logits, labels):
        with tf.name_scope("loss"):
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels)
            if self.adjust_weight:
                labelids = tf.to_int32(labels)
                weights = tf.nn.embedding_lookup(self.label_weight, labelids)
                cross_entropy = tf.mul(cross_entropy, weights)
            cross_entropy_mean = tf.reduce_mean(cross_entropy)
            tf.add_to_collection("losses", cross_entropy_mean)
        total_loss = tf.add_n(tf.get_collection("losses"), name="total_loss")
        return total_loss

    # calculate acc
    def accuracy(self, logits, labels):
        prediction = tf.argmax(logits, 1, name="prediction")
        with tf.name_scope("acc"):
            correct_pred = tf.equal(tf.cast(prediction, "int32"), labels)
            acc = tf.reduce_mean(tf.cast(correct_pred, "float"), name="acc")
        return acc, prediction
