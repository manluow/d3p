# -*- coding: utf-8 -*-

import tensorflow as tf

STATION_NUM = 16


class Model(object):

    def __init__(self, predict=False):
        self.x = tf.placeholder(shape=(None, 649), dtype=tf.float32)
        self.o = tf.placeholder(shape=(None, 14, 1), dtype=tf.float32)
        self.ext_inp = tf.placeholder(shape=(None, 14, 7), dtype=tf.float32)
        self.mask = tf.placeholder(shape=(None, 7, 1), dtype=tf.float32)
        self.g1 = tf.placeholder(shape=(None, None), dtype=tf.float32)
        self.g2 = tf.placeholder(shape=(None, None), dtype=tf.float32)
        self.y = tf.placeholder(shape=(None, 7, 1), dtype=tf.float32)  # Monday to Sunday

        self.predict = predict

        self.build_net()

    def _add_lstm(self):
        rnn_layers = [tf.nn.rnn_cell.GRUCell(size, kernel_initializer=tf.contrib.layers.xavier_initializer())
                      for size in [64, 64]]
        multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(rnn_layers)

        inputs = tf.concat([self.o, self.ext_inp], -1)

        outputs, states = tf.nn.dynamic_rnn(cell=multi_rnn_cell, inputs=inputs, dtype=tf.float32)

        self.memory = outputs
        self.lstm_output = outputs[:, -1, :]

 

    def _add_gcn(self):
        # Convolution on POI, and do not making Convolution on temporal sequence

        with tf.variable_scope('spatial_gcn_layer'):
            with tf.variable_scope('gcn_vars'):
                w1 = tf.get_variable('w1', shape=(649, 64), dtype=tf.float32,
                                     initializer=tf.contrib.layers.xavier_initializer())
                w2 = tf.get_variable('w2', shape=(64, 64), dtype=tf.float32,
                                     initializer=tf.contrib.layers.xavier_initializer())

            # For time
            g1, g2 = self.g1, self.g2
            g1 = g1 + tf.nn.dropout(tf.eye(STATION_NUM), 1.0)
            g2 = g2 + tf.nn.dropout(tf.eye(STATION_NUM), 1.0)


            with tf.variable_scope('time_gcn'):
                x = self.lstm_output
                pre_sup = tf.matmul(x, w2)

                sups = [tf.matmul(g1, pre_sup), tf.matmul(g2, pre_sup)]
                output1 = tf.nn.relu(tf.add_n(sups))

            # For poi
            g1, g2 = self.g1, self.g2
            g1 = g1 + tf.eye(STATION_NUM)
            g2 = g2 + tf.eye(STATION_NUM)

            with tf.variable_scope('poi_layer1'):
                x = self.x
                pre_sup = tf.matmul(x, w1) 

                sups = [tf.matmul(g1, pre_sup), tf.matmul(g2, pre_sup)]
                output2 = tf.nn.relu(tf.add_n(sups))

            self.gcn_output = tf.concat([output1, output2], -1)

    def _add_decoder(self):
        o_w = tf.get_variable('output_projection_w', shape=(128, 7), dtype=tf.float32,
                              initializer=tf.contrib.layers.xavier_initializer())
        o_b = tf.get_variable('output_projection_b', shape=(1,), dtype=tf.float32,
                              initializer=tf.contrib.layers.xavier_initializer())

        final_outputs = self.gcn_output
        self.preds = tf.expand_dims(tf.einsum('ij,jk->ik', final_outputs, o_w) + o_b,
                                    2)


    def _add_loss(self):
        count = tf.cast(tf.count_nonzero(self.mask), tf.float32)
        tgt_oup = self.y
        preds = self.preds
        loss = tf.reduce_sum(tf.pow((preds - tgt_oup) * self.mask, 2)) / count

        # Add l2 loss
        vars = tf.trainable_variables()
        l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in vars]) * 0.001
        self.loss = loss + l2_loss

    def _add_train_op(self):
        optimizer = tf.train.AdamOptimizer()
        self.train_op = optimizer.minimize(self.loss)

    def build_net(self):
        self._add_lstm()
        self._add_gcn()
        self._add_decoder()
        self._add_loss()
        self._add_train_op()
