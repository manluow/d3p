# -*- coding: utf-8 -*-

import tensorflow as tf

STATION_NUM = 16


class Model(object):

    def __init__(self, predict=False):
        self.x = tf.placeholder(shape=(None, 649), dtype=tf.float32)
        self.o = tf.placeholder(shape=(None, 14, 1), dtype=tf.float32)
        self.ext_inp = tf.placeholder(shape=(None, 14, 7), dtype=tf.float32)
        self.ext_oup = tf.placeholder(shape=(None, 7, 7), dtype=tf.float32)
        self.mask = tf.placeholder(shape=(None, 7, 1), dtype=tf.float32)
        self.g1 = tf.placeholder(shape=(None, None), dtype=tf.float32)
        self.g2 = tf.placeholder(shape=(None, None), dtype=tf.float32)
        self.y = tf.placeholder(shape=(None, 8, 1), dtype=tf.float32)

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

    def _process_memory(self):
        with tf.variable_scope('memory_gcn_layer'):
            with tf.variable_scope('gcn_vars'):
                w1 = tf.get_variable('w1', shape=(64, 64), dtype=tf.float32,
                                     initializer=tf.contrib.layers.xavier_initializer())

            g1, g2 = self.g1, self.g2

            # Layer1
            with tf.variable_scope('gcn_layer1'):
                x = self.memory
                pre_sup = tf.einsum('aij,jk->aik', x, w1)

                sups = [tf.einsum('ab,bij->aij', g1, pre_sup), tf.einsum('ab,bij->aij', g2, pre_sup)]
                output = tf.nn.relu(tf.add_n(sups))

            self.memory = output

        # Add LSTM
        with tf.variable_scope("lstm_above_gcn"):
            rnn_layers = [tf.nn.rnn_cell.GRUCell(size, kernel_initializer=tf.contrib.layers.xavier_initializer())
                          for size in [64, 64]]
            multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(rnn_layers)

            inputs = output

            outputs, states = tf.nn.dynamic_rnn(cell=multi_rnn_cell, inputs=inputs, dtype=tf.float32)
            self.memory = outputs
            self.lstm_output = outputs[:, -1, :]
        # End

 

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
            g1 = g1 + tf.nn.dropout(tf.eye(STATION_NUM), 0.5)
            g2 = g2 + tf.nn.dropout(tf.eye(STATION_NUM), 0.5)


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
        o_w = tf.get_variable('output_projection_w', shape=(192, 1), dtype=tf.float32,
                              initializer=tf.contrib.layers.xavier_initializer())
        o_b = tf.get_variable('output_projection_b', shape=(1,), dtype=tf.float32,
                              initializer=tf.contrib.layers.xavier_initializer())

        att_w = tf.get_variable('att_w', shape=(128, 64), dtype=tf.float32,
                                initializer=tf.contrib.layers.xavier_initializer())
        att_v = tf.get_variable('att_v', shape=(64, 1), dtype=tf.float32,
                                initializer=tf.contrib.layers.xavier_initializer())

        def loop_fn(prev, i):
            a = tf.reshape(prev, [-1, 1, 128])
            b = tf.reshape(self.memory, [-1, 14, 64])

            a = tf.einsum('abc,cd->abd', a, att_w)
            c = tf.nn.tanh(a * b)
            c = tf.squeeze(tf.einsum('abc,cd->abd', c, att_v), -1)

            h = tf.expand_dims(tf.nn.softmax(c, -1), 1)
            s = tf.einsum('aij,ajk->aik', h, self.memory)

            s = tf.squeeze(s, 1)

            prev = tf.concat((prev, s), -1)

            r_next = tf.einsum('ij,jk->ik', prev, o_w) + o_b
            r_next = tf.concat([r_next, self.ext_oup[:, i, :]], -1)

            return r_next

        rnn_layers = [tf.nn.rnn_cell.GRUCell(size, kernel_initializer=tf.contrib.layers.xavier_initializer())
                      for size in [128, 128]]
        multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(rnn_layers)

        enc_inp = self.y[:, :7]

        # Add weekday information
        enc_inp = tf.concat([enc_inp, self.ext_oup], -1)
        # End

        enc_inp = tf.unstack(enc_inp, axis=1)

        initial_state = (self.gcn_output, self.gcn_output)

        lf = None
        if self.predict:
            lf = loop_fn

        outputs, _ = tf.contrib.legacy_seq2seq.rnn_decoder(enc_inp,
                                                           initial_state,
                                                           multi_rnn_cell,
                                                           lf)

        final_outputs = tf.stack(outputs, axis=1)

        a = tf.reshape(final_outputs, [-1, 7, 1, 128])
        b = tf.reshape(self.memory, [-1, 1, 14, 64])

        a = tf.einsum('abcd,de->abce', a, att_w)
        c = tf.nn.tanh(a * b)
        c = tf.squeeze(tf.einsum('abcd,de->abce', c, att_v), -1)

        h = tf.nn.softmax(c, -1)
        s = tf.einsum('aij,ajk->aik', h, self.memory)

        final_outputs = tf.concat((final_outputs, s), -1)
        # End

        self.preds = tf.einsum('ijk,kl->ijl', final_outputs, o_w) + o_b


    def _add_loss(self):
        count = tf.cast(tf.count_nonzero(self.mask), tf.float32)
        tgt_oup = self.y[:, 1:]
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
        self._process_memory()
        self._add_gcn()
        self._add_decoder()
        self._add_loss()
        self._add_train_op()
