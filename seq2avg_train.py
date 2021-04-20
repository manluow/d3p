# -*- coding: utf-8 -*-
import os
import time
import json
import pickle
import logging
from datetime import datetime

import numpy as np
import tensorflow as tf

from seq2avg_model import Model


class Feeder(object):

    def __init__(self, x, o, ext_inp,
                 mask1, mask2, mask3, g1, g2, y):
        self.x = x
        self.o = o
        self.ext_inp = ext_inp
        self.mask1 = mask1
        self.mask2 = mask2
        self.mask3 = mask3
        self.g1 = g1
        self.g2 = g2
        self.y = y

        self.size = o.shape[0]
        self.idx = 0
        self.epoch = 1
        self.step_per_epoch = self.size

    def normalize_adj(self, adj):
        """Symmetrically normalize adjacency matrix."""
        d = np.sum(adj, 1)
        d_inv_sqrt = np.power(d, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = np.diagflat(d_inv_sqrt)
        return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)

    def preprocess_adj(self, adj, mask):
        adj_normalized = self.normalize_adj(adj)
        return adj_normalized

    def preprocess_poi(self, adj, mask):
        adj_normalized = self.normalize_adj(adj)
        return adj_normalized

    def mask_graph(self, graph, mask):
        idx = np.where(mask == 0)[0]
        graph[idx, :] = 0
        graph[:, idx] = 0
        return graph

    def feed(self):
        x = self.x[self.idx]
        o = self.o[self.idx]
        mask1 = self.mask1[self.idx]
        mask2 = self.mask2[self.idx]
        mask3 = self.mask3[self.idx]

        g1 = self.mask_graph(self.g1.copy(), mask1)
        g2 = self.mask_graph(self.g2.copy(), mask1)

        g1 = self.preprocess_adj(g1, mask3)
        g2 = self.preprocess_poi(g2, mask3)
        y = self.y[self.idx]

        ext_inp = self.ext_inp[self.idx]

        self.idx += 1

        return x, o, mask2, g1, g2, y, ext_inp

    def shuffle(self):
        self.idx = 0
        self.epoch += 1

    def debug(self, idx=None):
        if idx is None:
            idx = self.size - 1

        x = self.x[idx]
        o = self.o[idx]
        mask1 = self.mask1[idx]
        mask2 = self.mask2[idx]

        g1 = self.mask_graph(self.g1.copy(), mask1)
        g2 = self.mask_graph(self.g2.copy(), mask1)

        g1 = self.preprocess_adj(g1)
        g2 = self.preprocess_poi(g2)

        y = self.y[idx]

        return x, o, mask2, g1, g2, y


def is_first_day(seq, dt):
    timeline = shop_timeline[seq]
    for tl in timeline:
        if tl[0] <= dt < tl[1] and tl[0] == dt:
            return True
    return False


def fa(month, day):
    dt = datetime(1991, month, day)
    r1 = []
    r2 = []
    for seq in shops[:, 0].astype(np.int32):
        if is_first_day(seq, dt):
            r1.append(seq2id[seq])
            r2.append(seq)
    return r1, r2


logger = logging.getLogger('GCN')
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter(
    fmt='%(asctime)s %(message)s',
    datefmt='%m-%d %H:%M'
)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(formatter)
logger.addHandler(ch)

if os.path.exists('logs') == False:
        os.makedirs('logs')

fh = logging.FileHandler('logs/train-%s.log' % (time.strftime('%Y%m%d%H%M%S', time.localtime())), encoding='utf-8')
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)
logger.addHandler(fh)

logger.info('Load data...')

shops = np.genfromtxt('data/shops_example.csv', delimiter=',')

n_shops = shops.shape[0]

id2seq = {idx: seq for idx, seq in enumerate(shops[:, 0].astype(np.int32))}
seq2id = {seq: idx for idx, seq in enumerate(shops[:, 0].astype(np.int32))}

# Processing online and offline events
shop_timeline = pickle.load(open('cache/timeline.pkl', 'rb'))


shop2valid = {}
for day in range(1, 32):
    r1, r2 = fa(1, day)
    shop2valid[day] = r1

logger.info('Build networks...')

model = Model()

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

saver = tf.train.Saver()

logger.info('Train...')

dis = np.load('cache/dis.npy')
n_shops = dis.shape[0]
for i in range(n_shops - 1):
    for j in range(i + 1, n_shops):
        dis[i, j] = 1 / dis[i, j]
        dis[j, i] = dis[i, j]
poi = np.load('cache/poi.npy')

x = np.load('dev/train/x.npy')
o = np.load('dev/train/o.npy')
ext_inp = np.load('dev/train/ext_inp.npy')
mask1 = np.load('dev/train/mask1.npy')
mask2 = np.load('dev/train/mask2.npy')
mask3 = np.load('dev/train/mask3.npy')
y = np.load('dev/train/y.npy')
train_feeder = Feeder(x, o, ext_inp,
                      mask1, mask2, mask3, dis, poi, y)

x = np.load('dev/test/x.npy')
o = np.load('dev/test/o.npy')
ext_inp = np.load('dev/test/ext_inp.npy')
mask1 = np.load('dev/test/mask1.npy')
mask2 = np.load('dev/test/mask2.npy')
mask3 = np.load('dev/test/mask3.npy')
y = np.load('dev/test/y.npy')
val_feeder = Feeder(x, o, ext_inp,
                    mask1, mask2, mask3, dis, poi, y)

best_loss = 1000000
best_new_loss = 1000000

train_epoch = 200
for i in range(train_epoch):
    total_loss = 0
    for _ in range(train_feeder.step_per_epoch):
        in1, in2, in3, in4, in5, in6, in7 = train_feeder.feed()
        loss, _ = sess.run([model.loss, model.train_op],
                           {model.x: in1, model.o: in2, model.mask: in3, model.g1: in4, model.g2: in5, model.y: in6,
                            model.ext_inp: in7})
        total_loss += loss
        logger.info('Step %d/%d - mse: %f' % (train_feeder.idx, train_feeder.size, loss))

    logger.info(
        'Epoch %d/%d, train loss %f' % (train_feeder.epoch, train_epoch, total_loss / train_feeder.step_per_epoch))

    total_loss = []
    total_new_loss = []
    for idx in range(val_feeder.step_per_epoch):
        in1, in2, in3, in4, in5, in6, in7 = val_feeder.feed()
        loss, pred = sess.run([model.loss, model.preds],
                              {model.x: in1, model.o: in2, model.mask: in3, model.g1: in4, model.g2: in5, model.y: in6,
                               model.ext_inp: in7})
        total_loss.append(loss)

        #Loss of the new stations
        for shop_id in shop2valid[idx + 15]:
            print(shop_id)
            if in3[shop_id, 0]:
                y_true = np.squeeze(in6[shop_id], -1)
                y_pred = np.squeeze(pred[shop_id], -1)
                y_mask = np.squeeze(in3[shop_id], -1)
                n_valid = np.count_nonzero(y_mask)
                total_new_loss.append(np.sum(pow(y_true - y_pred, 2) * y_mask) / n_valid)

    print(len(total_loss))

    dev_loss = sum(total_loss) / len(total_loss)
    if dev_loss < best_loss:
        best_loss = dev_loss
        saver.save(sess, "./checkpoint/model.ckpt")

    print(len(total_new_loss))

    dev_new_loss = sum(total_new_loss) / len(total_new_loss)
    if dev_new_loss < best_new_loss:
        best_new_loss = dev_new_loss
        saver.save(sess, "./checkpoint/model.new.ckpt")

    logger.info('Epoch %d/%d, val loss %f, best loss %f' % (train_feeder.epoch, train_epoch, dev_loss, best_loss))
    logger.info('Epoch %d/%d, val new loss %f' % (train_feeder.epoch, train_epoch, dev_new_loss))

    val_feeder.shuffle()

    logger.info('')

    train_feeder.shuffle()

