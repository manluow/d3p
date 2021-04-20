# -*- coding: utf-8 -*-

import json
import pickle
from datetime import datetime

import numpy as np
import tensorflow as tf

from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, r2_score

from seq2seq_model import Model

class Feeder(object):

    def __init__(self, x, o, ext_inp, ext_oup, mask1, mask2, mask3, g1, g2, y):
        self.x = x
        self.o = o
        self.ext_inp = ext_inp
        self.ext_oup = ext_oup
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
        ext_oup = self.ext_oup[self.idx]

        self.idx += 1

        return x, o, mask2, g1, g2, y, ext_inp, ext_oup

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


shops = np.genfromtxt('data/shops_example.csv', delimiter=',')
n_shops = shops.shape[0]

id2seq = {idx: seq for idx, seq in enumerate(shops[:, 0].astype(np.int32))}
seq2id = {seq: idx for idx, seq in enumerate(shops[:, 0].astype(np.int32))}

# Processing online and offline events
shop_timeline = pickle.load(open('cache/timeline.pkl', 'rb'))


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


print('Build networks...')

model = Model(predict=True)
saver = tf.train.Saver()

sess = tf.Session()
saver.restore(sess, './checkpoint/model.new.ckpt')

print('Predict...')

dis = np.load('cache/dis.npy')
n_shops = dis.shape[0]
for i in range(n_shops - 1):
    for j in range(i + 1, n_shops):
        dis[i, j] = 1 / dis[i, j]
        dis[j, i] = dis[i, j]
poi = np.load('cache/poi.npy')

x = np.load('dev/test/x.npy')
o = np.load('dev/test/o.npy')
ext_inp = np.load('dev/test/ext_inp.npy')
ext_oup = np.load('dev/test/ext_oup.npy')
mask1 = np.load('dev/test/mask1.npy')
mask2 = np.load('dev/test/mask2.npy')
mask3 = np.load('dev/test/mask3.npy')
y = np.load('dev/test/y.npy')
val_feeder = Feeder(x, o, ext_inp, ext_oup, mask1, mask2, mask3, dis, poi, y)

total_loss = []
total_new_loss = []
total_norm_loss = []
meta = []
norm_meta = []
for idx in range(val_feeder.step_per_epoch):
    print(idx + 1)
    in1, in2, in3, in4, in5, in6, in7, in8 = val_feeder.feed()
    loss, pred = sess.run([model.loss, model.preds],
                          {model.x: in1, model.o: in2, model.mask: in3, model.g1: in4, model.g2: in5, model.y: in6,
                           model.ext_inp: in7, model.ext_oup: in8})
    total_loss.append(loss)

    r1, r2 = fa(1, idx + 15)

    for shop_id in r1:
        if in3[shop_id, 0]:
            y_true = np.squeeze(in6[shop_id], -1)[1:]
            y_pred = np.squeeze(pred[shop_id], -1)
            y_mask = np.squeeze(in3[shop_id], -1)
            n_valid = np.count_nonzero(y_mask)
            total_new_loss.append(np.sum(pow(y_true - y_pred, 2) * y_mask) / n_valid)
            meta.append((idx + 1, shop_id, id2seq[shop_id], y_true, y_pred))

            norm_y_pred = y_pred.copy()
            norm_y_pred[norm_y_pred < 0] = 0

            # check mask
            if np.sum(in2[shop_id]) == -14.0:
                total_norm_loss.append(np.sum(pow(y_true - norm_y_pred, 2) * y_mask) / n_valid)
                norm_meta.append((idx + 1, shop_id, id2seq[shop_id], y_true, norm_y_pred))

print(len(total_loss))
print(sum(total_loss) / len(total_loss))
print(sum(total_new_loss) / len(total_new_loss))
print(sum(total_norm_loss) / len(total_norm_loss))

y_true = np.concatenate([m[3] for m in norm_meta])
y_pred = np.concatenate([m[4] for m in norm_meta])

print(r2_score(y_true, y_pred))
print(pearsonr(y_true, y_pred)[0])
