# -*- coding: utf-8 -*-


import os
import json
import pickle
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from utils import day_of_month

def is_first_day(shop_timeline, seq, dt):
    '''Jude whether a day is the first day of the sequence
    '''
    timeline = shop_timeline[seq]
    for tl in timeline:
        if tl[0] <= dt < tl[1] and tl[0] == dt:
            return True
    return False


def is_open(shop_timeline, seq, dt):
    '''Jude whether a day is still open at the specified day
    '''
    timeline = shop_timeline[seq]
    for tl in timeline:
        if tl[0] <= dt < tl[1]:
            return True
    return False


def get_prev(shop_id, cache_orders, cache_masks, cache_ext, cur_mon, cur_day, start_mon, n_day):
    '''Get the weekly average number of orders in previous n_days

    Arg:
        shop_id: the id of the station
        cur_mon: the current month
        cur_day: the current day
        start_mon : the start month
        n_day: the number of days will be calculated
    '''
    end_idx = cur_day - 1
    for i in range(start_mon, cur_mon):
        end_idx += day_of_month(i)

    start_idx = end_idx - n_day
    r_orders = cache_orders[shop_id, start_idx: end_idx]
    r_masks = cache_masks[shop_id, start_idx: end_idx]
    r_ext = cache_ext[shop_id, start_idx: end_idx]

    return r_orders, r_masks, r_ext


def get_next(shop_id, cache_orders, cache_masks, cache_ext, cur_mon, cur_day, start_mon, n_day=14):
    '''Get the weekly average number of orders in next n_days

    Arg:
        shop_id: the id of the station
        cur_mon: the current month
        cur_day: the current day
        start_mon : the start month
        n_day: the number of days will be calculated
    '''
    start_idx = cur_day - 1

    for i in range(start_mon, cur_mon):
        start_idx += day_of_month(i)

    end_idx = start_idx + n_day

    r_orders = cache_orders[shop_id, start_idx: end_idx]
    r_masks = cache_masks[shop_id, start_idx: end_idx]
    r_ext = cache_ext[shop_id, start_idx: end_idx]

    avg_orders = np.zeros([7, ], dtype=np.float32)
    valid_days = np.zeros([7, ], dtype=np.float32)

    cur_dt = datetime(1991, cur_mon, cur_day)
    delta = timedelta(days=1)

    for i in range(n_day):
        if r_masks[i]:
            wd = cur_dt.weekday()
            avg_orders[wd] += r_orders[i]
            valid_days[wd] += 1
        cur_dt += delta

    avg_orders = avg_orders / valid_days
    avg_orders[np.where(valid_days == 0)] = 0

    r_masks = np.ones(shape=[7, ], dtype=np.float32)
    r_masks[np.where(valid_days == 0)] = 0

    return avg_orders, r_masks, r_ext

def main():
    shops = np.genfromtxt('data/shops_example.csv', delimiter=',')

    n_shops = shops.shape[0]

    id2seq = {idx: seq for idx, seq in enumerate(shops[:, 0].astype(np.int32))}
    seq2id = {seq: idx for idx, seq in enumerate(shops[:, 0].astype(np.int32))}

    # Processing online and offline events
    shop_timeline = pickle.load(open('cache/timeline.pkl', 'rb'))

    shop_info = pd.read_csv('data/shop_info_example.csv')

    restrict = {}
    n_parks = {}
    for idx, shop in shop_info.iterrows():
        shop_seq = shop['SHOP_SEQ']
        is_restrict = shop['IS_RESTRICT']
        n_park = shop['PARK_NUM']

        restrict[shop_seq] = is_restrict
        n_parks[shop_seq] = n_park

    dis = np.load('cache/dis.npy')

    for i in range(n_shops - 1):
        for j in range(i + 1, n_shops):
            dis[i, j] = 1 / dis[i, j]
            dis[j, i] = dis[i, j]

    poi = np.load('cache/poi.npy')

    # Prepare the training dataset
    cache_orders = []
    cache_masks = []
    cache_ext = []
    for month in [1]:
        mats = np.load('cache/mat%d.npy' % month)

        for day in range(1, day_of_month(month) + 1):
            tmp_order = np.full((n_shops,), -1.0, dtype=np.float32)
            tmp_mask = np.full((n_shops,), 0.0, dtype=np.float32)
            tmp_ext = np.zeros((n_shops, 7), dtype=np.float32)

            mat = mats[day]
            n_returns = np.sum(mat, 1)
            cur_dt = datetime(1991, month, day)

            for i in range(n_shops):
                seq = id2seq[i]

                if not is_open(shop_timeline, seq, cur_dt):
                    continue

                n_return = n_returns[i]
                tmp_order[i] = n_return
                tmp_mask[i] = 1.0
                tmp_ext[i, cur_dt.weekday()] = 1.0

            cache_orders.append(tmp_order)
            cache_masks.append(tmp_mask)
            cache_ext.append(tmp_ext)

    cache_orders = np.transpose(np.array(cache_orders))
    cache_masks = np.transpose(np.array(cache_masks))
    cache_ext = np.transpose(np.array(cache_ext), [1, 0, 2])

    x = [] # The input
    o = [] # The number of orders in previous weeks


    ext_inp = []  # The day of the week

    mask1 = []  # The mask of the graph
    mask2 = []  # The mask of the prediction
    mask3 = []  # The mask of the new stations
    y = [] # The prediction

    for month in [1]:
        pickup_amounts = np.load('cache/pickup_amounts%d.npy' % month)
        return_amounts = np.load('cache/return_amounts%d.npy' % month)

        for day in range(15, day_of_month(month) + 1):
            print(month, day)
            pickup_amount = pickup_amounts[day]
            return_amount = return_amounts[day]

            cur_dt = datetime(1991, month, day)

            # Select the data from 1st, Jan, 1991 to 16th, Jan, 1991
            if datetime(1991, 1, 1) <= cur_dt < datetime(1991, 1, 16):

                t_x = np.zeros((n_shops, 649), dtype=np.float32)
                t_o = np.zeros((n_shops, 14, 1), dtype=np.float32)
                t_ext_inp = np.zeros((n_shops, 14, 7), dtype=np.float32)
                t_y = np.zeros((n_shops, 7, 1), dtype=np.float32)
                t_mask1 = np.zeros((n_shops, 1), dtype=np.float32)
                t_mask2 = np.zeros((n_shops, 7, 1), dtype=np.float32)
                t_mask3 = np.zeros((n_shops, 1), dtype=np.float32)

                for i in range(n_shops):
                    seq = id2seq[i]

                    n_park = n_parks[seq]
                    n_pickup_amount = pickup_amount[i]
                    n_return_amount = return_amount[i]

                    t_x[i, :646] = shops[i, 227:]
                    t_x[i, 646] = n_pickup_amount
                    t_x[i, 647] = n_return_amount
                    t_x[i, 648] = n_park

                    if is_first_day(shop_timeline, seq, cur_dt):
                        t_mask3[i] = 1.0


                    a, _, c = get_prev(i,cache_orders, cache_masks, cache_ext, month, day, 1, 14)
                    t_o[i] = np.expand_dims(a, 1)
                    t_mask1[i] = 1.0
                    t_ext_inp[i] = c


                    a, b, c = get_next(i, cache_orders, cache_masks, cache_ext, month, day, 1)
                    t_y[i] = np.expand_dims(a, 1)
                    t_mask2[i] = np.expand_dims(b, 1)


                x.append(t_x)
                o.append(t_o)
                ext_inp.append(t_ext_inp)
                mask1.append(t_mask1)
                mask2.append(t_mask2)
                mask3.append(t_mask3)
                y.append(t_y)

    x = np.array(x)
    o = np.array(o)
    ext_inp = np.array(ext_inp)
    mask1 = np.array(mask1)
    mask2 = np.array(mask2)
    mask3 = np.array(mask3)
    y = np.array(y)

    if os.path.exists('dev/train') == False:
        os.makedirs('dev/train')


    np.save('dev/train/x', x)
    np.save('dev/train/o', o)
    np.save('dev/train/ext_inp', ext_inp)
    np.save('dev/train/mask1', mask1)
    np.save('dev/train/mask2', mask2)
    np.save('dev/train/mask3', mask3)
    np.save('dev/train/y', y)

    # Prepare the testing dataset
    cache_orders = []
    cache_masks = []
    cache_ext = []

    for month in [1]:
        mats = np.load('cache/mat%d.npy' % month)

        for day in range(1, day_of_month(month) + 1):
            tmp_order = np.full((n_shops,), -1.0, dtype=np.float32)
            tmp_mask = np.full((n_shops,), 0.0, dtype=np.float32)
            tmp_ext = np.zeros((n_shops, 7), dtype=np.float32)

            mat = mats[day]
            n_returns = np.sum(mat, 1)
            cur_dt = datetime(1991, month, day)

            for i in range(n_shops):
                seq = id2seq[i]

                if not is_open(shop_timeline, seq, cur_dt):
                    continue

                n_return = n_returns[i]
                tmp_order[i] = n_return
                tmp_mask[i] = 1.0
                tmp_ext[i, cur_dt.weekday()] = 1.0

            cache_orders.append(tmp_order)
            cache_masks.append(tmp_mask)
            cache_ext.append(tmp_ext)

    cache_orders = np.transpose(np.array(cache_orders))
    cache_masks = np.transpose(np.array(cache_masks))
    cache_ext = np.transpose(np.array(cache_ext), [1, 0, 2])

    x = [] # The input
    o = [] # The number of orders in previous weeks
    ext_inp = []  # The day of the week
    mask1 = []  # The mask of the graph
    mask2 = []  # The mask of the prediction
    mask3 = []  # The mask of the new stations
    y = [] # The prediction

    for month in [1]:
        pickup_amounts = np.load('cache/pickup_amounts%d.npy' % month)
        return_amounts = np.load('cache/return_amounts%d.npy' % month)

        # 1 - 30
        for day in range(15, day_of_month(month) + 1):
            print(month, day)
            pickup_amount = pickup_amounts[day]
            return_amount = return_amounts[day]

            cur_dt = datetime(1991, month, day)

            # Select the data from 1st, Jan, 1991 to 16th, Jan, 1991
            if datetime(1991, 1, 1) <= cur_dt < datetime(1991, 1, 16):

                t_x = np.zeros((n_shops, 649), dtype=np.float32)
                t_o = np.zeros((n_shops, 14, 1), dtype=np.float32)
                t_y = np.zeros((n_shops, 7, 1), dtype=np.float32)
                t_ext_inp = np.zeros((n_shops, 14, 7), dtype=np.float32)
                t_mask1 = np.zeros((n_shops, 1), dtype=np.float32)
                t_mask2 = np.zeros((n_shops, 7, 1), dtype=np.float32)
                t_mask3 = np.zeros((n_shops, 1), dtype=np.float32)

                for i in range(n_shops):
                    seq = id2seq[i]

                    n_park = n_parks[seq]
                    n_pickup_amount = pickup_amount[i]
                    n_return_amount = return_amount[i]

                    t_x[i, :646] = shops[i, 227:]  # 1~4, 224~227
                    t_x[i, 646] = n_pickup_amount
                    t_x[i, 647] = n_return_amount
                    t_x[i, 648] = n_park

                    if is_first_day(shop_timeline, seq, cur_dt):
                        t_mask3[i] = 1.0

                    a, _, c = get_prev(i, cache_orders, cache_masks, cache_ext, month, day, 1, 14)
                    t_o[i] = np.expand_dims(a, 1)
                    t_mask1[i] = 1.0
                    t_ext_inp[i] = c

                    a, b, c = get_next(i, cache_orders, cache_masks, cache_ext, month, day, 1)
                    t_y[i] = np.expand_dims(a, 1)
                    t_mask2[i] = np.expand_dims(b, 1)

                x.append(t_x)
                o.append(t_o)
                ext_inp.append(t_ext_inp)
                mask1.append(t_mask1)
                mask2.append(t_mask2)
                mask3.append(t_mask3)
                y.append(t_y)

    x = np.array(x)
    o = np.array(o)
    mask1 = np.array(mask1)
    mask2 = np.array(mask2)
    mask3 = np.array(mask3)
    y = np.array(y)

    if os.path.exists('dev/test') == False:
        os.makedirs('dev/test')

    np.save('dev/test/x', x)
    np.save('dev/test/o', o)
    np.save('dev/test/ext_inp', ext_inp)
    np.save('dev/test/mask1', mask1)
    np.save('dev/test/mask2', mask2)
    np.save('dev/test/mask3', mask3)
    np.save('dev/test/y', y)

if __name__ == "__main__":
    main()




