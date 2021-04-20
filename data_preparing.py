# -*- coding: utf-8 -*-

# Preparing Data

import os
import json
import pickle
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from scipy.spatial.distance import cosine

from utils import haversine

def main():

    shops = np.genfromtxt('data/shops_example.csv', delimiter=',')
    n_shops = shops.shape[0]
    shop_info = pd.read_csv('data/shop_info_example.csv')
    restrict = {}
    n_parks = {}

    for idx, shop in shop_info.iterrows():
        shop_seq = shop['SHOP_SEQ']
        is_restrict = shop['IS_RESTRICT']
        n_park = shop['PARK_NUM']

        restrict[shop_seq] = is_restrict
        n_parks[shop_seq] = n_park

    seqs = shops[:, 0].astype(np.int32)
    id2seq = {idx: seq for idx, seq in enumerate(seqs)}
    seq2id = {seq: idx for idx, seq in enumerate(seqs)}

    # Calculate Distance
    if os.path.exists('cache') == False:
            os.makedirs('cache')

    if os.path.isfile('cache/dis.npy'):
        dis = np.load('cache/dis.npy')
    else:
        dis = np.zeros((n_shops, n_shops))
        for i in range(n_shops - 1):
            for j in range(i + 1, n_shops):
                shop1, shop2 = shops[i], shops[j]
                lat1, lon1, lat2, lon2 = shop1[1], shop1[2], shop2[1], shop2[2]
                d = haversine(lon1, lat1, lon2, lat2)
                dis[i, j] = d
                dis[j, i] = d
        np.save('cache/dis.npy', dis)


    for i in range(n_shops - 1):
        for j in range(i + 1, n_shops):
            dis[i, j] = 1 / dis[i, j]
            dis[j, i] = dis[i, j]

    # Processing POI data
    if os.path.isfile('cache/poi.npy'):
        poi = np.load('cache/poi.npy')
    else:
        poi = np.zeros((n_shops, n_shops))
        for i in range(n_shops - 1):
            for j in range(i + 1, n_shops):
                shop1, shop2 = shops[i], shops[j]
                poi[i, j] = 1 - cosine(shop1[4:], shop2[4:])
                poi[j, i] = poi[i, j]
        np.save('cache/poi.npy', poi)

    # # Processing weather data
    # weather = pd.read_csv('data/weather.csv')
    # weather_out = {}

    # for i in range(len(weather)):
    #     print(weather['Date'][i],weather['Weather'][i])
    #     weather_out[weather['Date'][i]] = weather['Weather'][i]

    # pickle.dump(weather_out, open('cache/weather.pkl', 'wb'))

    # Processing online and offline events
    if os.path.isfile('cache/timeline.pkl'):
        shop_timeline = pickle.load(open('cache/timeline.pkl', 'rb'))
    else:
        operator_log = pd.read_csv('data/iss_opeartor_log_example.csv', encoding='utf-8')
        shop_timeline = {int(seq): [] for seq in seqs}

        for idx, log in operator_log.iterrows():
            print(idx)
            seq = log['shop_seq']
            content = log['operate_content']
            if seq in seqs and content in ['online', 'offline']:
                raw_dt = log['create_time']
                dt = datetime.strptime(raw_dt, '%Y/%m/%d %H:%M:%S')

                dt = datetime(dt.year, dt.month, dt.day)
                if content == 'online':
                    dt += timedelta(days=1)

                shop_timeline[seq].append((content, dt))

        for seq in shop_timeline.keys():
            timeline = shop_timeline[seq]
            shop_timeline[seq] = []

            last_status = 'online'
            last_dt = datetime(2017, 4, 1)
            for tl in timeline:
                cur_status, cur_dt = tl

                if cur_status == 'offline':
                    shop_timeline[seq].append((last_dt, cur_dt))

                last_status = cur_status
                last_dt = cur_dt
            if last_status == 'online':
                shop_timeline[seq].append((last_dt, datetime(2018, 1, 1)))

        pickle.dump(shop_timeline, open('cache/timeline.pkl', 'wb'))

    for month in [1]:

        orders = pd.read_csv('data/%d.clean.csv' % month)

        if os.path.isfile('cache/mat%d.npy' % month):
            mat = np.load('cache/mat%d.npy' % month)
            pickup_amounts = np.load('cache/pickup_amounts%d.npy' % month)
            return_amounts = np.load('cache/return_amounts%d.npy' % month)
        else:
            mat = np.zeros((32, n_shops, n_shops), dtype=np.int32)
            pickup_amounts = np.zeros((32, n_shops), dtype=np.float32)
            return_amounts = np.zeros((32, n_shops), dtype=np.float32)

            for idx, order in orders.iterrows():
                print(idx)


                try:
                    pickup_store_seq = order['PICKUP_STORE_SEQ']
                    return_store_seq = order['RETURN_STORE_SEQ']

                    pickup_datetime = str(order['PICKUPDATETIME'])[:-2]

                    dt = datetime.strptime(pickup_datetime, '%Y%m%d%H%M%S')

                    pickup_amount = order['PICKVEH_AMOUNT']
                    return_amount = order['RETURNVEH_AMOUNT']


                    mat[dt.day, seq2id[pickup_store_seq], seq2id[return_store_seq]] += 1
                    pickup_amounts[dt.day, seq2id[pickup_store_seq]] = pickup_amount
                    return_amounts[dt.day, seq2id[return_store_seq]] = return_amount

                except Exception as e:
                    print(e)
                    continue

            np.save('cache/mat%d.npy' % month, mat)
            np.save('cache/pickup_amounts%d.npy' % month, pickup_amounts)
            np.save('cache/return_amounts%d.npy' % month, return_amounts)


if __name__ == "__main__":
    main()

