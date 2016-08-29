# -*- coding: utf-8 -*-
import urllib
import pandas as pd
import networkx as net
import time
import datetime
import numpy as np
from sklearn import metrics
import scipy
from scipy import sparse
from pymongo import MongoClient
import pymongo
import pickle
import matplotlib.pyplot as plt
import logging
import gc

db_pool = {}

class MongodbUtils(object):
    def __init__(self, table, collection, ip, port):
        self.table = table
        self.ip = ip
        self.port = port
        self.collection = collection
        if (ip, port) not in db_pool:
            db_pool[(ip, port)] = self.db_connection()
        self.db = db_pool[(ip, port)]
        self.db_table = self.db_table_connect()

    def __enter__(self):
        return self.db_table

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def db_connection(self):
        db = None
        try:
            db = MongoClient(self.ip, self.port)
        except Exception as e:
            print 'Can not connect mongodb'
            raise e
        return db

    def db_table_connect(self):
        table_db = self.db[self.table][self.collection]
        return table_db


def logistic_function(x):
    import math
    return np.power(1+np.exp(-x), -1)


def calculate_ks_np(y_true, y_pred):
    sort_idx = np.argsort(y_pred)[::-1]
    y_true = (y_true == 1).astype(float)
    y_true = y_true[sort_idx]
    total_true = y_true.sum()
    total_false = y_true.size - total_true
    true_cum_sum = y_true.cumsum()
    true_cum_ratio = true_cum_sum/total_true
    false_cum_ratio = (1+np.arange(len(true_cum_sum))-true_cum_sum)/total_false
    return (true_cum_ratio-false_cum_ratio).max()

def is_cellphone_china(data):
    if data.isdigit() and len(data) == 11 and data[0] == '1' and \
            (data[1] == '3' or data[1] == '4' or data[1] == '5' or data[1] == '7' or data[1] == '8') and \
            data[:5] != '13800':
        return True
    else:
        return False


def transform_rpt_data_2_sparse():
    with MongodbUtils("cp_contact_weight", "cp_contact_weight", "localhost", 27017) as coll:
        cur = coll.find({'report.updt': {'$lt': datetime.datetime(2015, 10, 1), "$gt": datetime.datetime(2015, 9, 15)}})
        total_length = cur.count()
        reg_phone_list = []
        non_reg_phone_list = []
        call_in_cnt_list = []
        call_out_cnt_list = []
        call_in_len_list = []
        call_out_len_list = []

        print "total_count %s" % total_length
        try:
            for i, v in enumerate(cur):
                if len(v.get('cell_behavior', [])) > 0:
                    phone_num = v['cell_behavior'][-1].get('phone_num', "")

                    if is_cellphone_china(phone_num):
                        updt = v['report']['updt']
                        for v_contact in v['contact_list']:
                            other_phone_num = v_contact.get('phone_num', "")
                            if is_cellphone_china(other_phone_num):
                                call_in_cnt = v_contact.get('call_in_cnt', 0)
                                call_out_cnt = v_contact.get('call_out_cnt', 0)
                                call_in_len = v_contact.get('call_in_len', 0)
                                call_out_len = v_contact.get('call_out_len', 0)
                                reg_phone_list.append(phone_num)
                                non_reg_phone_list.append(other_phone_num)
                                call_in_cnt_list.append(call_in_cnt)
                                call_in_len_list.append(call_in_len)
                                call_out_cnt_list.append(call_out_cnt)
                                call_out_len_list.append(call_out_len)

                if i % 1000 == 0:
                    print "load rate %s %s" % (i/float(total_length), datetime.datetime.now())
            done = True
        except pymongo.errors.OperationFailure as e:
            msg = e.message
            if not (msg.startswith('cursor id') and msg.endswith('not valid at server')):
                raise e
            else:
                print 'Loading interrupted, try to reload...'
        pd.to_pickle(reg_phone_list, 'reg_phone.pkl')
        pd.to_pickle(non_reg_phone_list, 'non_reg_phone.pkl')
        pd.to_pickle(call_in_cnt_list, 'call_in_cnt.pkl')
        pd.to_pickle(call_out_cnt_list, 'call_out_cnt.pkl')
        pd.to_pickle(call_in_len_list, 'call_in_len.pkl')
        pd.to_pickle(call_out_len_list, 'call_out_len.pkl')

def cut_data(rate, cp_list):
    length = len(cp_list)
    print length
    return cp_list[:int(rate*length)]

if __name__ == "__main__":
    if 'logger' not in dir():
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        logging_file_handler = logging.FileHandler("train_trustrank_weight.log")
        logging_file_handler.setFormatter(logging.Formatter('[%(asctime)s %(levelname)s]: [%(module)s.%(funcName)s()]: %(message)s'))
        logging_file_handler.setLevel(logging.DEBUG)
        logging_stream_handler = logging.StreamHandler()
        logging_stream_handler.setFormatter(logging.Formatter('[%(asctime)s %(levelname)s]: [%(module)s.%(funcName)s()]: %(message)s'))
        logging_stream_handler.setLevel(logging.DEBUG)
        logger.addHandler(logging_file_handler)
        logger.addHandler(logging_stream_handler)
    logger.info("read phone list")
    reg_call_phone = pd.read_pickle('make_call_phone.pkl')
    non_reg_call_phone = pd.read_pickle('receive_call_phone.pkl')
    logger.info('read unique_index, weight data and updt')
    weight_name = ['call_len_count', 'contact_weekday_count', "contact_night_count", "contact_3m_plus_count"]
    logger.info('weight data: %s' % weight_name)
    unique_idx = pd.read_pickle('unique_index.pkl').values
    # call_in_cnt = np.array(pd.read_pickle('call_in_cnt.pkl'))
    # call_out_cnt = np.array(pd.read_pickle('call_out_cnt.pkl'))
    # call_in_len = np.array(pd.read_pickle('call_in_len.pkl'))
    # call_out_len = np.array(pd.read_pickle('call_out_len.pkl'))
    call_len_count = np.array(pd.read_pickle("call_len_count.pkl"))
    contact_weekday_count = np.array(pd.read_pickle("contact_weekday_count.pkl"))
    contact_night_count = np.array(pd.read_pickle("contact_night_count.pkl"))

    contact_3m_plus_count = np.array(pd.read_pickle("contact_3m_plus_count.pkl"))
    updt = np.array(pd.read_pickle('updt_list.pkl'))

    logger.info('unique data')
    reg_call_phone = np.array(reg_call_phone)[unique_idx]
    non_reg_call_phone = np.array(non_reg_call_phone)[unique_idx]
    # call_in_cnt = call_in_cnt[unique_idx]
    # call_out_cnt = call_out_cnt[unique_idx]
    # call_in_len = call_in_len[unique_idx]
    # call_out_len = call_out_len[unique_idx]
    call_len_count = call_len_count[unique_idx]
    contact_weekday_count = contact_weekday_count[unique_idx]
    contact_night_count = contact_night_count[unique_idx]
    contact_3m_plus_count = contact_3m_plus_count[unique_idx]
    updt = updt[unique_idx]
    logger.info('time_interval')
    begin_date = datetime.datetime(2015, 6, 1)
    end_date = datetime.datetime(2015, 10, 1)
    logger.info("begin date: %s, end_date: %s" % (begin_date, end_date))
    time_interval_mark = (updt < end_date) & (updt > begin_date)
    reg_call_phone = reg_call_phone[time_interval_mark]
    non_reg_call_phone = non_reg_call_phone[time_interval_mark]
    updt = updt[time_interval_mark]

    call_len_count = call_len_count[time_interval_mark]
    contact_night_count = contact_night_count[time_interval_mark]
    contact_weekday_count = contact_weekday_count[time_interval_mark]
    contact_3m_plus_count = contact_3m_plus_count[time_interval_mark]
    logger.info("phone pair length: %s" % len(reg_call_phone))
    master_phone_s = pd.Series(reg_call_phone).map(unicode)
    slave_phone_s = pd.Series(non_reg_call_phone).map(unicode)
    logger.info("make index")
    nodelist = list(set(master_phone_s.values).union(set(slave_phone_s.values)))
    nlen = len(nodelist)
    index = dict(zip(nodelist, range(nlen)))
    logger.info('transfer the phone to idx')
    master_phone_idx = master_phone_s.map(index)
    slave_phone_idx = slave_phone_s.map(index)
    logger.info('nlen: %s' % nlen)
    logger.info('make column to create sparse matrix')
    master_phone = np.concatenate([master_phone_idx.values, slave_phone_idx.values])
    slave_phone = np.concatenate([slave_phone_idx.values, master_phone_idx.values])
    # call_in_cnt_weight = np.concatenate([call_in_cnt, call_out_cnt])
    # call_out_cnt_weight = np.concatenate([call_out_cnt, call_in_cnt])
    # call_in_len_weight = np.concatenate([call_in_len, call_out_len])
    # call_out_len_weight = np.concatenate([call_out_len, call_in_len])
    call_len_count_weight = np.concatenate([call_len_count, call_len_count])
    contact_weekday_count_weight = np.concatenate([contact_weekday_count, contact_weekday_count])
    contact_night_count_weight = np.concatenate([contact_night_count, contact_night_count])
    contact_3m_plus_count_weight = np.concatenate([contact_3m_plus_count, contact_3m_plus_count])
    WEIGHT = [call_len_count_weight, contact_weekday_count_weight, contact_night_count_weight, contact_3m_plus_count_weight]
    del non_reg_call_phone
    del index
    del unique_idx

    gc.collect()
    cut_date = datetime.datetime(2015, 8, 1)
    logger.info('data before %s as train data' % cut_date)
    is_train = updt < cut_date
    logger.info("length of is_train %s" % len(is_train))
    logger.info('create personalization array')
    train_reg_cp_set = reg_call_phone[is_train]
    black_reg_cp_set = pd.read_pickle('black_reg_cp_set.pkl')
    reg_cp_set = set(reg_call_phone)
    train_reg_cp_set = set(train_reg_cp_set)
    bad_seed = train_reg_cp_set.intersection(black_reg_cp_set)
    bad_seed_cp_mark = [1 for node in bad_seed]
    personalization = dict(zip(bad_seed, bad_seed_cp_mark))
    logging.info('create validation mark')
    validation_reg_cp = reg_cp_set.difference(train_reg_cp_set)
    validation_mark = pd.DataFrame({"cp": list(validation_reg_cp)})
    black_reg_cp_mark = pd.DataFrame({'cp': list(black_reg_cp_set)})
    black_reg_cp_mark['in_bl'] = 1
    validation_mark = pd.merge(validation_mark, black_reg_cp_mark, how='left', on='cp')
    validation_mark.fillna(0, inplace=True)
    logger.info("bad_seed count %s, train_reg_count: %s, validation_reg_count: %s, validation_reg_bad_count: %s" % (len(bad_seed), len(train_reg_cp_set), len(validation_mark), validation_mark['in_bl'].sum()))
    del is_train
    del train_reg_cp_set
    del black_reg_cp_mark
    del reg_cp_set
    del reg_call_phone
    del bad_seed
    del bad_seed_cp_mark
    del validation_reg_cp
    gc.collect()
    logger.info('initialize w, theta')
    w_length = len(WEIGHT)
    w = np.ones([w_length, 1])/float(w_length)
    theta = np.zeros([1, w_length])
    # w = np.array([[0.00196348], [-0.30027138], [0.15476576], [ 0.81958508]])
    # theta = np.array([[0.04022745, 0.0061195,  0.06642075,  0.17480195]])
    weight_data = np.c_[WEIGHT[0], WEIGHT[1], WEIGHT[2], WEIGHT[3]]
    theta_matrix = theta.repeat(len(weight_data), axis=0)
    logger.info('start to train weight, theta')
    for l in range(100):
        weight_data_sub_theta = weight_data - theta_matrix
        weight = logistic_function(np.dot(weight_data_sub_theta, w)).transpose()[0]
        logger.info('weight max value: %s' % weight.max())
        M_R = sparse.coo_matrix((weight, (master_phone, slave_phone)), shape=(nlen, nlen), dtype=None)
        M_R = M_R.asformat('csr')
        S = scipy.array(M_R.sum(axis=1)).flatten()
        S = S.astype(float)
        S[S != 0] = 1.0 / S[S != 0]
        Q = scipy.sparse.spdiags(S.T, 0, *M_R.shape, format='csr')
        M = Q * M_R

        M_DEV_W_LIST = []
        M_DEV_THETA_LIST = []
        logger.info('start to make m_dev_w, m_dev_theta matrix')
        for i in range(w_length):
            log_dev = weight * (1 - weight)
            # 改变算法
            f_dev_w = sparse.coo_matrix((log_dev * weight_data_sub_theta[:, i], (master_phone, slave_phone)), shape=(nlen, nlen), dtype=None)
            f_dev_theta = sparse.coo_matrix((log_dev * (-w[i, 0]), (master_phone, slave_phone)), shape=(nlen, nlen), dtype=None)

            S_W = scipy.array(f_dev_w.sum(axis=1)).flatten()
            S_W = S_W.astype(float)
            S_W_W_2 = S_W * S * S
            S_W_W_2[S_W_W_2 == np.inf] = 1e20
            Q_W_W_2 = scipy.sparse.spdiags(S_W_W_2.T, 0, *f_dev_w.shape, format='csr')
            m_dev_w = Q * f_dev_w - Q_W_W_2* M_R
            M_DEV_W_LIST.append(m_dev_w)
            S_THETA = scipy.array(f_dev_theta.sum(axis=1)).flatten()
            S_THETA = S_THETA.astype(float)
            S_THETA_S_2 = S_THETA * S * S
            S_THETA_S_2[S_THETA_S_2 == np.inf] = 1e20
            Q_THRTA_Q_2 = scipy.sparse.spdiags(S_THETA_S_2.T, 0, *f_dev_theta.shape, format='csr')
            m_dev_theta = Q * f_dev_theta - Q_THRTA_Q_2 * M_R
            M_DEV_THETA_LIST.append(m_dev_theta)
        logger.info("done")
        p = scipy.array([int(personalization.get(n, 0)) for n in nodelist], dtype=float)
        p = p / p.sum()

        x = p
        dangling_weights = p

        is_dangling = scipy.where(S == 0)[0]

        x_dev_w_list = [scipy.repeat(0, nlen) for m_dev_w in M_DEV_W_LIST]
        x_dev_theta_list = [scipy.repeat(0, nlen) for m_dev_theta in M_DEV_THETA_LIST]

        M = M.asformat("csc")
        M_DEV_W_LIST_CSC = [m_dev_w.asformat('csc') for m_dev_w in M_DEV_W_LIST]
        M_DEV_THETA_LIST_CSC = [m_dev_theta.asformat('csc') for m_dev_theta in M_DEV_THETA_LIST]
        beta = 10
        alpha = 0.7
        tol = 1e-14
        delta = 1e8
        logger.info("delta : %s" % delta)
        logger.info('start the trustrank algorithm')
        for _ in range(200):
            xlast = x
            for i in range(len(M_DEV_W_LIST)):
                # x_dev_w_list[i] = alpha * (x_dev_w_list[i] * M + x * M_DEV_W_LIST[i])
                # x_dev_theta_list[i] = alpha * (x_dev_theta_list[i] * M + x * M_DEV_THETA_LIST[i])
                x_dev_w_list[i] = alpha * (x_dev_w_list[i] * M + x * M_DEV_W_LIST_CSC[i])
                x_dev_theta_list[i] = alpha * (x_dev_theta_list[i] * M + x * M_DEV_THETA_LIST_CSC[i])
            x = alpha * (x * M + sum(x[is_dangling]) * dangling_weights) + (1 - alpha) * p
            err = scipy.absolute(x - xlast).sum()
            if _ % 10 == 0:
                logger.info('%sth iteration: %sth iteration, err %s' % (l, _, err))
            if err < tol:
                break
        logger.info('trust rank done')
        pg_df = pd.DataFrame({'pg': x}, index=nodelist)
        for j in range(len(x_dev_w_list)):
            column_name = "x_dev_w_%s" % j
            pg_df[column_name] = x_dev_w_list[j]
        for j in range(len(x_dev_theta_list)):
            column_name = "x_dev_theta_%s" % j
            pg_df[column_name] = x_dev_theta_list[j]

        y_true = validation_mark['in_bl'].astype(int).values.copy()
        y_pred = pg_df.loc[validation_mark.cp]['pg'].values.copy()*nlen

        logger.info("%s iteration: roc auc %s" % (l, metrics.roc_auc_score(y_true, y_pred)))
        precision, recall, thresh = metrics.precision_recall_curve(y_true, y_pred)
        logger.info("%s iteration: precision %s" % (l, metrics.auc(recall, precision)))
        logger.info("%s iteration: ks %s" % (l, calculate_ks_np(y_true, y_pred)))

        logger.info("start to create roc auc dev w")
        y_pred_true = y_pred[y_true == 1]
        y_pred_false = y_pred[y_true == 0]
        logger.info("make matrix")
        y_pred_true_2_false_list = np.meshgrid(y_pred_true, y_pred_false)
        y_pred_true_2_false = y_pred_true_2_false_list[0] - y_pred_true_2_false_list[1]
        logger.info("log matrix")
        y_pred_true_2_false_logistic = logistic_function(beta*y_pred_true_2_false)
        logger.info("calculate dev")
        y_pred_true_2_false_matrix = beta * y_pred_true_2_false_logistic * (1 - y_pred_true_2_false_logistic)
        logger.info("done")
        true_length = len(y_pred_true)
        false_length = len(y_pred_false)
        auc_dev_w = []
        auc_dev_theta = []

        for k in range(len(x_dev_w_list)):
            column_name = "x_dev_w_%s" % k
            logger.info("%s" % column_name)
            y_pred_dev_w = pg_df[column_name].loc[validation_mark.cp].values.copy()
            y_pred_dev_w_true = y_pred_dev_w[y_true == 1]
            y_pred_dev_w_false = y_pred_dev_w[y_true == 0]
            logger.info("make matrix")
            y_pred_dev_w_true_2_false_list = np.meshgrid(y_pred_dev_w_true, y_pred_dev_w_false)
            y_pred_dev_w_true_2_false = y_pred_dev_w_true_2_false_list[0] - y_pred_dev_w_true_2_false_list[1]
            logger.info("calculate w delta")
            auc_dev_w.append((y_pred_true_2_false_matrix * y_pred_dev_w_true_2_false).sum()/(float(true_length*false_length)))
        for k in range(len(x_dev_theta_list)):
            column_name = "x_dev_theta_%s" % k
            logger.info("%s" % column_name)
            y_pred_dev_theta = pg_df[column_name].loc[validation_mark.cp].values.copy()
            y_pred_dev_theta_true = y_pred_dev_theta[y_true == 1]
            y_pred_dev_theta_false = y_pred_dev_theta[y_true == 0]
            logger.info("make matrix")
            y_pred_dev_theta_true_2_false_list = np.meshgrid(y_pred_dev_theta_true, y_pred_dev_theta_false)
            y_pred_dev_theta_true_2_false = y_pred_dev_theta_true_2_false_list[0] - y_pred_dev_theta_true_2_false_list[1]
            logger.info("calculate w delta")
            auc_dev_theta.append((y_pred_true_2_false_matrix * y_pred_dev_theta_true_2_false).sum()/(float(true_length*false_length)))

        w = w + np.array(auc_dev_w).reshape([w_length, 1])*delta
        theta = theta + np.array(auc_dev_theta).reshape([1, w_length])*delta
        logger.info('%s iteration done' % l)


