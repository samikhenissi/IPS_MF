import math
import pandas as pd
from sklearn.metrics import mean_squared_error
from math import sqrt
import numpy as np


"""
 modified from : https://github.com/yihong-chen/neural-collaborative-filtering
"""


class MetronAtK(object):
    def __init__(self, top_k):
        self._top_k = top_k
        self._subjects = None  # Subjects which we ran evaluation on

    @property
    def top_k(self):
        return self._top_k

    @top_k.setter
    def top_k(self, top_k):
        self._top_k = top_k

    @property
    def subjects(self):
        return self._subjects

    @property
    def subjects_explicit(self):
        return self._subjects_explicit


    @subjects_explicit.setter
    def subjects(self, subjects):
        """
        args:
            subjects: list, [test_users, test_items, test_true,test_pred,exp_score]
        """
        assert isinstance(subjects, list)
        test_users, test_items,test_pred, test_true = subjects[0], subjects[1], subjects[2], subjects[3]

        full = pd.DataFrame({'user': test_users,
                             'test_item': test_items,
                             'test_true': test_true,
                             'test_pred': test_pred})

        # rank the items according to the scores for each user
        full['rank'] = full.groupby('user')['test_pred'].rank(method='first', ascending=False)
        full['rank_true'] = full.groupby('user')['test_true'].rank(method='first', ascending=False)

        full.sort_values(['user', 'rank'], inplace=True)
        self._subjects_explicit = full


    def cal_ndcg(self):
        topk = self._top_k

        full = self._subjects_explicit
        topp_k = full[full['rank_true']<=topk].copy()
        topp_k['idcg_unit'] = topp_k['rank_true'].apply(lambda x: math.log(2) / math.log(1 + x)) # the rank starts from 1
        topp_k['idcg'] = topp_k.groupby(['user'])['idcg_unit'].transform('sum')

        test_in_top_k =topp_k[topp_k['rank'] <=topk].copy()
        test_in_top_k['dcg_unit'] = test_in_top_k['rank'].apply(lambda x: math.log(2) / math.log(1 + x)) # the rank starts from 1
        test_in_top_k['dcg'] = test_in_top_k.groupby(['user'])['dcg_unit'].transform('sum')
        test_in_top_k['ndcg'] = test_in_top_k['dcg'] / topp_k['idcg']
        ndcg = np.sum(test_in_top_k.groupby(['user'])['ndcg'].max()) / len(full['user'].unique())
        del(topp_k,test_in_top_k)
        return ndcg

    def cal_rmse(self):
        """Hit Ratio @ top_K"""
        full = self._subjects_explicit

        return sqrt(mean_squared_error(full['test_true'], full['test_pred']))



