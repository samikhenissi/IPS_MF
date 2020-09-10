import pandas as pd
import numpy as np
from hpfrec import HPF



"""
Build the Propensity Model. For the moment we use popularity based model and poisson based model

To create a Propensity instance you can use: 
propensity = Propensity()
To specify the model:
propensity = Propensity(propensity_model = poisson)

To fit a propensity model:
propensity.fit(data)

To predict on a new data:
propensity.predict(newdata)

TODO: Add new models
TODO: Improve wrapper

"""

class Popularity():

    def __init__(self ,config):
        self.num_users = config['num_users']
        self.num_items = config['num_items']

    def fit(self,train_data):
        column_names = train_data.columns
        userid = column_names[0]
        itemid = column_names[1]
        ratings = column_names[2]
        self.counts_df = train_data.rename(columns = {userid:"UserId",itemid:"ItemId",ratings:"Count"})
        self.counts_df["Count"] = 1
        self.counts_df['popularity_score'] = self.counts_df.groupby(["ItemId"])['Count'].transform(sum) / self.num_users
        self.counts_df = self.counts_df[['ItemId','popularity_score']].drop_duplicates().set_index('ItemId').to_dict()


    def predict(self,test_data):
        column_names = test_data.columns
        userid = column_names[0]
        itemid = column_names[1]
        ratings = column_names[2]
        self.test_data = test_data.rename(columns = {userid:"UserId",itemid:"ItemId",ratings:"Count"})
        prediction = self.test_data['ItemId'].map(self.counts_df['popularity_score']).fillna(1/(self.num_items+1)).values
        return prediction


class Propensity():

    def __init__(self ,config):
        self.num_users = config['num_users']
        self.num_items = config['num_items']
        self.method = config['propensity_model']
        if self.method == 'poisson':
            self.model = HPF(k=10,check_every=10, ncores=-1, maxiter=150)
        if self.method == 'popularity':
            self.model = Popularity(config)

    def fit(self,train_data):
        column_names = train_data.columns
        userid = column_names[0]
        itemid = column_names[1]
        ratings = column_names[2]
        self.counts_df = train_data.rename(columns = {userid:"UserId",itemid:"ItemId",ratings:"Count"})
        self.counts_df["Count"] = 1
        self.model.fit(self.counts_df)

    def predict(self,test_data):
        column_names = test_data.columns
        userid = column_names[0]
        itemid = column_names[1]
        ratings = column_names[2]
        self.test_data = test_data.rename(columns = {userid:"UserId",itemid:"ItemId",ratings:"Count"})
        if self.method == 'poisson':
            prediction = self.model.predict(self.test_data["UserId"].values,self.test_data["ItemId"].values)
            return 1 - np.exp(-prediction)
        else:
            prediction = self.model.predict(self.test_data)
            return prediction

