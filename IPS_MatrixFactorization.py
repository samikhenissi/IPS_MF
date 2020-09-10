
from MatrixFactorization import *
from Dataset import *

class IPSMF(MatrixFactorization):
    def __init__(self, config):
        super(IPSMF, self).__init__(config)

    def training_step(self,batch,batch_idx):
        user,items,ratings,prop_score = batch[0],batch[1], batch[2],batch[3]
        ratings_pred = self(user,items)
        loss = (1/prop_score) * mse_loss(ratings_pred.view(-1), ratings,reduction= 'none')

        return {'loss':loss.mean()}


    def validation_step(self,batch,batch_idx):
        test_users, test_items , test_true,prop_score = batch[0],batch[1], batch[2] , batch[3]
        test_pred = self(test_users, test_items)
        loss =  mse_loss(test_pred.view(-1), test_true)
        return {'val_loss':loss,'test_users': test_users.detach(),'test_items':test_items.detach(),
                'test_pred':test_pred.detach(),'test_true':test_true.detach(),'prop_score':prop_score.detach() }


    def init_weight(self):
        pass



