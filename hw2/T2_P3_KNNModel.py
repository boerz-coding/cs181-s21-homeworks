import numpy as np

# Please implement the predict() method of this class
# You can add additional private methods by beginning them with
# two underscores. It may look like the __dummyPrivateMethod below. You can feel
# free to change any of the class attributes, as long as you do not change any
# of the given function headers (they must take and return the same arguments).

class KNNModel:
    def __init__(self, k):
        self.X = None
        self.y = None
        self.K = k



    # TODO: Implement this method!
    def predict(self, X_pred):
        nfitdata=self.y.shape[0]
        npreddata=X_pred.shape[0]
        Y_pred=np.zeros(npreddata)
        for idpred in range(npreddata):
            dist_list=np.zeros(nfitdata)
            for idfit in range(nfitdata):
                dist_list[idfit]=((X_pred[idpred,0]-self.X[idfit,0])/3)**2+((X_pred[idpred,1]-self.X[idfit,1])/1)**2
            
            kidarray=np.argsort(dist_list)
            
            count_class=np.zeros(3)
            for k in range(self.K):
                count_class[self.y[kidarray[k]]]=count_class[self.y[kidarray[k]]]+1
                            
            Y_pred[idpred]=np.argmax(count_class)
                            
        
        return Y_pred

    # In KNN, "fitting" can be as simple as storing the data, so this has been written for you
    # If you'd like to add some preprocessing here without changing the inputs, feel free,
    # but it is completely optional.
    def fit(self, X, y):
        self.X = X
        self.y = y