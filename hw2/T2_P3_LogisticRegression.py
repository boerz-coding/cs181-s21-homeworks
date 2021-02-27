import numpy as np
import matplotlib.pyplot as plt



# Please implement the fit(), predict(), and visualize_loss() methods of this
# class. You can add additional private methods by beginning them with two
# underscores. It may look like the __dummyPrivateMethod below. You can feel
# free to change any of the class attributes, as long as you do not change any
# of the given function headers (they must take and return the same arguments).

class LogisticRegression:
    def __init__(self, eta, lam):
        self.eta = eta
        self.lam = lam

    # Just to show how to make 'private' methods
    def __softmax(self,z):
        expz=np.exp(z);
        #print("expzshape",expz.shape)
        #print("sumexpzshape",np.sum(expz))
        for i in range(expz.shape[0]):
            expz[i,:]=expz[i,:]/np.sum(expz[i,:])
        return expz

    # TODO: Implement this method!
    def fit(self, X, y):
        
        self.W = np.random.rand(X.shape[1]+1, 3)
        #self.W = np.random.rand(X.shape[1], 3)
        nrun=200000
        
        #print(X.shape[0])
        Xreshape=np.hstack((X,np.ones([X.shape[0],1])))
        #Xreshape=X #np.hstack((X,np.ones([X.shape[0],1])))
        #print(Xreshape.shape)
        Yreshape=np.zeros([X.shape[0],3])
        
        for rowy in range(X.shape[0]):
            Yreshape[rowy,y[rowy]]=1
            
        
        self.nloglikelihood=np.zeros(nrun)
        for irun in range(nrun):
            yhat=self.__predict(X)
            gradient=np.dot(Xreshape.T,yhat-Yreshape)/y.shape[0]+2*self.lam*self.W
            self.nrun=irun
            self.nloglikelihood[irun]=-np.sum(np.multiply(Yreshape,np.log(yhat)))
            self.W-=self.eta*gradient
            if irun>0:
                if(np.abs(self.nloglikelihood[irun]-self.nloglikelihood[irun-1])<1e-5):
                    break
            
            
        #return niteration,negative_likelihood
 
 

    # TODO: Implement this method!
    def __predict(self, X_pred):
        X=np.hstack((X_pred,np.ones([X_pred.shape[0],1])))
        #X=X_pred
        return self.__softmax(np.dot(X,self.W))
    # TODO: Implement this method!
    def predict(self, X_pred):
        yhat=self.__predict(X_pred)
        y=np.zeros([yhat.shape[0]])
        #print("yshape:",y.shape,"yhatshape",yhat.shape)
        for idy in range(yhat.shape[0]):
            yidy=np.argmax(yhat[idy,:])
            y[idy]=yidy
        return y

    # TODO: Implement this method!
    def visualize_loss(self, output_file, show_charts=False):
        plt.figure()
        title=output_file+'_eta='+str(self.eta)+'_lam='+str(self.lam)
        plt.title(title)
        plt.plot(range(0,self.nrun+1),self.nloglikelihood[0:self.nrun+1])
        plt.xlabel('Number of iterations')
        plt.ylabel('Negative Log-Likelihood Loss')
        plt.savefig(title + '.png')
        return self.nloglikelihood
