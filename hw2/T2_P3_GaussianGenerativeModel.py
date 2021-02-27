import numpy as np
from scipy.stats import multivariate_normal as mvn  # you may find this useful


# Please implement the fit(), predict(), and negative_log_likelihood() methods
# of this class. You can add additional private methods by beginning them with
# two underscores. It may look like the __dummyPrivateMethod below. You can feel
# free to change any of the class attributes, as long as you do not change any
# of the given function headers (they must take and return the same arguments).

class GaussianGenerativeModel:
    def __init__(self, is_shared_covariance=False):
        self.is_shared_covariance = is_shared_covariance

    # Just to show how to make 'private' methods
    def __dummyPrivateMethod(self, input):
        return None

    # TODO: Implement this method!
    def fit(self, X, y):
        nclass=3;
        self.prior=np.zeros(nclass)
        ny=y.shape[0]
        #estimate class prior pi
        for iy in range(ny):
            self.prior[y[iy]]+=1
        self.prior=self.prior/ny
        #print("priors:",self.prior)
       
        
         
        
        ##estimate mu's
        self.mu=np.zeros([3,X.shape[1]])
        
        for iy in range(ny):
            self.mu[y[iy],:]=self.mu[y[iy],:]+X[iy,:]
        for iclass in range(nclass):
            self.mu[iclass,:]=self.mu[iclass,:]/(ny*self.prior[iclass])
            
        ##estimate sigma
        if self.is_shared_covariance:
            self.sigma=np.zeros([X.shape[1],X.shape[1]])
            for iy in range(ny):
                deltaXmu=X[iy,:]-self.mu[y[iy],:]
                deltaXmu=deltaXmu.reshape(deltaXmu.shape[0],1)
                #print("deltaXmu:",deltaXmu,"kjjk")
                deltasigma=np.matmul(deltaXmu,deltaXmu.T)
                #print("deltasigma:",deltasigma)
                #print("deltasigmashape:",deltasigma.shape)
                              
                self.sigma+=deltasigma
            self.sigma=self.sigma/ny
        else:
            self.sigma=np.zeros([X.shape[1],X.shape[1],3])
            for iy in range(ny):
                deltaXmu=X[iy,:]-self.mu[y[iy],:]
                deltaXmu=deltaXmu.reshape(deltaXmu.shape[0],1)
                #print("deltaXmu:",deltaXmu,"kjjk")
                deltasigma=np.matmul(deltaXmu,deltaXmu.T)
                #print("deltasigma:",deltasigma)
                #print("deltasigmashape:",deltasigma.shape)
                self.sigma[:,:,y[iy]]+=deltasigma
            for iclass in range(nclass):
                self.sigma[:,:,iclass]=self.sigma[:,:,iclass]/(ny*self.prior[iclass])
            
        
        
        return

    # TODO: Implement this method!
    def predict(self, X_pred):
        # The code in this method should be removed and replaced! We included it
        # just so that the distribution code is runnable and produces a
        # (currently meaningless) visualization.
        Pclass=np.zeros([X_pred.shape[0],3])
        
        if self.is_shared_covariance:
            for iclass in range(3):
                #print("mean:",self.mu[iclass,:],"sigma",self.sigma)
                #rv=mvn(X_pred,mean=self.mu[iclass,:],cov=self.sigma)
                Px=mvn.pdf(X_pred,mean=self.mu[iclass,:],cov=self.sigma)
                #print("Xpredshape:",X_pred.shape)
                #print("Px:",Px,"Pxshape:",Px.shape)
                pclassi=self.prior[iclass]*Px
                #print("pclassi:",pclassi)
                #print("pclassishape:",pclassi.shape)
                Pclass[:,iclass]=pclassi
                
      
            return np.argmax(Pclass,axis=1)
               
                
        else:
            for iclass in range(3):
                #print("mean:",self.mu[iclass,:],"sigma",self.sigma)
                #rv=mvn(X_pred,mean=self.mu[iclass,:],cov=self.sigma)
                Px=mvn.pdf(X_pred,mean=self.mu[iclass,:],cov=self.sigma[:,:,iclass])
                #print("Xpredshape:",X_pred.shape)
                #print("Px:",Px,"Pxshape:",Px.shape)
                pclassi=self.prior[iclass]*Px
                #print("pclassi:",pclassi)
                #print("pclassishape:",pclassi.shape)
                Pclass[:,iclass]=pclassi
                
      
            return np.argmax(Pclass,axis=1)

    # TODO: Implement this method!
    def negative_log_likelihood(self, X, y):
        L=0
        ny=y.shape[0]

       
        #print("Pxshape:",Px)
        for iy in range(ny):
            if self.is_shared_covariance:
                Px=mvn.pdf(X[iy,:],mean=self.mu[y[iy],:],cov=self.sigma)
            else:
                Px=mvn.pdf(X[iy,:],mean=self.mu[y[iy],:],cov=self.sigma[:,:,y[iy]])
            L-=np.log(self.prior[y[iy]]*Px)
            
            
        return L
        
        
