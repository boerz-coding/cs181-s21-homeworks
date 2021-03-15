#!/usr/bin/env python
# coding: utf-8

# In[19]:


import numpy as np
import matplotlib.pyplot as plt
from scipy import stats 


# In[33]:


fig, axs = plt.subplots(5,3,figsize=(10,10),constrained_layout=True)

Data= np.array([3.3, 3.5, 3.1, 1.8, 3.0, 0.74, 2.5, 2.4, 1.6, 2.1, 2.4, 1.3, 1.7, 0.19])
length=14;
sumx=0
sigma2=1
tau2=5
for i in range(length):
    xid=i%3
    yid=i//3
    print(xid,yid)
    
    sumx+=Data[i]
    xbar=sumx/(i+1)
    sigman2=1/((i+1)/sigma2+1/tau2)
    mun=sigman2*sumx/sigma2
    
    x_values = np.arange(-5, 5, 0.1)
    

    Pfull=stats.norm(mun, (sigman2+sigma2)**0.5)
    PMLE=stats.norm(xbar, sigma2**0.5)
    PMAP=stats.norm(mun, sigma2**0.5)
    
    axs[yid,xid].title.set_text('ndata='+str(i+1))
    axs[yid,xid].plot(x_values,Pfull.pdf(x_values))
    axs[yid,xid].plot(x_values,PMLE.pdf(x_values))
    axs[yid,xid].plot(x_values,PMAP.pdf(x_values))
    


# In[48]:


sigma2=1
tau2=500000
n=14
sumx=0
sumx2=0
for i in range(length):
    sumx+=Data[i]
    sumx2+=Data[i]**2
    
pD=sigma2**0.5/(((2*np.pi*sigma2)**0.5)**n*(n*tau2+sigma2)**0.5)*np.exp(-sumx2/(2*sigma2))*np.exp(tau2*sumx**2/sigma2/(2*(n*tau2+sigma2)))
pD


# In[ ]:




