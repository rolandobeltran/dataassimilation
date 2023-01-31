import numpy as np
import scipy
from scipy.integrate import odeint
import scipy.linalg as linalg
import matplotlib.pyplot as plt

import data_assimilation.da_base as da_base

n=40
N=100
errb=0.05
erro=0.01
M=15
r=2

print(np.arange(1,3))

dm = 2;
H = np.arange(0,n,dm)
m = len(H)

F = 8  # Forcing


T0 =(0,10)#np.arange(0.0, 10.0, 0.01) #Ensemble initialization
TF =(0,1)#np.arange(0.0, 1.0, 0.001) ; #Forecast step
TJ =(0,1) #np.arange(0.0, 1.0, 0.01) ; #Plot ensemble trajectories


EA=np.zeros(M)
EB=np.zeros(M)


np.random.seed(10)
xt,XB=da_base.generate_init_esenmble(errb,n,T0,N)
print("third:",linalg.norm(xt-np.mean(XB,1)))
##################
XB_no_da=XB
xmb_no_da=np.mean(XB_no_da, 1,keepdims = True)
#################
for k in range(0,M):
    print('ASSIMILATION STEP:',k)
    xt = da_base.propagate_model(xt, TF);

    XB = da_base.forecast_ensemble(XB, TF, N);
    xmb = np.mean(XB, 1,keepdims = True);

    ###############
    XB_no_da = da_base.forecast_ensemble(XB_no_da, TF, N);
    xmb_no_da = np.mean(XB_no_da, 1, keepdims=True)

    ########


    y = xt[H] + erro * np.random.randn(m);

    print(xmb.shape)
    print("Dif_step",linalg.norm(xmb[:,0]-xt))
    #exit()
   # Analysis ensemble

    ones_=np.ones((1,N))
    print(ones_.shape)

    #Sb = XB - np.matmul(xmb , ones_)
    Sb = XB - np.repeat(xmb, N,axis=1)
    print ("Sb:",Sb.shape)

    XA = da_base.analysis_ensemble_mod_Cholesky(XB,xmb,N,m,H,y,r,erro,Sb)

    xma = np.mean(XA,1,keepdims = True);

    EA[k] = linalg.norm(xma[:,0] - xt);
    EB[k] = linalg.norm(xmb[:,0] - xt);
    print("EA:",EA[k])
    print("EB:",EB[k])
    XB = XA;

plt.plot(EA,'bo-',label='Analysis EnKF_CH')
plt.plot(EB,'r*-',label='Background')
plt.legend()
plt.title('Assimilation performance')
plt.xlabel('Time')
plt.ylabel('RMSE')
plt.show()