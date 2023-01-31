import numpy as np
import scipy
from scipy.integrate import odeint
import scipy.linalg as linalg
import matplotlib.pyplot as plt

def lorenz96_n40(t,x):
    """Lorenz 96 model."""
    # Compute state derivatives
    # d = np.zeros(n)
    # # First the 3 edge cases: i=1,2,N
    # d[0] = (x[1] - x[n-2]) * x[n-1] - x[0]
    # d[1] = (x[2] - x[n-1]) * x[0] - x[1]
    # d[n-1] = (x[0] - x[n-3]) * x[n-2] - x[n-1]
    # # Then the general case
    # for i in range(2, n-1):
    #     d[i] = (x[i+1] - x[i-2]) * x[i-1] - x[i]
    # # Add the forcing term
    # d = d + F
    n=40
    F=8

    d = np.zeros(n)
    # First the 3 edge cases: i=1,2,N
    d[0] = -x[n-1]*(x[n-2]-x[1])-x[0]+F
    d[1] = -x[0]*(x[n-1]-x[2])-x[1]+F

    # Then the general case
    for i in range(2, n-1):
        d[i] = -x[i-1]*(x[i-2]-x[i+1])-x[i]+F
    # Add the forcing term
    d[n-1] = -x[n-2]*(x[n-3]-x[0])-x[n-1]+F


    # Return the state derivatives
    return d

def propagate_model(x,t):
 #   sol_= odeint(lorenz96, x, t)
    sol_=scipy.integrate.solve_ivp(lorenz96_n40,t,x)
    return sol_.y[:,-1]

def forecast_ensemble(EN,t,N):
    for i in range(0,N):
        EN[:,i]=propagate_model(EN[:,i],t)
    return EN

def get_subdom(i,r,n):
    c = np.arange(i - r,i + r+1)
    c[np.argwhere(c < 1)] = c[np.argwhere(c < 1)] + n
    c[np.argwhere(c > n)] = c[np.argwhere(c > n)] - n
    return c


def B_MC_r(X,r,ts,n):
    T = np.zeros((n, n));

    #W = [];

    for i in range(0,n):
        T[i, i] = 1;
        s = get_subdom(i, r, n);
       # print("s:",s.shape)
        s = s[np.where(s < i)];
       # print("sslice:",s.shape)

        if (s.size>0):

      #      print("X:",X.shape)
      #      print("Xslice:", X[s,:].shape)
            Z = X[s,:]
            y = X[i,:].T

            nl = Z.shape[0];
            sl = Z.shape[1];
            U, S, V = np.linalg.svd(Z);
            V = V.T
     #       print("Z:", Z.shape)
     #       print("U:",U.shape)
     #       print("S:",S.shape)
     #       print("V",V.shape)
            if (nl == 1):
                maxS = 1
            else:
                maxS=max(S)
                #maxS = max(S.diagonal());

            b = np.zeros((nl, 1));
            for v in range (0,min(nl, sl)):
                #if (S[v, v] / maxS >= ts):
                if (S[v] / maxS >= ts):
                    atemp=V[:, v].T
                 #   print((V).shape)
                 #   print((V[:,2]).shape)
                 #   print("atemp:",atemp.shape)
                 #   print(y.shape)
                    #print((V[:, v].T).shape)
                    #print(y.shape)
                    #print((np.expand_dims(y, axis=0)).shape)
                    weight = (np.matmul(V[:, v].T,y))/S[v]
                    #print("weight:",weight)
                    #b = b + np.matmul(U[:, v],weight)
                    #print("b_bef:",b.shape)
                    update_term=U[:, v]* weight
                    update_term= np.expand_dims(update_term, 1)
                    b = b + update_term
                    #print("b_upd:",b.shape)
                    #print("update_term_:",(update_term).shape)
     #               W[i, v] = weight;
                else:
                    break
            #T[i, s] = -b;
                #print("s:",s.shape)
                #print("b:",b.shape)
                #print("T_is:",T[i,s].shape)


            T[i,s]=- (np.squeeze(b))
    D_temp=np.matmul(T , X)
    #print("D_temp_mul:", D_temp.shape)
    D_temp=np.var(D_temp.T,axis=0,ddof=1)
    #print("D_temp:",D_temp.shape)
    D = np.diag(D_temp.T)
    #BE = np.matmul(T.T,(D\T));
    BE = np.matmul(T.T, linalg.solve(D,T));
    #sqrtB = linalg.inv(  np.matmul( T.T,np.sqrt(linalg(D)) )  )
    return BE

def analysis_ensemble_mod_Cholesky(EN,mb,N,m,H,y,r,erro,Sb):
    n=EN.shape[0]
    Binv = B_MC_r(Sb, r, 0.25,n);
    #Ys = np.matmul(y , np.ones(1, N)) + erro * np.random.randn((m, N))
    Ys = np.repeat(np.expand_dims(y,axis=1), N, axis=1) + erro * np.random.randn(m, N)

    W = Binv;
    Z = np.matmul(Binv , EN);
    for i in range (0,m):
        j = H[i];
        W[j, j] = W[j, j] + 1 / erro ** 2;
        Z[j,:] = Z[j,:] + (1 / erro ** 2) * Ys[i,:];


    XA = linalg.solve(W,Z)
    return XA


def analysis_ensemble_nino(XB,N,m,H,y,erro,Sb):

    Ys = np.repeat(np.expand_dims(y,axis=1), N, axis=1) + erro * np.random.randn(m, N)
    S = np.sqrt(1 / (N - 1)) * Sb

    D = (Ys - XB[H,:]);
    V = S[H,:]

    Z = (1 / erro ** 2) * D;
    U = (1 / erro ** 2) * V;


    for k in range(N):
        h = linalg.solve(1 + np.matmul(np.expand_dims(V[:, k],axis=0) , U[:,k]) ,U[:,k])
        Z = Z -    np.matmul(np.expand_dims(h,axis=1) , np.matmul( np.expand_dims(V[:, k],axis=0),Z))
        for i in range (k+1,N):
            U[:, i] = U[:, i]-  np.matmul( np.expand_dims(h,axis=1)  , np.matmul(np.expand_dims(V[:, k],axis=0),U[:,i]))

    XA = XB + np.matmul(S , np.matmul(V.T,Z));
    return XA

def generate_init_esenmble(errb_,n_,T_,N_):
    xt_ = np.random.randn(n_);
    xt_ = propagate_model(xt_, T_);
    xb0_ = xt_ + errb_ * np.random.randn(n_);

    print("first dif:", linalg.norm(xt_ - xb0_))

    xt_ = propagate_model(xt_, T_);
    xb0_ = propagate_model(xb0_, T_);
    print("second dif:", linalg.norm(xt_ - xb0_))
    # Initialensemble
    XB_0 = np.zeros((n_, N_))
    for i in range(0, N_):
        XB_0[:, i] = xb0_ + errb_ * np.random.randn(n_);
        XB_0[:, i] = propagate_model(XB_0[:, i], T_);

    xt_0 = propagate_model(xt_, T_)
    return xt_0,XB_0

