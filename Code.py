import numpy as np
MUS = (0.8255179,0.1654958,5.555786,0,12)
OYR = (0.6321328,0.1064985,6.743644,0,1)
FYR = (0.4395853,0.06066575,8.292693,0,60)
TYR = (0.3535297,0.04573934,9.250609,0,10)
# New Scheme
import numpy as np
import pickle
# New Scheme
class CIR:
    def __init__(self,sigma,alpha,b,R0,delta):
        # (P33: line 10)
        lam = lambda X_lag,alpha,delta,sigma:(4*alpha*np.exp(-alpha*delta))*X_lag/((sigma**2)*(1 - np.exp(-alpha*delta)))
        d = 4*b*alpha/(sigma**2)
        c = (sigma**2)*(1 - np.exp(-alpha*(delta)))/(4*alpha)
        temp_list = []
        if d<=1:
            # (P35: 3-6)
            Xd0 = np.random.gamma(d/2,2)
            self.Xd0 = Xd0
            # (P35: 3-7)
            U = np.random.uniform(0,1)
            temp = lam(R0,alpha,delta,sigma) + 2*np.log(U)
            temp_list.append(temp)
            if temp<=0:
                Y = 0
            else:
                Z = np.random.normal(0,1)
                Z_tilde = np.random.normal(0,1)
                Y = (Z + np.sqrt(temp))**2 + Z_tilde**2
            self.cir = c*(Xd0 + Y)
        else:
            # (P31: 3-4)
            Z = np.random.normal(0,1)
            self.cir = (Z + np.sqrt(lam(R0,alpha,delta,sigma) ))**2 + np.random.chisquare(d-1)
            self.cir = c*self.cir

# Possion Scheme
import numpy as np
# Possion Scheme
class CIR_Possion:
    def __init__(self,sigma,alpha,b,R0,delta):
        # (P33: line 10)
        lam = lambda X_lag,alpha,delta,sigma:(4*alpha*np.exp(-alpha*delta))*X_lag/((sigma**2)*(1 - np.exp(-alpha*delta)))
        d = 4*b*alpha/(sigma**2)
        c = (sigma**2)*(1 - np.exp(-alpha*(delta)))/(4*alpha)
        if d<=1:
        # (P31: 3-5)
            N = np.random.poisson(lam(R0,alpha,delta,sigma)/2)
            self.cir = np.random.chisquare(d + 2*N)
            self.cir = c*self.cir
        else:
            # (P31: 3-4)
            Z = np.random.normal(0,1)
            self.cir = (Z + np.sqrt(lam(R0,alpha,delta,sigma)))**2 + np.random.chisquare(d-1)
            self.cir = c*self.cir
# Euler-Maruyama Scheme
def CIR_Euler1(sigma,alpha,b,R0,T,N):
    delta = T/N
    X_list = [R0]
    X = R0
    for i in range(N):
        if X<0:
            X = 0
        X = X + alpha*(b - X)*delta + sigma*np.sqrt(X)*np.random.normal(0,np.sqrt(delta))
        X_list.append(X)
    import matplotlib.pyplot as plt
    plt.plot(X_list)
    return X
# Euler-Maruyama Scheme
def CIR_Euler2(sigma,alpha,b,R0,T,N):
    delta = T/N
    X_list = [R0]
    X = R0
    for i in range(N):
        X = X + alpha*(b - X)*delta + sigma*np.sqrt(X)*np.random.normal(0,np.sqrt(delta))
        X = np.absolute(X)
        X_list.append(X)
    import matplotlib.pyplot as plt
    plt.plot(X_list)
    return X
class SV_Euler1:
    def __init__(self,sigma,kappa,rho,T,V0,X0,theta,delta,r):
        c = (sigma**2)*(1 - np.exp(-kappa*delta))/(4*kappa)
        d = 4*kappa*theta/(sigma**2)
        lam = 4*kappa*np.exp(-kappa*delta)*V0
        lam = lam/((sigma**2)*(1 - np.exp(-kappa*delta)))
        # Condition 2kappa*theta>sigma^2 , V0 >0
        N = int(T/delta)
        S = X0
        V = V0
        for i in range(N):
            if V<0:
                V = 0
            # (P49: naive Euler)
            Z1 = np.random.normal(0,1)
            Z2 = np.random.normal(0,1)
            lnS = np.log(S) + (r - 0.5*V)*delta + np.sqrt(V)*(rho*Z1 + np.sqrt(1 - rho**2))*np.sqrt(delta)
            S = np.exp(lnS)
            V = V + kappa*(theta - V)*delta + sigma*np.sqrt(V)*Z1*np.sqrt(delta)
        self.V = V
        self.S = S
class SV_Euler2:
    def __init__(self,sigma,kappa,rho,T,V0,X0,theta,delta,r):
        c = (sigma**2)*(1 - np.exp(-kappa*delta))/(4*kappa)
        d = 4*kappa*theta/(sigma**2)
        lam = 4*kappa*np.exp(-kappa*delta)*V0
        lam = lam/((sigma**2)*(1 - np.exp(-kappa*delta)))
        # Condition 2kappa*theta>sigma^2 , V0 >0
        N = int(T/delta)
        S = X0
        V = V0
        for i in range(N):
            if V<0:
                V = 0
            # (P49: naive Euler)
            Z1 = np.random.normal(0,1)
            Z2 = np.random.normal(0,1)
            V1 = V
            V2 = V1 + kappa*theta*delta + sigma*np.sqrt(V1)*Z1*np.sqrt(delta) + 0.25*(sigma**2)*delta*(Z1**2 - 1)
            V2 = V2/(1 + kappa*delta)
            lnS = np.log(S) + (r - 0.25*(V2)+V1)*delta + rho*np.sqrt(V1)*Z1*np.sqrt(delta) +\
            0.5*(np.sqrt(V2)+np.sqrt(V1))*np.sqrt(1 - rho**2)*Z2*np.sqrt(delta) + 0.25*sigma*rho*delta*(Z1**2 - 1)
            S = np.exp(lnS)
            V = V2
        self.V = V
        self.S = S
class SV_Possion:
    def __init__(self,sigma,kappa,rho,T,V0,X0,theta,delta,r):
        V1 = V0
        V2 = V1 + kappa*(theta - V1)*delta + sigma*np.random.normal(0,np.sqrt(V1*delta))
        Vdu = V1*delta # P51: Euler-like setting
        Vdw = np.random.normal(0,np.sqrt(Vdu))
        lnS = np.log(X0) + r*delta + rho*(V2 - V1 - kappa*theta*delta) + \
        (kappa*rho/sigma - 1/2)*Vdu + np.sqrt(1 - rho**2)*Vdw
        S = np.exp(lnS)
        self.S = S
        self.V = V2
class SV:
    def __init__(self,sigma,kappa,rho,T,V0,X0,theta,delta,r):
        gamma1  = 1
        gamma2 = 0
        V1 = V0
        V2 = V1 + kappa*(theta - V1)*delta + sigma*np.random.normal(0,np.sqrt(V1*delta))
        K0 = (r - rho*kappa*theta/sigma)*delta
        K1 = gamma1*delta*(kappa*rho/sigma - 1/2) - rho/sigma
        K2 = gamma2*delta*(kappa*rho/sigma - 1/2) + rho/sigma
        K3 = gamma1*delta*(1 - rho**2)
        K4 = gamma2*delta*(1 - rho**2)
        Z = np.random.normal(0,1)
        lnS = np.log(X0) + K0 + K1*V1 + K2*V2 + np.sqrt(K3*V1 + K4*V2)*Z
        S = np.exp(lnS)
        self.S = S
Case1 = [2,0.1,0.4,0.3]
Case2 = [1.2,0.2,0.2,0.1]
Case3 = [1,0.4,0.1,0.05]
