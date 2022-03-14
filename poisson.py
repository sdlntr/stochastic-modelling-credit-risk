import numpy as np
import matplotlib.pyplot as plt






def N(t, temps):
    r = 0
    for j in range(len(temps) - 1):
        if temps[j]<=t:
            r = r+1
    return r

def simY():
    uni = np.random.uniform(0,1)
    return -0.5 * (uni < 0.25) + 1 * (0.25 <= uni < 0.75) + 2 * (0.75 <= uni)

def Z(t, Y_i, temps):
    poisson = N(t, temps)
    print("poisson = ", poisson)
    z = 0
    
    for i in range(poisson):
        if len(Y_i) < poisson:
            Y_i.append(simY())
    z = sum(Y_i)
    return z



def S():
    Straj = np.ones(n+1) * A0
    for i in range(1,n+1):
        Straj[i] = Straj[i-1] + sigma * Straj[i-1] * np.sqrt(T/n) * np.random.normal(0,1)
    return Straj

def A(Ztraj, Straj, dates):
    return Straj + Ztraj - lbd * 0.875 * dates

def A_une(dates):
    intertemps = []
    
    temps = [0]
    test = 1


    while test == 1:
        intertemps.append(np.random.exponential(alpha))
        temps.append(temps[-1] + intertemps[-1])
        if temps[-1] > T:
            temps[-1] = T
            intertemps[-1] = temps[-1] - temps[-2]
            test = 0

    Ztraj = np.zeros(n+1)
    Ntraj = np.zeros(n+1)
    Straj = S()
    Y_i = []
    for i in range(n+1):
        Ztraj[i] = Z(dates[i], Y_i, temps)
        Ntraj[i] = N(dates[i], temps)
    Atraj = A(Ztraj, Straj, dates)
    return Atraj

alpha = 0.1
n = 1000
sigma  = 0.1
A0 = 100
K = 70
lbd = 5
T = 1
nt = 10
dates = np.linspace(0,T,n+1)

Acomb = np.ones((n+1,nt))
for j in range(nt):
    Acomb[:,j] = A_une(dates)


plt.plot(dates, Acomb)
plt.title("At")

plt.show()
