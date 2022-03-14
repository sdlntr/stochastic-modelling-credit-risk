import numpy as np
import matplotlib.pyplot as plt
import numpy.random as random
from math import erf
from random import randint

mu = 0.02
K = 85
r = 0.005
A0 = 100
T = 1
n = 1000 #nbr de divisions
N = 2#nbr trajectoires
pas = T/n
rpas = np.sqrt(pas)
sig = 0.1
sigma = lambda t : sig * (2 - np.exp(-t))



colors = []

for i in range(N):
    colors.append('#%06X' % randint(0, 0xFFFFFF))

def phi(x):
    #'Cumulative distribution function for the standard normal distribution'
    return (1.0 + erf(x / np.sqrt(2.0))) / 2.0

def rho(t):
    return max(2 * sig * np.sqrt( T - t + np.exp(-T) - np.exp(-t) - (1/8) * (np.exp(-2*T) - np.exp(-2*t))), 0.001)

def d1(i, Aj):
    return (1/rho(i*pas)) * (np.log(Aj[i]/K) + r*(T-i*pas) + (rho(i*pas)**2)/2)

def d2(i, Aj):
    return d1(i, Aj) - rho(i*pas)

def euler():
    A = np.ones((n+1,N)) * A0
    D = np.ones((n+1,N)) * ( A[0,0] * phi(-d1(0, A[:,0])) + K*np.exp(-r*(T))*phi(d2(0, A[:,0])) )
    E = np.ones((n+1,N)) * ( A[0,0] * phi(d1(0, A[:,0])) - K*np.exp(-r*(T))*phi(d2(0, A[:,0])) )
    spread = np.zeros((n+1,N))
    D0 = np.zeros(n+1)
    for i in range(n+1):
        D0[i] = K*np.exp(-r*(T - i*pas))
    for j in range(N):
        for i in range(1,n+1):
            A[i,j] = A[i-1,j] + sigma((i-1)*pas) * A[i-1,j] * rpas * np.random.normal(0,1) + mu * A[i-1,j] * pas
            D[i,j] = A[i,j] * phi(-d1(i, A[:,j])) + K*np.exp(-r*(T-i*pas))*phi(d2(i, A[:,j]))
            E[i,j] = A[i,j] - D[i,j]
            spread[i-1,j] = -1/(T-(i-1)*pas) * np.log(D[i-1,j]/D0[i-1])
    return [A, D, D0, E, spread]


output = euler()

A = output[0]

D = output[1]

D0 = output[2]

E = output[3]

sp = output[4]

temp = np.linspace(0,T,n+1)

for j in range(N):
    if A[n,j] < K:
        print("D final : ", D[n,j])
        print("A final : ", A[n,j])


plt.subplot(221)
plt.gca().set_prop_cycle(color=colors)
for j in range(N):
    plt.plot(temp, A[:,j])
plt.plot(temp, [K for i in range(n+1)], '--')
plt.title("Actif")

plt.subplot(222)
plt.gca().set_prop_cycle(color=colors)
for j in range(N):
    plt.plot(temp, E[:,j])
plt.title("Equity")


plt.subplot(223)
plt.gca().set_prop_cycle(None)
plt.gca().set_prop_cycle(color=colors)
for j in range(N):
    plt.plot(temp, D[:,j])
plt.plot(temp, D0, '--')
plt.title("Dette")

plt.subplot(224)
plt.gca().set_prop_cycle(None)
plt.gca().set_prop_cycle(color=colors)
plt.plot(temp, sp)
plt.title("Spread")


plt.show()

