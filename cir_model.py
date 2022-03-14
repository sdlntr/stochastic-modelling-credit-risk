import numpy as np
import matplotlib.pyplot as plt

###########################################
#----------------COMPUTING----------------#
###########################################

T = 1
n = 1000
N = 5
r0 = 0.06

#Cir-Cox-Ingersoll model parameters
gamma = 0.01
theta = 0.05
sigma = 0.15
sig0 = 0.1

#stochastic modelling for deviation
sigma_t = lambda t : sigma * (2 + np.sin(2 * np.pi * t)) 
pas = T / n
K = 120
A0_tilde = 100
A0 = 100
r_pas = np.sqrt(pas)

#Euler scheme function w/ first-value approx. scaling
def computing(T,n,N,r0,gamma,theta,sig0,A0):

    r = np.ones((n+1,N)) * r0
    A = np.ones((n+1,N)) * A0
    S0 = np.ones((n+1,N))
    A_tilde = np.ones((n+1,N)) * A0_tilde
    Aprime = np.ones((n+1,N)) * A0
    int = np.zeros((n+1,N))
    diff = np.zeros((n+1,N))


    for i in range(1,n+1):

        G = np.random.normal(0,1,N)
        A_tilde[i] = A_tilde[i-1] + A_tilde[i-1] * sigma_t((i-1)*T/n) * r_pas * G
        r[i] = r[i-1] + gamma * (theta-r[i-1]) * pas + sig0 * np.sqrt(abs(r[i-1])) * r_pas * np.random.normal(0,1,N)
        int[i] = int[i-1] + r[i-1] * pas
        S0[i] = [np.exp(int[i,j]) for j in range(0,N)]
        A[i] = [A_tilde[i,j] * S0[i,j] for j in range(0,N)]  
        Aprime[i] = Aprime[i-1] + Aprime[i-1] * sigma_t((i-1)*T/n) * r_pas * G + r[i-1] * pas * Aprime[i-1]
        diff[i] = A[i] - Aprime[i]
    return [A_tilde, A, r, S0, Aprime, diff]

#Monte-Carlo method estimation of initial asset price of both risk-free and risky processes
def montecarlo(T,n,N,r0,gamma,theta,sig0,A0,K):

    D0 = 0
    D00 = 0
    nbr = 1000
    processes = computing(T,n,N,r0,gamma,theta,sig0,A0)
    D0 = np.mean(1/processes[3][n] * [min(processes[1][n,j], K) for j in range(0,N)])
    D00 = np.mean(1/processes[3][n] * K)

    return [D0, D00, processes]


D = montecarlo(T,n,N,r0,gamma,theta,sig0,A0,K)

A_tilde = D[2][0]
A = D[2][1]
r =D[2][2]
Aprime = D[2][4]
diff = D[2][5]
#Estimation of the spread, i.e. the additional interest rate gained as a risk-taking bonus,
#lowering the initial price for the risky bond after normalization

spread = -1/T * np.log(D[0]/D[1])

print("D (not risky) = ", D[1])
print("D (risky) = ", D[0])
print("spread = ", spread)

##########################################
#----------------PLOTTING----------------#
##########################################

temp=np.linspace(0,T,n+1)

plt.subplot(311)

#Interest rate w/ Euler Scheme 
plt.plot(temp,r)
plt.title("Rate")

plt.subplot(312)

#Asset price of the bond following the stochastic interest rate
plt.plot(temp,A)
plt.title("Asset price (non-updated)")

plt.subplot(313)

#Updated asset price taking into consideration the risk-taking bonus
plt.plot(temp, Aprime)
plt.title("Asset price other method (non-updated)")

plt.show()