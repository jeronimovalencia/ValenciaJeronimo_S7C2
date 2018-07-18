import numpy as np
import matplotlib.pyplot as plt

V_0 = 10

datos = np.genfromtxt("CircuitoRC.txt")
time = datos[:,0]
data = datos[:,1]

def q(t,R,C):
	return (V_0*C)*(1-np.exp(-1.0*t/(R*C)))

def chi_squared(datos,modelo):
	return np.sum((datos-modelo)**2)

def likelihood(datos,modelo):
	return np.exp(-(1.0/2.0)*chi_squared(datos,modelo))

R_good = np.array([20.0*np.random.rand(1)]) 
C_good = np.array([20.0*np.random.rand(1)])
chi_good = np.array([chi_squared(data,q(time,R_good[0],C_good[0]))])

sigma = 0.1

N = 25000

for i in range(N):
	chi_good_data = chi_squared(data,q(time,R_good[i],C_good[i]))
	
	R_guess = np.random.normal(R_good[i],sigma,1)
	C_guess = np.random.normal(C_good[i],sigma,1)
	
	chi_guess = chi_squared(data,q(time,R_guess,C_guess))
	
	alpha = chi_good_data / chi_guess
	
	if(alpha > 1.0):
		R_good = np.append(R_good,R_guess)
		C_good = np.append(C_good,C_guess)
		chi_good = np.append(chi_good,chi_guess)
	
	else:
		beta = np.random.rand(1)
		if(beta < alpha):
			R_good = np.append(R_good,R_guess)
			C_good = np.append(C_good,C_guess)
			chi_good = np.append(chi_good,chi_guess)
		else:
			R_good = np.append(R_good,R_good[i])
			C_good = np.append(C_good,C_good[i])
			chi_good = np.append(chi_good,chi_good_data)



R_best = R_good[np.argmin(chi_good)]
C_best = C_good[np.argmin(chi_good)]

plt.figure()
plt.title("Carga del circuito RC")
plt.scatter(time,data,s=5,color="cyan", label="Datos")
plt.plot(time,q(time,R_best,C_best),color="black",label="R="+str(R_best)+"\nC="+str(C_best))
plt.legend(loc="lower right")
plt.savefig("CargaRC.pdf")





















