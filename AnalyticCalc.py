import numpy as np


Z = 2*(np.exp(8) + np.exp(-8)) + 12.
ExpE = 8*(np.exp(-8) - 2*np.exp(8))/Z
expM = 4*(2*np.exp(8) + 1)/Z
Cv = 8*8*(2*np.exp(8) + np.exp(-8))/Z - ExpE**2

chi = 8*(4*np.exp(8) + 1)/Z - expM**2

print "Energy =", ExpE/4
print "Magnetization", expM/4
print "Specific Heat", Cv
print "Susceptibility", chi