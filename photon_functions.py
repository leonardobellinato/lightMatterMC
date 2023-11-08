# Set of functions to include atom-light interaction during the MC simulation #

import numpy as np
import random


def Rscatt(Gamma, s, Delta):
    R = Gamma/2 * (s/(1+s+4*(Delta/Gamma)**2))
    return R

def getDopplerShift(rvVector, absProj=[1,0,0], lambd=583e-9):
    absProj = np.array(absProj)
    absProj = absProj/np.sqrt(sum(absProj**2))

    v = rvVector[3:]
    k = 2*np.pi / lambd * absProj
    DopplerShift = - sum(k*v)
    return DopplerShift

