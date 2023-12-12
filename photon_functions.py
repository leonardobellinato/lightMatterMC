# Set of functions to include atom-light interaction during the MC simulation #
# Includes definition of atom-light scattering rate, Doppler shift etc. #
# Also includes inverse-transform sampling which is not necessary anymore #
# Daniel S. Grun, Innsbruck 2023 #

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
    #print(rvVector)
    DopplerShift = - sum(k*v)
    return DopplerShift

def expDecay(t, rate):
    return np.exp(-rate*t)

def randGen(function, timeScale=1, xLen=int(1e3)):
    xs = np.linspace(0,timeScale,num=xLen)
    ys = function(xs)
    cdf = np.cumsum(ys) # cumulative distribution function, cdf
    cdf = cdf/cdf.max()
    # generator = interpolate.interp1d(cdf, xs)
    generator = lambda x: np.interp(x, cdf, xs)
    return [generator, cdf.min(), cdf.max()]


def drawFromGen(generator, cdf_min, cdf_max):
    num = random.uniform(cdf_min, cdf_max)
    occ = generator(num)
    return occ
