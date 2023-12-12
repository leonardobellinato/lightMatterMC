# Monte-Carlo simulation of dynamics inside the Tweezer with #
# Photon absorption and Spontaneous emission in the presence of #
# A light field #
# Daniel S. Grun, Innsbruck 2023 #

from new_MC_functions import *
from photon_functions import *
import sys
import numpy as np

# arg_input = int(sys.argv[1])
# n_jobs=arg_input
n_jobs = 4
Gamma583 = 2*np.pi * 180e3 # Gamma of the 583 nm transition (in Hz)
Gamma841 = 2*np.pi * 8e3 # Gamma of the 841 nm transition (in Hz)
lambd583 = 583e-9
lambd841 = 841e-9

conversion = 0.1482e-24 * 1.113e-16 # conversion from a.u. to S.I.
alpha_GS = 430 * conversion # polarizability


def createSimulation_loss(P,T,w0,titf, s0=0, lambd=lambd583,
                  Gamma=Gamma583, absProj=[[1.0,0,0]], delta=0, n_samples=1):
    

    initialCond = generateInitialCond(P,T,w0,n_samples=1)
    initialCond = initialCond[0]
    # print(initialCond)
        
    ti, tf, dt = titf
    tInit = ti
    
    currentTime = 0
    phScatt = 0
    solution = []
    
    while currentTime < tf:

        sols = odeRK45_solver(trapEvol, initialCond, dt, P, w0)
        sols = np.array(sols)
        
        scattProbs = []
        
        for i in range(len(absProj)):

            DopplerShift = getDopplerShift(sols, absProj[i], lambd)
            acStarkShift = getAcStarkShift(sols, P, w0, alpha_E=alpha_GS)
            Delta = 2*np.pi*delta + DopplerShift + acStarkShift

            # print(Delta/Gamma)

            scattProbs.append(Rscatt(Gamma, s0, Delta) * dt)
            
        scattProb = max(scattProbs)
        index = scattProbs.index(scattProb)
        
        auxNum = np.random.rand()
        auxMask = scattProb > auxNum
        phScatt += auxMask
        # print(int(phScatt), scattProb)
        
        sols[3:] = sols[3:] + recoilVel(absProj[index], lambd)*auxMask

        initialCond = sols
        
        currentTime += dt
                
        phScatt += auxMask
        
        if (100*currentTime/tf) % 5 == 0:
            print(int(100*currentTime/tf), "%")
        
        # print(phScatt)
        
        solution.append(sols)
        
                
    return solution


if __name__ == "__main__":
    P = 3e-3 # trap power, in W
    T = 30e-6 # temperature, in K
    w0 = 1e-6 # trap waist, in um

    s0 = 0.8/np.sqrt(2.) # near-resonant field, saturation parameter
    delta = -1*180e3 # detuning from resonance, in Hz
    lambd = lambd583
    Gamma = Gamma583
    
    t0 = 0
    tf = 30e-3
    dt = 1e-8
    titf = [t0, tf, dt]
    
    absZ = 0.
    
    absProj = [[1.0, 0, absZ], [0, 1.0, absZ]]
    
    solution = np.array([createSimulation_loss(P,T,w0,titf,s0,lambd,Gamma,absProj,delta) 
                         for i in tqdm(range(10))])


            
        
        
