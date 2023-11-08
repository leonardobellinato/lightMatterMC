# New Monte-Carlo suite of codes #
# for simulation of atom dynamics inside # 
# optical tweezer + light-matter interaction # 

# Author: Daniel S. Grun #
# Innsbruck, 2023 # 

from new_MC_functions import *
from photon_functions import *
import sys
import numpy as np

# arg_input = int(sys.argv[1])
# n_jobs=arg_input
n_jobs = 4
Gamma583 = 2*np.pi * 180e3 # Gamma of the 583 nm transition (in Hz)
Gamma841 = 2*np.pi * 8e3 # Gamma of the 841 nm transition (in Hz)
Gamma626 = 2*np.pi * 135e3

lambd583 = 583e-9
lambd841 = 841e-9
lambd626 = 626e-9

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
    passedTime = 1
    waitTime = 0
    
    while currentTime < tf:

        sols = odeRK45_solver(trapEvol, initialCond, dt, P, w0)
        sols = np.array(sols)
        
        auxRatios = []
        auxMasks = []
        
        for i in range(len(absProj)):

            DopplerShift = getDopplerShift(sols, absProj[i], lambd)
            acStarkShift = getAcStarkShift(sols, P, w0, alpha_E=alpha_GS)
            Delta = 2*np.pi*delta + DopplerShift + acStarkShift

            # print(Delta/Gamma)

            scattProb = Rscatt(Gamma, s0, Delta) * dt
            auxNum = np.random.rand()
            auxRatios.append(scattProb/auxNum)
            auxMasks.append(scattProb>auxNum)
        
        auxRatio = max(auxRatios)
        index = auxRatios.index(auxRatio)
        # print(index)

        initialCond = sols
        
        currentTime += dt
                        
        if auxMasks[index]:
            sols[3:] = sols[3:] + recoilVel(absProj[index], lambd, case='absorption')
            initialCond = sols
            phScatt += 1
            waitTime = np.random.exponential(scale=1/Gamma)
            passedTime = 0
            while passedTime < waitTime:
                sols = odeRK45_solver(trapEvol, initialCond, dt, P, w0, alpha=0.6*alpha_GS)
                sols = np.array(sols)

                currentTime += dt
                passedTime += dt
                
                initialCond = sols
                
                solution.append(sols)
            
            # print("Atom velocity:", sols[3:])
            # print("Photon recoil:", recoilVel(absProj[index], lambd, case='emission'))
            sols[3:] = sols[3:] + recoilVel(absProj[index], lambd, case='emission')
            initialCond = sols
                        
            solution.append(sols)
            
            # print("Wait time", waitTime*1e6, "us")
        
            print("Current time", currentTime*1e6, "us")
        
        # print(phScatt)
        else:
            solution.append(sols)
        
                
    return solution


if __name__ == "__main__":
    P = 0.6e-3 # trap power, in W
    T = 30e-6 # temperature, in K
    w0 = 0.5e-6 # trap waist, in um

    s0 = 0.8 # near-resonant field, saturation parameter
    delta = -1*135e3 # detuning from resonance, in Hz
    lambd = lambd626
    Gamma = Gamma626
    
    t0 = 0
    tf = 30e-3
    dt = 1e-8
    titf = [t0, tf, dt]
    
    absZ = 0.4
    
    absProj = [[1.0, 0, absZ]]
    
    solution = createSimulation_loss(P,T,w0,titf,s0,lambd,Gamma,absProj,delta)
    solution = np.array(solution)


            
        
        