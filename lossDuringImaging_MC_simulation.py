# Monte-Carlo simulation of dynamics inside the Tweezer with #
# Photon absorption and Spontaneous emission in the presence of #
# A light field #
# Daniel S. Grun and Leonardo B. Giacomelli, Innsbruck 2023 #

# Notes:

    # Atom with highest probability of absorption is chosen for the excitation/sponatneous emission
    # For the moment we only consider C3, i.e. LJ-potential if one atom is excited. Otherwise both evolve separately 
    # AC Stark shift is not considered at the moment
           
'''Import packages'''
from new_MC_functions import *
from photon_functions import *
import sys
import numpy as np

'''Useful parameters and inputs'''
arg_input = 1 #int(sys.argv[1])
n_jobs=arg_input
# n_jobs = 4
Gamma583 = 2*np.pi * 180e3 # Gamma of the 583 nm transition (in Hz)
Gamma841 = 2*np.pi * 8e3 # Gamma of the 841 nm transition (in Hz)
Gamma626 = 2*np.pi * 135e3 # Gamma of the 626 nm transition (Dy) (in Hz)
lambd583 = 583e-9
lambd841 = 841e-9
lambd626 = 626e-9

conversion = 0.1482e-24 * 1.113e-16 # conversion from a.u. to S.I.
alpha_GS = 430 * conversion # polarizability

'''Functions'''
def checkLost(rvVector, P, w0, a,PotFlag):
    r1 = rvVector[:3]
    v1 = rvVector[3:6]
    r2 = rvVector[6:9]
    v2 = rvVector[9:12]
    
    if a == 0:
        potEnergy = trapPotential(r1, r2,P,w0,PotFlag)
        kinEnergy = kineticEnergy(v1)
        msg = potEnergy + kinEnergy
    else:
        potEnergy = trapPotential(r2, r1,P,w0,PotFlag)
        kinEnergy = kineticEnergy(v2)
        msg = potEnergy + kinEnergy
        
    return int(msg > 0)


def createSimulation_loss(P,T,w0,titf, s0=0, lambd=lambd583,
                  Gamma=Gamma583, absProj=[[1.0,0,0]], delta=0):
    
    #Adding initialization for both atoms
    initialCond1 = generateInitialCond(P,T,w0,n_samples=1)
    initialCond1 = initialCond1[0]
    initialCond2 = generateInitialCond(P,T,w0,n_samples=1)
    initialCond2 = initialCond2[0]
    # print(initialCond)
    initialCond = np.concatenate([initialCond1, initialCond2])
    
    # initialCond[3:6] = 1
    # initialCond[9:12] = 1
    
    #print(initialCond)
    ti, tf, dt = titf
    tInit = ti
    
    currentTime = 0
    phScatt = 0
    #solution = [[],[]]
    solution = []
    
##################################################################################################
  
    #print(initialCond)

    while currentTime < tf:
        
        #a = trapEvol(initialCond, P, w0)
        #print(a)
        PotFlag = 0
        #print(PotFlag)
        
        sols = odeRK45_solver(trapEvol, initialCond, dt, P, w0, currentTime, PotFlag)
        sols = np.array(sols[-1])
        
        #print(sols)
        
        auxRatios1 = []
        auxMasks1 = []
        auxRatios2 = []
        auxMasks2 = []
        
        initialCond = sols
        
        print(initialCond)
        
        solution.append(initialCond)
        
        for i in range(len(absProj)):
          
            solsw1 = sols[:6]
            solsw2 = sols[6:12]
            
            DopplerShift1 = getDopplerShift(solsw1, absProj[i], lambd)
            
           # acStarkShift1 = getAcStarkShift(solsw1, P, w0, alpha_E=alpha_GS)
            
            Delta1 = 2*np.pi*delta + DopplerShift1 #+ acStarkShift1
           
            DopplerShift2 = getDopplerShift(solsw2, absProj[i], lambd)
            
           # acStarkShift2 = getAcStarkShift(solsw2, P, w0, alpha_E=alpha_GS)
            
            Delta2 = 2*np.pi*delta + DopplerShift2 #+ acStarkShift2
            # print(Delta/Gamma)

            scattProb1 = Rscatt(Gamma, s0, Delta1) * dt
            scattProb2 = Rscatt(Gamma, s0, Delta2) * dt
            
            auxNum1 = np.random.rand()
            auxNum2 = np.random.rand()
            
            auxRatios1.append(scattProb1/auxNum1)
            auxMasks1.append(scattProb1>auxNum1)
            auxRatios2.append(scattProb2/auxNum2)
            auxMasks2.append(scattProb2>auxNum2)
            
        if max(auxRatios1) >= max(auxRatios2):
            solsw = sols[:6]
            auxRatios = auxRatios1
            a = 0
            auxMasks = auxMasks1
        else:
            solsw = sols[6:12]
            auxRatios = auxRatios2
            a = 1
            auxMasks = auxMasks2
            
        auxRatio = max(auxRatios)
        index = auxRatios.index(auxRatio)
          
        lost = checkLost(initialCond, P, w0, a,PotFlag)
        currentTime += dt
        
        #lost = checkLost(initialCond, P, w0)
##################################################################################################
        if auxMasks[index]:
            
            PotFlag = 1 
            #print(PotFlag)
            
            solsw[3:6] = solsw[3:6] + recoilVel(absProj[index], lambd, case='absorption')
                
            if a == 0:
                initialCond[:6] = solsw
            else:
                initialCond[6:12] = solsw
            solution.append(initialCond)
            #initialCond = sols
            phScatt += auxMasks[index]
            waitTime = np.random.exponential(scale=1/Gamma)
            passedTime = 0
            while passedTime < waitTime:
                sols = odeRK45_solver(trapEvol, initialCond, dt, P, w0, currentTime, PotFlag)      #trapEvol, initialCond, dt, P, w0)
                sols = np.array(sols[-1])

                currentTime += dt
                passedTime += dt
                
                initialCond = sols
                solution.append(initialCond)
            if a == 0:
                solsw = sols[:6]

            else:
                solsw = sols[6:12]

            solsw[3:] = solsw[3:] + recoilVel(absProj[index], lambd, case='emission')
            
            if a == 0:
                initialCond[:6] = solsw
            else:
                initialCond[6:12] = solsw
                
            solution.append(initialCond)

##################################################################################################      

        # print(phScatt)
        
        # if lost:
        #     lostTime = currentTime
        #     currentTime = tf+1
            
        #     # print("Atom lost after {} us of illumination".format(np.round(1e6*lostTime, 2)))
        #     solution.append(1-lost)
        #     solution.append(lostTime)
        #     solution.append(phScatt)
            
        # else:
        #     # print("currentTime = {} us".format(np.round(1e6*currentTime, decimals=2)))
        #     if currentTime >= tf:            
        #         solution.append(1-lost)
        #         solution.append(currentTime)    
        #         solution.append(phScatt)
        

            
    return solution
##################################################################################################

def runSimulation_loss(P,T,w0,titf, s0=0, lambd=lambd583, Gamma=Gamma583, absProj=[[1.0,0,0]], delta=0, n_samples=1):
    print(n_jobs)
    results = Parallel(n_jobs=n_jobs)(delayed(createSimulation_loss)(P,T,w0,titf,s0,lambd,Gamma,absProj,delta)
                                               for i in tqdm(range(n_samples)))
    return results
        

def survivalProb(P,T,w0,titf,s0=0,lambd=lambd583,Gamma=Gamma583,absProj=[[1.0,0,0]],delta=0,n_samples=1):
    results = runSimulation_loss(P,T,w0,titf,s0,lambd,Gamma,absProj,delta,n_samples)
    
    results = np.array(results)
    
    surv = results[:,0]
    phScatt = results[:,2]
    survProb = len(surv[surv==1])/len(surv)
    
   
    
    return [survProb, phScatt]

##################################################################################################

'''Run code'''

if __name__ == "__main__":
    
    Gamma = Gamma583
    lambd = lambd583
    
    P = 2.4e-3 # trap power, in W
    T = 20e-6 # temperature, in K
    w0 = 1.0e-6 # trap waist, in um
    s0 = 0.8 # near-resonant field, saturation parameter 
    dParam = 1
    delta = -dParam*180e3 # detuning from resonance, in Hz
    
    t0 = 0
    tf = 1e-3#30e-3
    dt = 1e-8
    titf = [t0, tf, dt]
    
    n_samples = int(1)#1000
    
    def checkSurvProjZ(absZ):
        absProj = [[1.0, 0, absZ]]
        result = survivalProb(P,T,w0,titf,s0,lambd,Gamma,absProj,delta,n_samples)
        return result
    
    def checkSurvTime(tf):
        titf = [t0, tf, dt]
        absProj = [[1.0, 0, 0.4]]
        result = survivalProb(P,T,w0,titf,s0,lambd,Gamma,absProj,delta,n_samples)
        return result
    
    absZ_scan = np.arange(0,0.4+0.05,0.1)
    tf_scan = np.arange(10,100+5, 10)*1e-3
    # absZ_scan = np.array([0,0.05,0.1,0.15,0.2])
    # absZ_scan = [0.2]
    # delta_scan = np.linspace(-3,3,10) * 180e3
    print(n_jobs)
    result = np.array([checkSurvProjZ(absZ_scan[i]) 
                for i in tqdm(range(len(absZ_scan)))])
    #result = np.array([checkSurvTime(tf_scan[i])
    #            for i in tqdm(range(len(tf_scan)))])

    # result = Parallel(n_jobs=4)(delayed(checkSurvDelta)(delta_scan[i]) 
    #                             for i in tqdm(range(len(delta_scan))))
    
    abszAngle = np.array([np.arcsin(i/np.sqrt(2+i**2)) for i in absZ_scan])
    abszAngle *= 180/np.pi
    
    result = np.array(result)  
    
    # direct = 'C:\\Users\\x2241135\\Desktop\\PhD\\codes\\new_monteCarlo\\results\\'
    # name = 'result_delta_m{0:1.0f}_sat{1:1.0f}_tf{2:1.0f}.txt'.format(dParam, s0, 1e3*tf)
    
    np.save('resultEr_absZscan.npy', result)
    
    # np.savetxt(direct+name, np.c_[abszAngle, result])
   

            
        
        
