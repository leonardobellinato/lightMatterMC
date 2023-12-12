# Monte-Carlo simulation of dynamics inside the Tweezer with #
# Photon absorption and Spontaneous emission in the presence of #
# A light field #
# Daniel S. Grun, Innsbruck 2023 #

from new_MC_functions import *
from photon_functions import *
import sys
import numpy as np

arg_input = int(sys.argv[1])
n_jobs=arg_input
# n_jobs = 4
Gamma583 = 2*np.pi * 180e3 # Gamma of the 583 nm transition (in Hz)
Gamma841 = 2*np.pi * 8e3 # Gamma of the 841 nm transition (in Hz)
Gamma626 = 2*np.pi * 135e3 # Gamma of the 626 nm transition (Dy) (in Hz)
lambd583 = 583e-9
lambd841 = 841e-9
lambd626 = 626e-9

e0 = 8.85e-12
c = 299792458
hbar = 1.0545e-34

conversion = 0.1482e-24 * 1.113e-16 # conversion from a.u. to S.I.
alpha_GS = 430 * conversion # polarizability

def checkLost(rvVector, P, w0):
    r = rvVector[:3]
    v = rvVector[3:]
    potEnergy = trapPotential(r,P,w0)
    kinEnergy = kineticEnergy(v)
    msg = potEnergy + kinEnergy
    return int(msg > 0)


def createSimulation_loss(P,T,w0,titf, s0=0, lambd=lambd583,
                  Gamma=Gamma583, absProj=[[1.0,0,0]], delta=0, alpha=[alpha_GS]):
    

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
        
        auxRatios = []
        auxMasks = []
        
        for i in range(len(absProj)):

            DopplerShift = getDopplerShift(sols, absProj[i], lambd[i])
            acStarkShift = getAcStarkShift(sols, P, w0, alpha_E=alpha[i])
            Delta = 2*np.pi*delta[i] + DopplerShift + acStarkShift

            # print(Delta/Gamma)

            scattProb = Rscatt(Gamma[i], s0[i], Delta) * dt
            auxNum = np.random.rand()
            auxRatios.append(scattProb/auxNum)
            auxMasks.append(scattProb>auxNum)
            

        auxRatio = max(auxRatios)
        index = auxRatios.index(auxRatio)

        initialCond = sols
        
        currentTime += dt
        
        lost = checkLost(initialCond, P, w0)

        if auxMasks[index]:
            #print("Horizontal"*auxMasks[0] + "Vertical"*auxMasks[1])
            #print("s0 = ", s0[index])
            sols[3:] = sols[3:] + recoilVel(absProj[index], lambd[index], case='absorption')
            initialCond = sols
            phScatt += auxMasks[index]
            waitTime = np.random.exponential(scale=1/Gamma[index])
            passedTime = 0
            while passedTime < waitTime:
                sols = odeRK45_solver(trapEvol, initialCond, dt, P, w0)
                sols = np.array(sols)

                currentTime += dt
                passedTime += dt

                initialCond = sols

            sols[3:] = sols[3:] + recoilVel(absProj[index], lambd[index], case='emission')
            initialCond = sols
        

        # print(phScatt)
        
        if lost:
            lostTime = currentTime
            currentTime = tf+1
            # print("Atom lost after {} us of illumination".format(np.round(1e6*lostTime, 2)))
            solution.append(1-lost)
            solution.append(lostTime)
            solution.append(phScatt)
            
        else:
            # print("currentTime = {} us".format(np.round(1e6*currentTime, decimals=2)))
            if currentTime >= tf:            
                solution.append(1-lost)
                solution.append(currentTime)    
                solution.append(phScatt)
                
    return solution


def runSimulation_loss(P,T,w0,titf, s0=0, lambd=lambd583, Gamma=Gamma583, absProj=[[1.0,0,0]], delta=[0], alpha=[alpha_GS],n_samples=1):
    print(n_jobs)
    results = Parallel(n_jobs=n_jobs)(delayed(createSimulation_loss)(P,T,w0,titf,s0,lambd,Gamma,absProj,delta, alpha)
                                               for i in tqdm(range(n_samples)))
    return results
        

def survivalProb(P,T,w0,titf,s0=0,lambd=lambd583,Gamma=Gamma583,absProj=[[1.0,0,0]],delta=[0],alpha=[alpha_GS], n_samples=1):
    results = runSimulation_loss(P,T,w0,titf,s0,lambd,Gamma,absProj,delta,alpha,n_samples)
    results = np.array(results)
    surv = results[:,0]
    phScatt = results[:,2]
    survProb = len(surv[surv==1])/len(surv)
    return [survProb, phScatt]

if __name__ == "__main__":
    
    Gamma1 = Gamma583
    lambd1 = lambd583

    Gamma2 = Gamma841
    lambd2 = lambd841
    
    P = 2.4e-3 # trap power, in W
    T = 20e-6 # temperature, in K
    w0 = 1.0e-6 # trap waist, in um
    s0 = 0.5 # near-resonant field, saturation parameter

    alpha_E = 1*alpha_GS

    dParam1 = 1.5
    dParam2 = -80
    delta1 = -dParam1*180e3 # detuning from resonance, in Hz
    delta2 = -dParam2*8e3 # detuning from resonance, in Hz

    t0 = 0
    tf = 70e-3
    dt = 1e-8
    titf = [t0, tf, dt]
    
    n_samples = int(300)
    
    def checkSurvIntVert(sScan):
        absProj = [[1.0, 0, 0.18],
                    [0, 0, -1.0]]
        result = survivalProb(P,T,w0,titf,[s0,sScan],[lambd1, lambd2], [Gamma1, Gamma2],absProj,[delta1, delta2], [alpha_GS, alpha_GS], n_samples)
        return result

    def checkSurvDeltaVert(dScan):
        absProj = [[1.0, 0, 0.18], [0, 0, 1.0]]
        s2 = 10
        delta2 = -dScan*8e3 # how many linewidth of detuning? 
        result = survivalProb(P,T,w0,titf,[s0,s2],[lambd1,lambd2],[Gamma1,Gamma2],absProj,[delta1,delta2], [alpha_GS,0.75*alpha_GS], n_samples)
        return result

    #vertBeamInt_scan = np.arange(0,0.3,0.03)
    vertBeamD_scan = -np.arange(80,83,0.25)

    print(n_jobs)
    #result = np.array([checkSurvIntVert(vertBeamInt_scan[i]) 
    #            for i in tqdm(range(len(vertBeamInt_scan)))])
    
    # result = np.array([checkSurvDeltaVert(vertBeamD_scan[i])
    #             for i in tqdm(range(len(vertBeamD_scan)))])

    result = createSimulation_loss(P,T,w0,titf, s0=[0.8], lambd=[lambd583],
                      Gamma=[Gamma583], absProj=[[1.0,0,0.3]], delta=[-1.5*180e3], alpha=[alpha_GS])

    result = np.array(result)  
    
    # direct = 'C:\\Users\\x2241135\\Desktop\\PhD\\codes\\new_monteCarlo\\results\\'
    # name = 'result_delta_m{0:1.0f}_sat{1:1.0f}_tf{2:1.0f}.txt'.format(dParam, s0, 1e3*tf)
    
    np.save('resultEr_1horBeam_1vert841Beam_dm1-5_dmFinerScan_s0-5_s100_70ms.npy', result)
    
    # np.savetxt(direct+name, np.c_[abszAngle, result])
    
    

            
        
        
