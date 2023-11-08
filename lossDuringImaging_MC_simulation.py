# New Monte-Carlo suite of codes #
# for simulation of atom dynamics inside # 
# optical tweezer + light-matter interaction # 

# Author: Daniel S. Grun #
# Innsbruck, 2023 # 

from new_MC_functions import *
from photon_functions import *
from energies_transitions import Gammas, Wls, alpha_GS
import sys
import numpy as np
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-j', '--njobs', required=True, type=int, help='# of parallel jobs')
parser.add_argument('-p', '--power', default=2, type=float, help='Tweezer power (mW)')
parser.add_argument('-w', '--waist', default=1, type=float, help='Tweezer waist (um)')
parser.add_argument('-wls', '--wavelengths', default=['583'], type=list, help='Wavelength of each beam (nm)')
parser.add_argument('-s', '--sat', default=[1], type=list, help='Sat. parameter for each beam')
parser.add_argument('-d', '--delta', default=[0], type=list, 
                    help='Detuning from resonance for each beam (in terms of respective scattering length)')
parser.add_argument('-tf', '--tfinal', default=30, type=float, help='Exposure time (ms)')
parser.add_argument('-n', '--nsamples', default=100, type=int, help='# of samples')

args = parser.parse_args()

n_jobs=args['njobs']

def checkLost(rvVector, P, w0):
    r = rvVector[:3]
    v = rvVector[3:]
    potEnergy = trapPotential(r,P,w0)
    kinEnergy = kineticEnergy(v)
    msg = potEnergy + kinEnergy
    return int(msg > 0)
    

def createSimulation_loss(P,T,w0,titf, s0=[0], lambd=[lambd583],
                  Gamma=[Gamma583], absProj=[[1.0,0,0]], delta=[0], alphas=[1]):
    

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
            acStarkShift = getAcStarkShift(sols, P, w0, alpha_E=alphas[i])
            Delta = delta + DopplerShift + acStarkShift

            # print(Delta/Gamma)

            scattProb = Rscatt(Gamma[i], s0[i], Delta[i]) * dt
            auxNum = np.random.rand()
            auxRatios.append(scattProb/auxNum)
            auxMasks.append(scattProb>auxNum)
            

        auxRatio = max(auxRatios)
        index = auxRatios.index(auxRatio)

        initialCond = sols
        
        currentTime += dt
        
        lost = checkLost(initialCond, P, w0)

        if auxMasks[index]:
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

            sols[3:] = sols[3:] + recoilVel(absProj[index], lambd, case='emission')
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


def runSimulation_loss(P,T,w0,titf, s0=[0], lambd=[lambd583], Gamma=[Gamma583], absProj=[[1.0,0,0]], delta=[0], n_samples=1):
    print(n_jobs)
    results = Parallel(n_jobs=n_jobs)(delayed(createSimulation_loss)(P,T,w0,titf,s0,lambd,Gamma,absProj,delta)
                                               for i in tqdm(range(n_samples)))
    return results
        

def survivalProb(P,T,w0,titf,s0=[0],lambd=[lambd583],Gamma=[Gamma583],absProj=[[1.0,0,0]],delta=[0],n_samples=1):
    results = runSimulation_loss(P,T,w0,titf,s0,lambd,Gamma,absProj,delta,n_samples)
    results = np.array(results)
    surv = results[:,0]
    phScatt = results[:,2]
    survProb = len(surv[surv==1])/len(surv)
    return [survProb, phScatt]

if __name__ == "__main__":
    
    Gamma = [Gammas[name] for name in args['wavelengths']]
    lambd = [Wls[name] for name in args['wavelengths']]
    
    P = 1e-3 * args['power'] # trap power, in W
    w0 = 1e-6 * args['waist'] # trap waist, in um
    s0 = args['sat'] # near-resonant field, saturation parameter 
    dParam = args['delta']
    delta = [dParam[i] * Gamma[i] for i in range(len(Gamma))]
    
    t0 = 0
    tf = args['tfinal']
    dt = 1e-8
    titf = [t0, tf, dt]
    
    n_samples = args['nsamples']
    
    def checkSurvProjZ(absZ):
        absProj = [[1.0, 0, absZ], [0,1.0,absZ]] # needs to have same length as gammas/lambds/# of beams
        result = survivalProb(P,T,w0,titf,s0,lambd,Gamma,absProj,delta,n_samples)
        return result
    
    def checkSurvTime(tf):
        titf = [t0, tf, dt]
        absProj = [[1.0, 0, 0.4], [0, 1.0, 0.4]] # needs to have same length as gammas/lambds/# of beams
        result = survivalProb(P,T,w0,titf,s0,lambd,Gammas,absProj,delta,n_samples)
        return result
    
    absZ_scan = np.arange(0,0.4+0.05,0.1)
    tf_scan = np.arange(10,100+5, 10)*1e-3
    # absZ_scan = np.array([0,0.05,0.1,0.15,0.2])
    # absZ_scan = [0.2]
    # delta_scan = np.linspace(-3,3,10) * 180e3
    print(n_jobs)
    #result = np.array([checkSurvProjZ(absZ_scan[i]) 
    #            for i in tqdm(range(len(absZ_scan)))])
    result = np.array([checkSurvTime(tf_scan[i])
                for i in tqdm(range(len(tf_scan)))])

    # result = Parallel(n_jobs=4)(delayed(checkSurvDelta)(delta_scan[i]) 
    #                             for i in tqdm(range(len(delta_scan))))
    
    abszAngle = np.array([np.arcsin(i/np.sqrt(2+i**2)) for i in absZ_scan])
    abszAngle *= 180/np.pi
    
    result = np.array(result)  
    
    # direct = 'C:\\Users\\x2241135\\Desktop\\PhD\\codes\\new_monteCarlo\\results\\'
    # name = 'result_delta_m{0:1.0f}_sat{1:1.0f}_tf{2:1.0f}.txt'.format(dParam, s0, 1e3*tf)
    
    np.save('result_expTimeScan_20degZ_bothBeams_sover2.npy', result)
    
    # np.savetxt(direct+name, np.c_[abszAngle, result])
    
    

            
        
        
