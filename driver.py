#Compares IRFs from NK model with learning under two solution algorithms.

import numpy as np
import polydef as po
import model_details as md
import sys

import importlib
importlib.reload(md)

#####################################
#Compute IRFs from linear solution algorithm plus learning
###################################

from MarkovSwitchingModel import MarkovSwitchingDSGE
nklearn = MarkovSwitchingDSGE.read('3eq_ms.yaml')
nk = nklearn.compile_model()
p0 = nklearn.p0()
paramlist = ['beta_tr','sigma_r','rho_e','sigma_e','rho_z','sigma_z','rho_R','psi_dp','psi_y','sigma_r4','rho_r4']
p0[0] = 0.005
p0[1] = 0.08
p0[2] = 0.85
p0[3] = 0.001
p0[6] = 0.0
p0[7] = 1.5
p0[8] = 0.0
p0[9] = 0.1
p0[10] = 0.9
param_d = dict(zip(paramlist,p0))

nkirf = nk.impulse_response(p0, shock=0.,h=1)

############################################
#Computes IRFs from projection method.
############################################

get_solution_from_disk = False

#To make parameters equivalent to above may need to make changes to yaml file (gamma,ap,phip,etc.)  
params = {'beta': 1/(1+p0[0]/100), 'eta': 0.0025, 'gamma': 0.0, 'epp': 6.0, 'phip': 100.0, 'dpss': 0.005, 'ap': 1.0, 'rhor' : p0[6], 'gammapi': p0[7], 'gammax': p0[8], \
          'stdm' : p0[1]/100, 'rhoeta' : p0[2], 'stdeta' : p0[3]/100, 'rhogam': p0[10], 'stdgam' : p0[9]/100}

nexog_fe = 1  #one shock on finite element part of grid (eta)
nendog_nmsv = 3 #3 endogenous state variables (besides probabilities)
nsreg = 2  #number of regimes (nsreg-1 additional state variables)
ngrid = np.ones(nexog_fe,dtype=int)
ngrid[0] = 3
acoeff0,poly = po.initialize_poly(nexog_fe,nendog_nmsv,nsreg,ngrid) 

#construct regimes from Rouwenhurst
p_gam = (1.0+params['rhogam'])/2.0
poly['P'] = po.transmat(nsreg,p_gam,p_gam)
psi_gam = np.sqrt(nsreg-1)*params['stdgam']
poly['gamma0'] = np.linspace(-psi_gam,psi_gam,nsreg)

#solve model
if (get_solution_from_disk == False):
    acoeff,convergence = md.solve_model(acoeff0,params,poly)
    if (convergence == True):
        print('Model solved successfully.')
        np.save('solution_coeffs.npy',acoeff)
    else:
        print('Model failed to solve')
else:
    poly = md.get_griddetails(poly,params)
    acoeff = np.load('solution_coeffs.npy')

#construct IRFs and simulate data from the model
innov = np.zeros(poly['ninnov'])
endogvarm1 = np.zeros(poly['nvars'])
nx = poly['nmsv']-poly['nexog_nmsv']-(poly['nsreg']-1)
if (nsreg == 2):
    endogvarm1[nx] = np.log(0.5)
    regime = 1
    innov[1] = -poly['gamma0'][regime]/params['stdm']
else:
    endogvarm1[nx:nx+nsreg-1] = np.log(0.25)
    regime = 1


TT = 5000
irfdf = md.simulate(TT,endogvarm1,innov,regime,acoeff,poly,params,irfswitch=0)
model_stats = irfdf.describe()
    
TT = 1
irfdf = md.simulate(TT,endogvarm1,innov,regime,acoeff,poly,params,irfswitch=1)




####nsreg = 3
#acoeff elements:
#R 1 2
#y 3 4
#dp 5 6
# log(p1) 7 8
# log(p3) 9 10
#  agg 11 12
