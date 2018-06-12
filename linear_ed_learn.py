#Solves NK model with learning about change in intercept of rule.


#compile model
#what is the ordering of parameters in p0 -- where do I find in yaml file?
from MarkovSwitchingModel import MarkovSwitchingDSGE
nklearn = MarkovSwitchingDSGE.read('3eq_ms.yaml')
nk = nklearn.compile_model()
p0 = nklearn.p0()

#get solution matrices from gensys
#How do the matrices coming from gensys ftn map into equation 20?
##TT maps  from x(-1) to x, RR maps from innovations to x(0)
from dsge.gensys import gensys
import numpy as np
yy = np.array(nk.yy)
nobs = yy.shape[0]
GAM0 = nk.GAM0(p0)
GAM1 = nk.GAM1(p0)
PSI = nk.PSI(p0)
PPI = nk.PPI(p0)
TT, RR, M, TZ, TY, RC = gensys(GAM0, GAM1, PSI, PPI, return_everything=True)
TT, RR, QQ, DD, ZZ, HH = nk.system_matrices(p0)

#construct expectation of future shocks
#I need to be walked through this section again.  
initial_distribution = np.array(nk.initial_distribution(p0),dtype=float).squeeze()
filtered_prob = initial_distribution
strans = nk.transition(p0)
sigrq = np.asarray(np.sqrt(nk.conditional_variance(p0))).squeeze()
epsr_m = nk.conditional_mean(p0).squeeze()
ns = TT.shape[0]
neps = RR.shape[1]
ny = ZZ.shape[1]
nj = initial_distribution.size
EXPeffect_f = np.zeros((ns, nj))
EXPeffect_m = np.zeros((ns, nj))

noshock = ~(np.arange(neps)==nk.shock_ind)
Mj = np.eye(M.shape[0])
strans = np.atleast_2d(strans)
stransj = strans
shocksel = np.zeros(neps)
shocksel[~noshock] = 1

filtered_mean_mat = np.zeros((nobs, ns))
filtered_prob_mat = np.zeros((nobs, nj))
epsr_m = np.atleast_1d(np.array(epsr_m))

for j in range(100):
    EXPeffect_f = EXPeffect_f +  (TY @ Mj @ TZ @ shocksel)[:, np.newaxis] @ ( epsr_m.T @ stransj )[np.newaxis, :]
    #EXPeffect_m = EXPeffect_m +  TY @ Mj @ TZ @ shocksel @ ( epsr_m.T )

    Mj = Mj @ M
    stransj = stransj @ strans
EXPeffect_f = np.real(EXPeffect_f)

print(EXPeffect_f.round(3))
print(nk.state_names)

