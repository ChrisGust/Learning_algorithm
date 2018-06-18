import numpy as np

import sympy
from dsge.Prior import Prior as pri
from dsge import StateSpaceModel
from dsge.DSGE import DSGE

from sympy.stats import density
import pandas as p
from scipy.linalg import solve_discrete_lyapunov

from dsge.gensys import gensys

from numba import njit



# def _markov_switching_fullinfo_lik(yy, TT, RR, QQ, DD, ZZ, HH, EXPeffect_f, Pt, obs_ind,
#                                    shock_ind, initial_distribution, transition, shock_scaling,
#                                    conditional_mean, conditional_variance, shocks):


#     nobs = yy.shape[0]
#     neps = RR.shape[1]
#     ny, ns = ZZ.shape
#     noshock = ~(np.arange(neps)==shock_ind)
#     RRnoshock = RR[:, noshock]
#     QQnoshock = QQ[noshock, :][:,noshock]


#     noobs = ~(np.arange(ny)==obs_ind)
#     ZZnoobs = ZZ[noobs, :]
#     DDnoobs = DD[noobs]
#     RQR = np.dot(np.dot(RR, QQ), RR.T)
#     RQR2 = RRnoshock @ QQnoshock@ RRnoshock.T

#     Pt = 0.5*(Pt+Pt.T)
#     At = np.zeros(shape=(ns))

#     yhat = ZZ @ At + DD
#     nut = yy[0, :] - yhat
#     Ft = ZZ @ Pt @ ZZ.T
#     Ft = 0.5*(Ft + Ft.T)
#     iFtnut = np.linalg.solve(Ft, nut)


#     At = At + Pt @ ZZ.T @ iFtnut;
#     Pt = Pt - Pt @ ZZ.T @ np.linalg.inv(Ft) @ ZZ @ Pt.T

#     filtered_prob = initial_distribution
#     strans = transition
#     sigrq = np.sqrt(conditional_variance)

#     epsr_m = conditional_mean
    
#     nj = initial_distribution.size
#     nhist = 3
#     nbranchesmax = nj**nhist

#     filtered_mean_mat = np.zeros((nobs, ns))
#     filtered_prob_mat = np.zeros((nobs, nj))

#     # markov swithching
#     QQs = np.array([QQ for _ in range(nj)])
#     if sigrq.size != nj: sigrq = sigrq*np.ones(nj)
#     sigrq = np.atleast_1d(sigrq)

#     for j in range(nj):
#         QQs[j][self.shock_ind, self.shock_ind] = sigrq[j]**2

#     EXPeffect_f = np.real(EXPeffect_f)
#     probx = np.array(1)
#     probs = np.atleast_1d(initial_distribution)
#     Ats = At[np.newaxis,:]
#     Pts = Pt[np.newaxis,:]

#     # condition on first observation
#     loglh = 0.0



@njit
def _markov_switching_learning_lik(yy, TT, RR, QQ, DD, ZZ, HH, EXPeffect_f, Pt, obs_ind,
                                   shock_ind, initial_distribution, transition, shock_scaling,
                                   conditional_mean, conditional_variance, shocks):


    nobs = yy.shape[0]
    neps = RR.shape[1]
    ny, ns = ZZ.shape
    noshock = ~(np.arange(neps)==shock_ind)
    RRnoshock = RR[:, noshock]
    QQnoshock = QQ[noshock, :][:,noshock]


    noobs = ~(np.arange(ny)==obs_ind)
    ZZnoobs = ZZ[noobs, :]
    DDnoobs = DD[noobs]
    RQR = np.dot(np.dot(RR, QQ), RR.T)
    RQR2 = RRnoshock @ QQnoshock@ RRnoshock.T

    Pt = 0.5*(Pt+Pt.T)
    At = np.zeros(shape=(ns))

    yhat = ZZ @ At + DD
    nut = yy[0, :] - yhat
    Ft = ZZ @ Pt @ ZZ.T + HH
    Ft = 0.5*(Ft + Ft.T)
    iFtnut = np.linalg.solve(Ft, nut)


    At = At + Pt @ ZZ.T @ iFtnut;
    Pt = Pt - Pt @ ZZ.T @ np.linalg.inv(Ft) @ ZZ @ Pt.T

    filtered_prob = initial_distribution
    strans = transition
    sigrq = np.sqrt(conditional_variance)

    epsr_m = conditional_mean

    nj = initial_distribution.size
    
    filtered_mean_mat = np.zeros((nobs, ns))
    filtered_prob_mat = np.zeros((nobs, nj))

    # condition on first observation
    loglh = 0.0
    for t in range(1, nobs):

        At1 = At.copy()
        Pt1 = Pt.copy()

        forecast_prob = (strans @ filtered_prob)#.squeeze()

        epsr = shocks[t]

        zetat = np.exp(-0.5*((epsr-epsr_m)/sigrq)**2)/sigrq
        
        loglh = loglh - 0.5*np.log(2*np.pi) - np.log(shock_scaling) + np.log((forecast_prob*zetat).sum())

        filtered_prob = forecast_prob * zetat / (forecast_prob * zetat).sum()

        # forecasting
        At = TT @ At1 + RR[:, shock_ind]*epsr  + EXPeffect_f @ filtered_prob
        Pt = TT @ Pt1 @ TT.T + RQR2
        Pt = 0.5*(Pt + Pt.T)

        observed = ~np.isnan(yy[t][noobs])

        yhat = ZZnoobs[observed, :] @ At + DDnoobs[observed].flatten()
        nut = yy[t][noobs][observed] - yhat

        Ft = ZZnoobs[observed, :] @ Pt @ ZZnoobs[observed, :].T + (HH[noobs, :][:, noobs])[observed, :][:, observed]
        Ft = 0.5 * (Ft + Ft.T)

        dFt = np.log(np.linalg.det(Ft))
        iFtnut = np.linalg.solve(Ft, nut)

        loglh = loglh - 0.5*nut.size*np.log(2*np.pi) - 0.5*dFt - 0.5*np.dot(nut, iFtnut)
        
        At = At + Pt @ ZZnoobs[observed, :].T @ iFtnut;
        Pt = Pt - Pt @ ZZnoobs[observed, :].T @ np.linalg.inv(Ft) @ ZZnoobs[observed, :] @ Pt.T

        filtered_mean_mat[t] = At
        filtered_prob_mat[t] = filtered_prob

    return loglh, filtered_mean_mat, filtered_prob_mat


class MarkovSwitchingStateSpaceModel(StateSpaceModel.LinearDSGEModel):

    def __init__(self, *args, **kwargs):


        shock_var = kwargs.pop('shock')
        obs_var = kwargs.pop('observable')
        initial_distribution = kwargs.pop('initial_distribution') 
        transition = kwargs.pop('transition')           
        conditional_mean = kwargs.pop('conditional_mean')     
        conditional_variance = kwargs.pop('conditional_variance')
        construct_shock = kwargs.pop('construction')      
        shock_scaling = kwargs.pop('shock_scaling')

        super().__init__(*args, **kwargs)

        self.shock_var = shock_var
        self.obs_var = obs_var
        self.initial_distribution = initial_distribution
        self.transition = transition
        self.conditional_mean = conditional_mean
        self.conditional_variance = conditional_variance
        self.construct_shock = construct_shock
        self.shock_scaling = shock_scaling
        self.shock_ind = self.shock_names.index(self.shock_var)
        self.obs_ind = self.obs_names.index(self.obs_var)

    def get_EXPeffect(self, p0, nsum=100):

        # Solve the model
        GAM0 = self.GAM0(p0)
        GAM1 = self.GAM1(p0)
        PSI = self.PSI(p0)
        PPI = self.PPI(p0)

        TT, RR, M, TZ, TY, RC = gensys(GAM0, GAM1, PSI, PPI, return_everything=True)

        # get EXPeffect_f
        initial_distribution = np.array(self.initial_distribution(p0),dtype=float).squeeze()
        filtered_prob = initial_distribution
         
        strans = self.transition(p0)
        sigrq = np.asarray(np.sqrt(self.conditional_variance(p0))).squeeze()
        epsr_m = self.conditional_mean(p0).squeeze()
         
        ns, nj = TT.shape[0], initial_distribution.size
        EXPeffect_f = np.zeros((ns, nj))
         
        Mj = np.eye(M.shape[0])
        strans = np.atleast_2d(strans)
        stransj = strans

        neps = RR.shape[1]
        noshock = ~(np.arange(neps)==self.shock_ind)
        shocksel = np.zeros(neps)
        shocksel[~noshock] = 1


        for j in range(nsum):
            # forward looking
            EXPeffect_f = EXPeffect_f +  (TY @ Mj @ TZ @ shocksel)[:, np.newaxis] @ ( epsr_m.T @ stransj )[np.newaxis, :]

            # myopic 
            #EXPeffect_m = EXPeffect_m +  TY @ Mj @ TZ @ shocksel @ ( epsr_m.T )

            Mj = Mj @ M;
            stransj = stransj @ strans

        EXPeffect_f = np.real(EXPeffect_f)
        
        return EXPeffect_f

       
    def impulse_response(self, p0, shock=1., initial_distribution=None, h=20):
        
        if np.array(shock).size==1:
            shocks = np.zeros((h,))
            shocks[0] = shock
        else:
            shocks = np.array(shock)
            h = shocks.size

        # get standard irfs
        # irfs = super().impulse_response(p0, h=h)

        EXPeffect_f = self.get_EXPeffect(p0)
        TT, RR, QQ, DD, ZZ, HH = self.system_matrices(p0)

        if initial_distribution is None:
            initial_distribution = np.array(self.initial_distribution(p0),dtype=float).squeeze()

        strans = self.transition(p0)
        sigrq = np.asarray(np.sqrt(self.conditional_variance(p0))).squeeze()
        epsr_m = self.conditional_mean(p0).squeeze()

        shock_var = self.shock_var
        shock_ind = self.shock_names.index(shock_var)

        At = np.zeros((TT.shape[0], ))
        forecast_prob = initial_distribution

        irf_states = np.zeros((h, TT.shape[0]))
        nj = initial_distribution.size
        irf_probs = np.zeros((h, nj))

        for t in range(h):

            epsr = shocks[t]
            
            zetat = np.exp(-0.5*((epsr-epsr_m)/sigrq)**2)/sigrq
            filtered_prob = forecast_prob * zetat / (forecast_prob * zetat).sum()
            At = TT @ At + RR[:, shock_ind]*epsr  + EXPeffect_f @ filtered_prob

            irf_states[t] = At
            irf_probs[t] = filtered_prob

            forecast_prob = strans @ filtered_prob

        Ats = p.DataFrame(irf_states, columns=self.state_names, index=range(h))
        xts = p.DataFrame(irf_probs, columns=['pj%d' % d for d in range(nj)], index=range(h))
        res = p.concat([Ats, xts], axis=1)
        
        #irfs[shock_var] = res

        return res
 

    def log_lik(self, p0, y=None, return_filtered=False):

       yy = np.array(self.yy)

       TT, RR, QQ, DD, ZZ, HH = self.system_matrices(p0)
       EXPeffect_f = self.get_EXPeffect(p0)

       RQR = np.dot(np.dot(RR, QQ), RR.T)

       try:
           Pt = solve_discrete_lyapunov(TT, RQR)
           Pt = TT @ Pt @ TT.T + RQR
       except:
           return -1000000000.0

       initial_distribution = np.array(self.initial_distribution(p0),dtype=float).squeeze()

       strans = self.transition(p0)
       sigrq = np.asarray(np.sqrt(self.conditional_variance(p0))).squeeze()
       epsr_m = self.conditional_mean(p0).squeeze()

       nobs = yy.shape[0]
       shocks = np.zeros(nobs)
       for t in range(1, nobs):
           shocks[t]  = self.construct_shock(p0, yy[t], yy[t-1])


       epsr_m = np.atleast_1d(np.array(epsr_m))


       try:
           res = _markov_switching_learning_lik(yy, TT, RR, QQ, DD.squeeze(),
                                                np.asarray(ZZ,dtype=float), HH,
                                                EXPeffect_f,
                                                Pt, self.obs_ind,
                                                self.shock_ind, 
                                                np.asarray(np.atleast_1d(initial_distribution), dtype=float), 
                                                np.asarray(np.atleast_2d(strans), dtype=float), self.shock_scaling,
                                                np.asarray(epsr_m, dtype=float), sigrq**2, shocks)

       except:
           res = -100000000.0, 0.0, 0.0
    
       loglh, filtered_mean_mat, filtered_prob_mat = res

 
       if return_filtered:
           Ats = p.DataFrame(filtered_mean_mat, columns=self.state_names, index=self.yy.index)
           xts = p.DataFrame(filtered_prob_mat, columns=['pj%d' % d for d in range(nj)], index=self.yy.index)

           res = p.concat([Ats, xts], axis=1)
           return loglh, res
       else:
           if np.isnan(loglh): loglh = -10000000000
           return loglh


    def log_lik_fullinfo(self, p0, y=None, return_filtered=False):

       initial_distribution = np.array(self.initial_distribution(p0),dtype=float).squeeze()
       filtered_prob = initial_distribution

       strans = self.transition(p0)
       sigrq = np.asarray(np.sqrt(self.conditional_variance(p0))).squeeze()
       epsr_m = np.atleast_1d(self.conditional_mean(p0).squeeze())

       nj = initial_distribution.size
       nhist = 1
       nbranchesmax = nj**nhist

       yy = np.array(self.yy)
       nobs = yy.shape[0]
       GAM0 = self.GAM0(p0)
       GAM1 = self.GAM1(p0)
       PSI = self.PSI(p0)
       PPI = self.PPI(p0)

       TT, RR, M, TZ, TY, RC = gensys(GAM0, GAM1, PSI, PPI, return_everything=True)
       TT, RR, QQ, DD, ZZ, HH = self.system_matrices(p0)

       ns,neps = RR.shape
       noshock = ~(np.arange(neps)==self.shock_ind)
       
       # computing expectational effects
       EXPeffect_f = np.zeros((ns, nj))
       EXPeffect_m = np.zeros((ns, nj))
       Mj = np.eye(M.shape[0])
       strans = np.atleast_2d(strans)
       stransj = strans
       shocksel = np.zeros(neps)
       shocksel[~noshock] = 1

       filtered_mean_mat = np.zeros((nobs, ns))
       filtered_prob_mat = np.zeros((nobs, nj))


       for j in range(100):
           EXPeffect_f = EXPeffect_f +  (TY @ Mj @ TZ @ shocksel)[:, np.newaxis] @ ( epsr_m.T @ stransj )[np.newaxis, :]
           Mj = Mj @ M;
           stransj = stransj @ strans

       EXPeffect_f = np.real(EXPeffect_f)


       # t = 0 filtering 
       RQR = np.dot(np.dot(RR, QQ), RR.T)
       try:
           Pt = solve_discrete_lyapunov(TT, RQR)
       except:
           return -1000000000.0
       # initia distribution
       At = np.zeros(shape=(ns))
       yhat = ZZ @ At + DD.squeeze()
       nut = yy[0, :] - yhat
       Ft = ZZ @ Pt @ ZZ.T
       Ft = 0.5*(Ft + Ft.T)
       iFtnut = np.linalg.solve(Ft, nut)
       At = At + Pt @ ZZ.T @ iFtnut;
       Pt = Pt - Pt @ ZZ.T @ np.linalg.inv(Ft) @ ZZ @ Pt.T

       
       # markov swithching
       QQs = np.array([QQ for _ in range(nj)])
       if sigrq.size != nj: sigrq = sigrq*np.ones(nj)
       sigrq = np.atleast_1d(sigrq)

       for j in range(nj):
           QQs[j][self.shock_ind, self.shock_ind] = sigrq[j]**2

       EXPeffect_f = np.real(EXPeffect_f)
       probx = np.array(1)
       probs = np.atleast_1d(initial_distribution)
       Ats = At[np.newaxis,:]
       Pts = Pt[np.newaxis,:]
 
       #loglh, filtered_mean_mat, filtered_prob_mat = logl

       # condition on first observation
       loglh = 0.0
 
       for t in range(1, nobs):

           observed = ~np.isnan(yy[t])

           nbranches = Ats.shape[0]
 

           probs = strans @ probs.T
           probx = np.atleast_2d(probx)
           probx2 = np.kron(np.atleast_2d(probs), probx).flatten()
 
           Atsnew = np.zeros((nbranches*nj,ns))
           Ptsnew = np.zeros((nbranches*nj,ns,ns))
           loglhsnew = np.zeros(nbranches*nj)
 
           for i in range(nbranches):
               alphahat = np.array([TT @ Ats[i] + RR[:,self.shock_ind] * epsr_m[ii] + EXPeffect_f[:, ii] for ii in range(nj)])
               TPTt = TT @ Pts[i] @ TT.T
               Phat = np.array([TPTt + RR @ QQs[ii] @ RR.T for ii in range(nj)])
 
               yhats = ZZ[observed,:] @ alphahat.T + DD[observed]
               nuts = yy[t,observed][:,np.newaxis] - yhats
 
               Fts = np.array([ZZ[observed,:] @ Phat[ii] @ ZZ[observed,:].T + HH[np.ix_(observed,observed)] for ii in range(nj)])
               Fts = np.array([0.5*(Ft + Ft.T) for Ft in Fts])
 
               iFtnuts = [np.linalg.solve(Fts[ii], nuts[:,ii]) for ii in range(nj)]
               loglhs =  [(-0.5*nuts[:,ii].size*np.log(2*np.pi)
                           -0.5*np.log(np.linalg.det(Fts[ii]))
                           -0.5*np.dot(nuts[:,ii], iFtnuts[ii]))
                          for ii in range(nj)]
 
               At = np.array([alphahat[ii] + Phat[ii] @ ZZ[observed,:].T @ iFtnuts[ii] for ii in range(nj)])
               Pt = np.array([Phat[ii] - Phat[ii] @ ZZ[observed,:].T @ np.linalg.inv(Fts[ii]) @ ZZ[observed,:] @ Phat[ii].T
                         for ii in range(nj)])
 
               inds = np.arange(i, nj*nbranches, nbranches, dtype=int)
               Atsnew[inds] = At
               Ptsnew[inds] = Pt
               loglhsnew[inds] = loglhs

           loglhmax = np.max(loglhsnew)
           loglh = loglh + loglhmax + np.log(probx2.dot(np.exp(loglhsnew - loglhmax)))
           probx2 = probx2 * np.exp(loglhsnew  - loglhmax) / probx2.dot(np.exp(loglhsnew-loglhmax))
           probs = np.array([probx2[ii*nbranches:(ii+1)*nbranches].sum() for ii in range(nj)])
           #print('%f,%f,%f' %(t+1,loglh,loglhmax + np.log(probx2.dot(np.exp(loglhsnew - loglhmax)))))
           #print(probs)
           zeroprob = probx2 < 1E-5
            
           for filti in range(nj*nbranches):
               filtered_mean_mat[t]  = probx2[filti] * Atsnew[filti] + filtered_mean_mat[t]
           filtered_prob_mat[t] = probs
            
           if zeroprob.sum() >= (nj*nbranches - nbranchesmax):
               #print(t, 'Deleting ', zeroprob.sum(), 'branches (', nbranches*nj, ') total')
               Ats = Atsnew[~zeroprob]
               Pts = Ptsnew[~zeroprob]
               probx = probx2[~zeroprob]
               #print(Ats.shape)
           else:
               #print(t, 'Need to merge', nj*nbranches, 'is greater than ', nbranchesmax)
               selmat = np.kron(np.eye(nbranches), np.ones(nj))
            
               sumprobx2 = selmat @ probx2
               sumzeroprob = selmat @ zeroprob
            
               inds = np.argsort(sumprobx2 + sumzeroprob)[:nj*nbranches-nbranchesmax-zeroprob.sum()]
            
               Ats = np.zeros((nbranchesmax,ns))
               Pts = np.zeros((nbranchesmax,ns,ns))
               probx = np.zeros(nbranchesmax)
            
            
               jj = 0
               ii = 0
            
               while ii < nbranches:
                   if sumzeroprob[ii]>0.0:

                       px = 0.0
                       if ii in inds:
                           inds2 = ii*nj + np.arange(nj)
                           probx[jj] = probx2[inds2].sum()
                           for iii in inds2:
                               Ats[jj] = Ats[jj] + probx2[iii]/probx[jj]*Atsnew[iii]
                           for iii in inds2:
                               Pts[jj] = Pts[jj] + probx2[iii]/probx[jj]*(Ptsnew[iii]
                                                                    + np.outer(Ats[jj]-Atsnew[iii], Ats[jj]-Atsnew[iii]))
                           #print(jj,ii,inds2,'collapse')
                           jj += 1       
                       else:
                           for b in range(nj):
                               i0 = ii*nj + b
                               if ~zeroprob[i0]:
                                   #print(jj,ii,i0,'zeroprob',probx2[i0])
                                   Ats[jj] = Atsnew[i0]
                                   Pts[jj] = Ptsnew[i0]
                                   probx[jj] = probx2[i0]
                                   jj += 1

                   else:
                       if  ii in inds:
                           # collapse density
                           inds2 = ii*nj + np.arange(nj)
                           probx[jj] = probx2[inds2].sum()
                           for iii in inds2:
                               Ats[jj] = Ats[jj] + probx2[iii]/probx[jj]*Atsnew[iii]
                           for iii in inds2:
                               Pts[jj] = Pts[jj] + probx2[iii]/probx[jj]*(Ptsnew[iii]
                                                                    + np.outer(Ats[jj]-Atsnew[iii], Ats[jj]-Atsnew[iii]))
                           #print(jj,ii,inds2,'collapse')
                           jj += 1       
                       else:
                           for b in range(nj):
                               i0 = ii*nj + b
                               try:
                                   Ats[jj] = Atsnew[i0]
                                   Pts[jj] = Ptsnew[i0]
                                   probx[jj] = probx2[i0]
                                   #print(jj,ii,i0,'save')
                               except:
                                   print("Error!", jj, i0, b)
                                   #Ats[jj]
                                   #Atsnew[i0]
                               jj += 1
                   ii += 1

           probx = probx / probx.sum()


       if return_filtered:
           Ats = p.DataFrame(filtered_mean_mat, columns=self.state_names, index=self.yy.index)
           xts = p.DataFrame(filtered_prob_mat, columns=['pj%d' % d for d in range(nj)], index=self.yy.index)

           res = p.concat([Ats, xts], axis=1)
           return loglh, res
       else:
           if np.isnan(loglh): loglh = -10000000000
           return loglh






def transmat(n,p,q):
    #Contruct transition matrix using Rouwenhorst method
    if (n == 1):
        return np.array([[1]])

    P0 = np.array([[p,1-p],[1-q,q]])
    for i in np.arange(2,n):
        Pnew = np.zeros([i+1,i+1])
        z0 = np.zeros([i+1,i+1])
        z1 = np.zeros([i+1,i+1])
        z2 = np.zeros([i+1,i+1])
        z3 = np.zeros([i+1,i+1])
        z0[:i,:i] = P0
        z1[:i,1:] = P0
        z2[1:,:i] = P0
        z3[1:,1:] = P0
        mat_sum = p*z0+(1-p)*z1+(1-q)*z2+q*z3
        Pnew[0,:] = mat_sum[0,:]
        Pnew[i,:] = mat_sum[i,:]
        Pnew[1:i,:] = 0.5*mat_sum[1:i,:]
        P0 = Pnew
    return(P0)

def rouwenhorst_conditional_mean(rho, sigma, ns):
    psi = np.sqrt(ns-1)*sigma
    return np.linspace(-psi, psi, ns)


def rouwenhorst_transition(rho, sigma, ns):
    p = (1 + rho)/2
    q = p
    return transmat(ns, p, q)

from sympy.stats import Binomial

def rouwenhorst_initial_distribution(rho, sigma, ns):
    p = (1 + rho)/2
    q = p
    s = (1-p)/(2-(p+q))
    return list(density(Binomial('x', ns-1, s)).dict.values())


def rouwenhorst_kron_conditional_mean(rho1, sigma1, ns1, rho2, sigma2, ns2):
    rcm1 = rouwenhorst_conditional_mean(rho1,sigma1,ns1)
    rcm2 = rouwenhorst_conditional_mean(rho1,sigma2,ns2)
    rcm = np.zeros(ns1*ns2)
    for j in np.arange(ns1):
        rcm[ns2*j:ns2*(j+1)] = rcm2+rcm1[j]
    return(rcm)

def rouwenhorst_kron_transition(rho1, sigma1, ns1, rho2, sigma2, ns2):
    P1 = rouwenhorst_transition(rho1,sigma1,ns1)
    P2 = rouwenhorst_transition(rho2,sigma2,ns2)
    return(np.kron(P1,P2))

def rouwenhorst_kron_initial_distribution(rho1, sigma1, ns1, rho2, sigma2, ns2):
    init = np.zeros(ns1*ns2)
    init[int(ns1*ns2/2)] = 1
    return init

import sympy

class MarkovSwitchingDSGE(DSGE):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        context_dict = {str(v): v for v in self.parameters + self['other_parameters']+self['observables']}
        ms = self['__data__']['calibration']['markov_switching']
        self['markov_switching'] = {}

        context_dict['log'] = sympy.log
        self['markov_switching']['shock'] = ms['shock']
        self['markov_switching']['observable'] = ms['observable']
        self['markov_switching']['construction'] = eval(str(ms['construction']), context_dict)
        self['markov_switching']['shock_scaling'] = ms['scaling']

        if type(ms['states'])==str:
            function = ms['states'].split('(')[0]
            context_dict[function] = sympy.Function(function)
            rouwenhorst_args = eval(ms['states'], context_dict).args

            for moment in ['conditional_mean','transition','initial_distribution']:
                self['markov_switching'][moment] = sympy.Function('_'.join([function, moment]))(*rouwenhorst_args)

            self['markov_switching']['conditional_variance'] = eval(self['__data__']['calibration']['covariances'][ms['shock']], context_dict)


        else:
            initial_distribution = sympy.Matrix(eval(str(ms['states']['initial_distribution']).replace("'", ''), context_dict))
            self['markov_switching']['initial_distribution'] = initial_distribution
            conditional_mean = sympy.Matrix(eval(str(ms['states']['conditional_mean']).replace("'", ''), context_dict))
            self['markov_switching']['conditional_mean'] = conditional_mean

            conditional_variance = sympy.Matrix(eval(str(ms['states']['conditional_variance']).replace("'", ''), context_dict))
            self['markov_switching']['conditional_variance'] = conditional_variance

            transition = sympy.Matrix(eval(str(ms['states']['transition']).replace("'", ''), context_dict), real=True)
            self['markov_switching']['transition'] = transition




    def compile_model(self):
        from sympy.utilities.lambdify import lambdify
        self.python_sims_matrices()

        context = dict([(p.name, p) for p in self.parameters])
        context['exp'] = sympy.exp
        context['log'] = sympy.log

        ss = {}
        for p in self['other_para']:
            ss[str(p)] = eval(str(self['para_func'][p.name]), context)
            context[str(p)] = ss[str(p)]


        psi = lambdify([self.parameters], [ss[str(px)] for px in self['other_para']], context)

        def add_para_func(f):
            def wrapped_f(px, *args, **kwargs):
                return f([px, psi(px)], *args, **kwargs)
            return wrapped_f

        GAM0 = self.GAM0
        GAM1 = self.GAM1
        PSI = self.PSI
        PPI = self.PPI

        ms = {}
        for func in ['conditional_mean', 'conditional_variance', 'transition', 'initial_distribution']:
            context = globals()
            context['ImmutableDenseMatrix'] = np.array
            ms[func] = lambdify([self.parameters+self['other_para']], self['markov_switching'][func],
                                context)
            ms[func] = add_para_func(ms[func])


        ms['shock'] = self['markov_switching']['shock']
        ms['observable'] = self['markov_switching']['observable']

        QQ = self.QQ
        DD = self.DD
        ZZ = self.ZZ
        HH = self.HH

        obs_lagged = [o(-1) for o in self['observables']]
        ms['construction'] = lambdify([self.parameters+self['other_para'], self['observables'], obs_lagged],
                                      self['markov_switching']['construction'])

        ms['construction'] = add_para_func(ms['construction'])
        ms['shock_scaling'] = self['markov_switching']['shock_scaling']

        if 'observables' not in self:
            self['observables'] = self['variables'].copy()
            self['obs_equations'] = dict(self['observables'], self['observables'])

        try:
            datafile = self['__data__']['estimation']['data']

            if type(datafile)==dict:
                startdate = datafile['start']
                datafile = datafile['file']
            else:
                startdate = 0


            with open(datafile, 'r') as df:
                data = df.read()
                delim_dict = {}

                if data.find(',') > 0:
                    delim_dict['delimiter'] = ','

                data = np.genfromtxt(datafile, missing_values='NaN', **delim_dict)
        except:
            data = np.nan * np.ones((100, len(self['observables'])))
            startdate = 0
        import pandas as p
        if len(self['observables']) > 1:
            data = p.DataFrame(data[:, :len(self['observables'])], columns=list(map(lambda x: str(x), self['observables'])))
        else:
            data = p.DataFrame(data, columns=list(map(lambda x: str(x), self['observables'])))

        if startdate is not 0:
            nobs = data.shape[0]
            data.index = p.period_range(startdate, freq='Q', periods=nobs)

        prior = None
        if 'prior' in self['__data__']['estimation']:
            prior_type = ['beta', 'gamma', 'normal', 'inv_gamma', 'uniform', 'fixed']
            prior = []
            for par in self.parameters:
                prior_spec = self['__data__']['estimation']['prior'][par.name]

                ptype = prior_spec[0]
                pmean = prior_spec[1]
                pstdd = prior_spec[2]
                from scipy.stats import beta, norm, uniform, gamma
                from dsge.OtherPriors import InvGamma
                if ptype=='beta':
                    a = (1-pmean)*pmean**2/pstdd**2 - pmean
                    b = a*(1/pmean - 1)
                    pr = beta(a, b)
                    pr.name = 'beta'
                    prior.append(pr)
                if ptype=='gamma':
                    b = pstdd**2/pmean
                    a = pmean/b
                    pr = gamma(a, scale=b)
                    pr.name = 'gamma'
                    prior.append(pr)
                if ptype=='normal':
                    a = pmean
                    b = pstdd
                    pr = norm(loc=a, scale=b)
                    pr.name='norm'
                    prior.append(pr)
                if ptype=='inv_gamma':
                    a = pmean
                    b = pstdd
                    prior.append(InvGamma(a, b))
                if ptype=='uniform':
                    a, b = pmean, pstdd
                    pr = uniform(loc=a, scale=(b-a))
                    pr.name = 'uniform'
                    prior.append(pr)


        dsge = MarkovSwitchingStateSpaceModel(data, GAM0, GAM1, PSI, PPI,
                                              QQ, DD, ZZ, HH, t0=0,
                                              shock_names=list(map(str, self.shocks)),
                                              state_names=list(map(str, self.variables+self['fvars'])),
                                              obs_names=list(map(str, self['observables'])),
                                              prior=pri(prior), **ms)
        return dsge

# fsms = MarkovSwitchingDSGE.read('fs2005.yaml') 
# fsmslin = fsms.compile_model()

# p0 = np.array([[0.4],
#                [3.0],
#                [1.0],
#                [2.0],
#                [0.6],
#                [0.5],
#                [1.5],
#                [0.9],
#                [0.3],
#                [0.7],
#                [1.2],
#                [0.3],
#                [0.6],
#                [0.6],
#                [0.85],
#                [0.85]])


# fsmslin.log_lik(p0)
