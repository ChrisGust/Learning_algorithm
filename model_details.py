import numpy as np
import polydef as po

import importlib
importlib.reload(po)

#############################################################
#Initialize Polynomial Part Dependent on Model
##############################################################

def get_quadstates(poly,params):
    #returns exogenous states at each quadrature point and each exogenous grid point

    ninnov = poly['ninnov']
    nqs = poly['nqs']
    ns = poly['ns']
    npoly = poly['npoly']
    ne = poly['nexog_fe']
    quadgrid,quadweight = po.get_quadgrid(poly['nquad'],ninnov,nqs)
    sfut = np.zeros([ns,nqs,ninnov])
    ind_sfut = np.zeros([ns,nqs,ne],dtype=int)

    for i in np.arange(ns):
       for j in np.arange(nqs):
           sfut[i,j,:ne] = poly['rhos'][:ne]*poly['exoggrid'][i,:]+poly['stds'][:ne]*quadgrid[j,:ne]
           sfut[i,j,ne] = params['stdm']*quadgrid[j,ne]      
           ss = sfut[i,j,:ne]
           ind_sfut[i,j,:] = po.get_index(ss,ne,poly['ngrid'],poly['steps'],poly['bounds'])
    return(sfut,ind_sfut,quadweight)

def get_griddetails(polyapp,params):
    #Return details of polynomial associated with constructing the grids used

    nmsv = polyapp['nmsv']
    nmsve = polyapp['nexog_nmsv']
    nsreg = polyapp['nsreg']
    npoly = polyapp['npoly']
    nx = nmsv-nmsve-(nsreg-1)
    
    #grid for polynomial approximation
    polyapp['msvbounds'] = np.zeros([2*nmsv])
    if (nx >= 1):
        polyapp['msvbounds'][0] = -0.05
        polyapp['msvbounds'][nmsv] = 0.05
    if (nx >= 2):
            polyapp['msvbounds'][1] = -0.075
            polyapp['msvbounds'][nmsv+1] = 0.075
    if (nx == 3):
            polyapp['msvbounds'][2] = -0.02
            polyapp['msvbounds'][nmsv+2] = 0.02
    
    ubd = np.log(0.96)
    for i in np.arange(nsreg-1):
        polyapp['msvbounds'][nx+i] = np.log(0.0001)
        polyapp['msvbounds'][nmsv+nx+i] = ubd
        bd = polyapp['maxstd'] 
    polyapp['msvbounds'][nmsv-1] = -bd*params['stdm']
    polyapp['msvbounds'][nmsv+nmsv-1] = bd*params['stdm']

    polyapp['scmsv2xx'] = np.zeros([2*nmsv])
    polyapp['scxx2msv'] = np.zeros([2*nmsv])

    polyapp['scmsv2xx'][0:nmsv] = 2.0/(polyapp['msvbounds'][nmsv:2*nmsv]-polyapp['msvbounds'][0:nmsv])
    polyapp['scmsv2xx'][nmsv:2*nmsv] = -2.0*polyapp['msvbounds'][0:nmsv]/(polyapp['msvbounds'][nmsv:2*nmsv]-polyapp['msvbounds'][0:nmsv])-1.0
    polyapp['scxx2msv'][0:nmsv] = 0.5*(polyapp['msvbounds'][nmsv:2*nmsv]-polyapp['msvbounds'][0:nmsv])
    polyapp['scxx2msv'][nmsv:2*nmsv] = polyapp['msvbounds'][0:nmsv] + 0.5*(polyapp['msvbounds'][nmsv:2*nmsv]-polyapp['msvbounds'][0:nmsv])

    #for the approximating function.
    polyapp['rhos'][0] = params['rhoeta']
    polyapp['stds'][0] = params['stdeta']
    polyapp['exoggrid'],polyapp['exogindex'],polyapp['steps'],polyapp['bounds'],polyapp['ind2poly'] = po.get_exoggrid(polyapp['ngrid'],polyapp['nexog_fe'], \
        polyapp['ns'],polyapp['rhos'],polyapp['stds'],polyapp['maxstd'])
    polyapp['quadgrid'],polyapp['quadweight'] = po.get_quadgrid(polyapp['nquad'],polyapp['ninnov'],polyapp['nqs'])
    polyapp['sfutquad'],polyapp['ind_sfutquad'],polyapp['quadweights'] = get_quadstates(polyapp,params)

    #print('sfut = ', polyapp['sfutquad'][:,1,10])
    #sys.exit()
    
    return(polyapp)

# #############################################################
# #Functions for solving the model.
# #############################################################

def modelvariables(polycur,xtilm1,shocks,params,nmsv,nmsve,nsreg,gamma0,Pmat):
    #return model variables given expected output and inflation.
    #from sys import exit

    
    nx = nmsv-nmsve-(nsreg-1)
    #reduced form parameters
    kappap = (params['epp']-1.)/params['phip']
    nrss = np.log((1.-params['eta'])*params['dpss']/params['beta'])
    nu = params['gamma']/(1.-params['gamma'])
    theta = 1.-params['eta']
    omega = 1./(1.+params['beta']*(1-params['ap']))        

    nrm1 = 0.
    yym1 = 0.
    dpm1 = 0.
    if (nx >= 1):
        nrm1 = xtilm1[0]   
    if (nx >= 2):
        yym1 = xtilm1[1]
    if (nx >= 3):
        dpm1 = xtilm1[2]      
    
    expyy = polycur[0]
    expdp = polycur[1]
    biga_temp = 1.0+nu*(1.0+theta)+theta*(1.0-params['rhor'])*(params['gammax']+params['gammapi']*kappap*omega*(1.0+nu))
    zetartilde = params['rhor']*nrm1+(1.0-params['rhor'])*params['gammapi']*((1.0-params['ap'])*omega*dpm1+params['beta']*omega*polycur[1]-kappap*nu*omega*yym1)+shocks[1]
    yyhat = (nu*yym1+theta*(1.0+nu)*polycur[0]-theta*zetartilde+theta*polycur[1]-shocks[0])/biga_temp
    dphat = (1.0-params['ap'])*omega*dpm1+params['beta']*omega*polycur[1]+kappap*omega*(1.0+nu)*yyhat-kappap*nu*omega*yym1
    nomrhat = params['rhor']*nrm1 + (1.0-params['rhor'])*(params['gammapi']*dphat+params['gammax']*yyhat)+shocks[1]

    #update beliefs
    mid = np.int((nsreg+1)/2)
    prior = np.zeros(nsreg)
    prior[0:mid-1] = np.exp(xtilm1[nx:nx+mid-1])
    prior[mid:] = np.exp(xtilm1[nx+mid-1:nx+nsreg-1])
    temp_arr = np.append(prior[:mid-1],prior[mid:])
    prior[mid-1] = 1.0-np.sum(temp_arr)
    lik = np.zeros(nsreg)
    denominator = 0.0
    for i in np.arange(nsreg):
        lik[i] = np.exp( -0.5*(shocks[1]-gamma0[i])**2/params['stdm']**2 )
        denominator = denominator + lik[i]*prior[i]
    
    posterior = np.zeros(nsreg)
    if (np.abs(denominator) < 1e-08):
        posterior = prior
    else:
        for i in np.arange(nsreg):
            posterior[i] = prior[i]*lik[i]/denominator
    ptp_c_t = Pmat.dot(posterior)
    lptp = np.zeros(nsreg-1)
    for i in np.arange(mid-1):
        if (ptp_c_t[i] < 1e-08):
            lptp[i] = np.log(1e-08)
        else:
            lptp[i] = np.log(ptp_c_t[i])

        if (ptp_c_t[mid+i] < 1e-08):
            lptp[mid-1+i] = np.log(1e-08)
        else:
            lptp[mid-1+i] = np.log(ptp_c_t[mid+i])        
    return(yyhat,dphat,nomrhat,lptp,posterior)
                
def calc_euler(ind_state,gridindex,acoeff,polyapp,params):
    #returns predicted decision rules and prediction errors
    from sys import exit
    npoly = polyapp['npoly']
    nmsv = polyapp['nmsv']
    nmsve = polyapp['nexog_nmsv']
    ne = polyapp['nexog_fe']
    nsreg = polyapp['nsreg']
    nx = nmsv-nmsve-(nsreg-1)
            
    polycur = np.zeros([polyapp['nfunc']])
    polyvarm1 = polyapp['bbt'][gridindex,:]
    for ifunc in np.arange(polyapp['nfunc']):
        polycur[ifunc] = np.dot(acoeff[ind_state,ifunc*npoly:(ifunc+1)*npoly],polyvarm1)

    msvm1 = po.msv2xx(polyapp['pgrid'][gridindex,:],nmsv,polyapp['scxx2msv'])
    ss = np.append(polyapp['exoggrid'][ind_state,:],msvm1[nmsv-nmsve:])
    (yy,dp,nr,lptp,posterior) = modelvariables(polycur,msvm1[:nmsv-nmsve],ss,params,nmsv,nmsve,nsreg,polyapp['gamma0'],polyapp['P'])
    
    ind_futmat = polyapp['ind_sfutquad'][ind_state,:,:]  
    ss_futmat = polyapp['sfutquad'][ind_state,:,:]  

    exp_yyp = 0.
    exp_dpp = 0.
    msv = np.zeros(nmsv)
    if (nx >= 1):
        msv[0] = nr     
    if (nx >= 2):
        msv[1] = yy
    if (nx == 3):
        msv[2] = dp

    mid = np.int((nsreg+1)/2)
    msv[nx:nx+mid-1] = lptp[:mid-1]
    msv[nx+mid-1:nx+nsreg] = lptp[mid-1:]

    ptp_c_t = np.zeros(nsreg)
    ptp_c_t[0:mid-1] = np.exp(msv[nx:nx+mid-1])
    ptp_c_t[mid:] = np.exp(msv[nx+mid:nx+nsreg])

    temp_arr = np.append(ptp_c_t[:mid-1],ptp_c_t[mid:])
    ptp_c_t[mid-1] = 1.0-np.sum(temp_arr)
    for ireg in np.arange(nsreg):
        for j in np.arange(polyapp['nqs']):
            shockall = ss_futmat[j,:]
            shockall[ne] = polyapp['gamma0'][ireg]+ss_futmat[j,ne]
            msv[nmsv-nmsve] = shockall[ne]   
            xx1 = po.msv2xx(msv,nmsv,polyapp['scmsv2xx'])
            polyfut = po.get_linspline(xx1,ss_futmat[j,:ne],ind_futmat[j,:],acoeff,polyapp['exoggrid'],polyapp['steps'],polyapp['nfunc'],polyapp['ngrid'],npoly,nmsv,ne)
            (yyp,dpp,nrp,lptpp,postp) = modelvariables(polyfut,msv[:nmsv-nmsve],shockall,params,nmsv,nmsve,nsreg,polyapp['gamma0'],polyapp['P'])
            exp_yyp = exp_yyp + ptp_c_t[ireg]*polyapp['quadweight'][j]*yyp
            exp_dpp = exp_dpp + ptp_c_t[ireg]*polyapp['quadweight'][j]*dpp
                     
    polynew = np.zeros(polyapp['nfunc'])
    polynew[0] = exp_yyp
    polynew[1] = exp_dpp
        
    # print('difference in coeffs')
    # print(np.abs(polynew-polycur))
    # print('polycur = ', polycur)
    # print('-----------------')
    # print('ind_state = ', ind_state)
    # print('gridindex = ', gridindex)
    # print('msv = ', msv)
    # print('ptp_c_t = ', ptp_c_t)
    # print('polynew = ', polynew)
    # print('data = ', yy,dp,nr,posterior.round(3))
    # exit()
    res = np.sum(np.abs(polynew-polycur))/(polyapp['nfunc'])
    return(polynew,res)

def get_coeffs(acoeff0,params,poly,niter,stol,step):
    #iterate until convergence to find fixed point

    nfunc = poly['nfunc']
    npoly = poly['npoly']
    ns = poly['ns']
    nfunc = poly['nfunc']
            
    acoeff = acoeff0
    acoeffnew = np.zeros([ns,nfunc*npoly])    
    convergence = False

    for tt in np.arange(niter):
        avgerror = 0.
        for i in np.arange(ns):
            res_avg = 0.
            polyappnew = np.zeros([npoly,nfunc])
            for ip in np.arange(npoly):
                polyappnew[ip,:],res = calc_euler(i,ip,acoeff,poly,params)                
                res_avg = res_avg + res/npoly
            alphass = np.dot(poly['bbtinv'],polyappnew)
            # if (i < 1):
            #     print('i = ', i)
            #     print('polyappnew 0 = ', polyappnew[:,0])
            #     print('polyappnew 2 = ', polyappnew[:,2])
            #     #test = np.dot(poly['bbtinv'],polyappnew)
            #     #print('test = ', test[:,0])
            #     #print('alphass(:,0) = ', alphass[:,0])
            #     sys.exit()

            for ip in np.arange(npoly):
                for ifunc in np.arange(nfunc):
                    acoeffnew[i,ifunc*npoly+ip] = alphass[ip,ifunc]
            avgerror = avgerror + res_avg/ns
            

        #print(acoeffnew[:,nfunc*npoly:(nfunc*npoly+npoly)].round(4))
        #sys.exit()
        print(avgerror)
        
        if np.any(np.isnan(acoeffnew)):  #return acoeff
            break
        if (avgerror < stol):
            convergence = True
            break
        acoeff = (1.0-step)*acoeff + step*acoeffnew    
    return(acoeff,convergence)

def solve_model(acoeff0,params,poly):
#Inputs guess of metaparameters and returns metaparameters that solve the model.

    #update poly to get part that depends on model parameter values
    poly = get_griddetails(poly,params)
    niter = 1000
    stol = 1e-05
    step = 0.6
    acoeffstar,convergence = get_coeffs(acoeff0,params,poly,niter,stol,step)
    return(acoeffstar,convergence)

def decr(endogvarm1,innov,regime,acoeff,poly,params):
    #Decision rule.
    endogvar = np.zeros(poly['nvars'])
    ne = poly['nexog_fe']
    npoly = poly['npoly']
    nmsv = poly['nmsv']
    nmsve = poly['nexog_nmsv']
    nsreg = poly['nsreg']
    nx = nmsv-nmsve-(nsreg-1)
    ss = np.zeros(ne+nmsve)
    msvm1 = np.zeros(nmsv)
    shocks = np.zeros(ne+1)
    
    #update non-monetary shocks and compute place on grid
    endogvar[8] = params['rhoeta']*endogvarm1[8] + params['stdeta']*innov[0]
    ss[0] = endogvar[8]
    ind_ss = po.get_index(ss,ne,poly['ngrid'],poly['steps'],poly['bounds'])
    ss[1] = poly['gamma0'][regime]+params['stdm']*innov[1]
    endogvar[11] = ss[1]
    endogvar[10] = params['stdm']*innov[1]
    endogvar[9] = poly['gamma0'][regime]
    msvm1[:nx+nsreg-1] = endogvarm1[:nx+nsreg-1]
    msvm1[nx+nsreg-1] = ss[1]
    xxm1 = po.msv2xx(msvm1,nmsv,poly['scmsv2xx'])
    polycur = po.get_linspline(xxm1,ss[:ne],ind_ss,acoeff,poly['exoggrid'],poly['steps'],poly['nfunc'],poly['ngrid'],npoly,nmsv,ne)
    (yy,dp,nr,lptp,post) = modelvariables(polycur,msvm1[:nmsv-nmsve],ss,params,nmsv,nmsve,nsreg,poly['gamma0'],poly['P'])
    endogvar[0] = nr
    endogvar[1] = yy
    endogvar[2] = dp
    endogvar[nx:nx+nsreg-1] = lptp
    endogvar[nx+nsreg-1:nx+2*nsreg-1] = post
    return(endogvar)

#####################################
#Test code
###################################

params = {'beta': 0.99, 'eta': 0.0025, 'gamma': 0.0, 'epp': 6.0, 'phip': 100.0, 'dpss': 0.005, 'ap': 1.0, 'rhor' : 0.0, 'gammapi': 1.5, 'gammax': 0.25, 'stdm' : 0.0008, \
          'rhoeta' : 0.85, 'stdeta' : 0.003, 'rhogam': 0.9, 'stdgam' : 0.001}

nexog_fe = 1
nendog_nmsv = 3
nsreg = 3
ngrid = np.ones(nexog_fe,dtype=int)
ngrid[0] = 3
acoeff0,poly = po.initialize_poly(nexog_fe,nendog_nmsv,nsreg,ngrid) 
               
p_gam = (1.0+params['rhogam'])/2.0
poly['P'] = po.transmat(nsreg,p_gam,p_gam)
psi_gam = np.sqrt(nsreg-1)*params['stdgam']
poly['gamma0'] = np.linspace(-psi_gam,psi_gam,nsreg)


acoeff,convergence = solve_model(acoeff0,params,poly)

innov = np.zeros(poly['ninnov'])
endogvarm1 = np.zeros(poly['nvars'])
nx = poly['nmsv']-poly['nexog_nmsv']+poly['nsreg']-1
endogvarm1[nx:nx+nsreg-1] = np.log(0.001)
regime = 2
endogvar = decr(endogvarm1,innov,regime,acoeff,poly,params)


