import sys
import numpy as np

######################################################################
#Functions used to construct polynomial\spline approximation.
####################################################################

def smolyakpoly(xx,nmsv,npoly):
    #First order smolyak polynomial
    poly = np.zeros([npoly])
    poly[0] = 1.
    for i in np.arange(nmsv):
        poly[2*i+1] = xx[i]
        poly[2*i+2] = 2.0*xx[i]**2-1.0
    return(poly)
 
def sparsegrid(nmsv,npoly):
    #Return smolyak grid and associated matrices.
    #form smolyak grid
    xgrid = np.zeros([npoly,nmsv])
    for i in np.arange(nmsv):
        xgrid[2*i+1,i] = -1.0
        xgrid[2*i+2,i] = 1.0    
    #form bbt matrix 
    bbt = np.zeros([npoly,npoly])
    for i in np.arange(npoly):
        bbt[i,:] = smolyakpoly(xgrid[i,:],nmsv,npoly)
    bbtinv = np.linalg.inv(bbt)
    return(xgrid,bbt,bbtinv)

def finite_grid(n,rho,std,maxgridstd):
    #Returns xgrid [1Xn] equally spaced points; steps: spacing between points; xbound - lower and upper bound
    #n: number of points; rho: AR(1) coeffficient; std: Standard deviation of innovation; maxgridstd: determines size of grid
    nu = np.sqrt(1./(1.-rho**2))*maxgridstd*std
    xgrid,xstep = np.linspace(-nu,nu,num=n,retstep=True)
    xbound = np.array([xgrid[0],xgrid[n-1]])
    return(xgrid,xstep,xbound)

def kronindex(ngrid,nvar,ns):
    #Returns exoggridindex (nsXnvar) matrix of ns grid points each of length nvar.   
    #ngrid is 1Xnvar of individual indices
    #ns = ngrid[0]*...*ngrid[nvar-1] 
    exoggridindex = np.zeros([ns,nvar],dtype=int)
    for ie in np.arange(nvar-1,-1,-1):
        if (ie == nvar-1):
            blocksize = 1
        else:
            blocksize = blocksize*ngrid[ie+1]
        ncall = int(ns/(blocksize*ngrid[ie]))
        for ic in np.arange(ncall):
            for ib in np.arange(ngrid[ie]):
                exoggridindex[ngrid[ie]*blocksize*ic+blocksize*ib:ngrid[ie]*blocksize*ic+blocksize*(ib+1),ie] = ib
    return(exoggridindex)

def get_exoggrid(ngrid,nvar,ns,rhos,stds,maxgridstd):
    #Return exoggrid (nsXnvar) matrix of values of exogenous shocks used for finite element part of approximating functions;
    #Also returns exogindex the (nsXnvar) associated matrix of shock indices; steps (1Xnvar) vector of distances between grid points;
    #sbound 1X2*nvar vector of bounds for each shock.
    exogindex = kronindex(ngrid,nvar,ns)
    
    sgrid = np.zeros([np.max(ngrid),nvar])
    steps = np.zeros(nvar)
    sbound = np.zeros([nvar,2])
    ind2poly = np.zeros(ns,dtype=int)
    for i in np.arange(nvar):
        sgrid[:ngrid[i],i],steps[i],sbound[i,:] = finite_grid(ngrid[i],rhos[i],stds[i],maxgridstd)

    exoggrid = np.zeros([ns,nvar])
    for i in np.arange(ns):
        indvec = exogindex[i,:]
        #if (indvec[0] == ngrid[0]-1):
        ind2poly[i] = 1
        for j in np.arange(nvar):
            exoggrid[i,j] = sgrid[indvec[j],j]
    return(exoggrid,exogindex,steps,sbound,ind2poly)

def get_quadsingle(nqs_single):
    #Returns qhnodes and qhweights, 1Xnqs_single vectors, of quadrature nodes and weights.
    #nqs_single must equal 2 or 3.
    if (nqs_single == 2):
        qhnodes = np.sqrt(2.)*np.array([-np.sqrt(2.)/2., np.sqrt(2.)/2.])
        qhweights = np.sqrt(1./np.pi)*np.array([np.sqrt(np.pi)/2.,np.sqrt(np.pi)/2.])
    elif (nqs_single == 3):
        qhnodes = np.sqrt(2.)*np.array([-np.sqrt(6.)/2., 0., np.sqrt(6.)/2.])
        qhweights = np.sqrt(1./np.pi)*np.array([np.sqrt(np.pi)/6., 2.*np.sqrt(np.pi)/3., np.sqrt(np.pi)/6.])
    else:
        print('You can only have 2 or 3 quadrature points per shock.')
        sys.exit()
    return(qhnodes,qhweights)

def get_quadgrid(nquad,nshocks,nqs):
    #Uses tensor product to construct quadgrid (nqsXnshocks) matrix of points used to evaluate conditional expectations.
    #quadweight is 1Xnqs of weights used to weight quadrature points. Should sum to 1.
    #nquad is 1Xnshocks vector of the number of quadrature points for each shock.
    #nqs = nquad[0]*nquad[1]*...*nquad[nshocks-1] is the total number of quadrature points.

    quadindex = kronindex(nquad,nshocks,nqs)
    ghnodes = np.zeros([np.max(nquad),nshocks])
    ghweights = np.zeros([np.max(nquad),nshocks])
    for i in np.arange(nshocks):
        ghnodes[:nquad[i],i],ghweights[:nquad[i],i] = get_quadsingle(nquad[i])

    quadgrid = np.zeros([nqs,nshocks])
    quadweight = np.ones(nqs)
    for i in np.arange(nqs):
        indvec = quadindex[i,:]
        for j in np.arange(nshocks):
            quadgrid[i,j] = ghnodes[indvec[j],j]
            quadweight[i] = quadweight[i]*ghweights[indvec[j],j]
    return(quadgrid,quadweight)

def get_coeffind(ind_ss,ngrid,nvar):
    #Given ind_ss, 1xnvar, value of shock values return position on exogenous grid (ind_coeff).
    #ind_coeff must be integer between 0 and ns-1.
    if (nvar == 1):
        ind_coeff = ind_ss[0]
    elif (nvar == 2):
        ind_coeff = ngrid[1]*ind_ss[0]+ind_ss[1]
    else:
        print('There can only be 2 shocks on the finite element part of the solution.')
        sys.exit()
    return(ind_coeff)
    
def get_index(ss,nvar,ngrid,steps,bounds):
    #Given ss (1xnvar) value of shock values, return (1xnvar) locations of shocks on the grid.
    ind_ss = np.zeros(nvar,dtype=int)
    for i in np.arange(nvar):
        if (ss[i] < bounds[i,0]):
            ind_ss[i] = 0
        elif (ss[i] > bounds[i,1]):
            ind_ss[i] = ngrid[i]-2
        else:
            ind_ss[i] = np.floor((ss[i]-bounds[i,0])/steps[i])  
    return(ind_ss)

def msv2xx(msv,nmsv,slopeconmsv):
  #Maps between xx variable between [-1,1] and grid for msvbounds
  return(slopeconmsv[0:nmsv]*msv[0:nmsv] + slopeconmsv[nmsv:2*nmsv])

def get_linspline(xx,ss,ind_ss,acoeff,exoggrid,steps,nfunc,ngrid,npoly,nmsv,ne):
    #returns decision rule
    linspline = np.zeros([nfunc])
    y1 = np.zeros([nfunc])
    y2 = np.zeros([nfunc])
    
    poly_xx = smolyakpoly(xx,nmsv,npoly)
    ind_state = get_coeffind(ind_ss,ngrid,ne)
    if ne == 1:
        tt = (ss[0]-exoggrid[ind_state,0])/steps[0]
        for ifunc in np.arange(nfunc):
            y1[ifunc] = np.dot(acoeff[ind_state,ifunc*npoly:(ifunc+1)*npoly],poly_xx)
            y2[ifunc] = np.dot(acoeff[ind_state+1,ifunc*npoly:(ifunc+1)*npoly],poly_xx)
        linspline = (1.-tt)*y1+tt*y2
    elif ne == 2:
        y3 = np.zeros([nfunc])
        y4 = np.zeros([nfunc])
        ind2 = ind_ss+np.array([1,0],dtype=int)
        ind_state2 = get_coeffind(ind2,ngrid,ne)
        ind3 = ind_ss+np.array([0,1],dtype=int)
        ind_state3 = get_coeffind(ind3,ngrid,ne)
        ind4 = ind_ss+np.array([1,1],dtype=int)
        ind_state4 = get_coeffind(ind4,ngrid,ne)
        tt = (ss[0]-exoggrid[ind_state,0])/steps[0]
        uu = (ss[1]-exoggrid[ind_state,1])/steps[1]
        for ifunc in np.arange(nfunc):
            y1[ifunc] = np.dot(acoeff[ind_state,ifunc*npoly:(ifunc+1)*npoly],poly_xx)
            y2[ifunc] = np.dot(acoeff[ind_state2,ifunc*npoly:(ifunc+1)*npoly],poly_xx)
            y3[ifunc] = np.dot(acoeff[ind_state3,ifunc*npoly:(ifunc+1)*npoly],poly_xx)
            y4[ifunc] = np.dot(acoeff[ind_state4,ifunc*npoly:(ifunc+1)*npoly],poly_xx)         
        linspline = (1.-tt)*(1.-uu)*y1+tt*(1.-uu)*y2+(1.-tt)*uu*y3+tt*uu*y4
        # if (np.abs(acoeff[0,0]) > 0.):
        #     print('ind_state2 = ', ind_state2)
        #     print('ind2 = ', ind2)
        #     print('acoeff[ind_state2,:] = ', acoeff[ind_state2,0:npoly])
        #     print('tt = ', tt)
        #     print('uu = ', uu)
        #     print('ind_state = ', ind_state)
        #     print('y1[0] = ', y1[0])
        #     print('y2[0] = ', y2[0])
        #     print('y3[0] = ', y3[0])
        #     print('y4[0] = ', y4[0])
        #     print('acoeff[ind_state,:] = ', acoeff[ind_state,0:npoly])
        #     print('linspline = ', linspline)
        #     sys.exit()
    else:
        fprintf('ne must be less than 2')
        sys.exit()
    return(linspline)

def transmat(n,p,q):
    if (n <= 1):
        sys.exit('n needs to be bigger than 1 (transmat)')
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

def initialize_poly(nexog_fe,nendog_nmsv,nsreg,ngrid):
#Returns some details of polynomial approximation (the ones that do not depend on model parameters).

    from sys import exit

    polyapp = {'nexog_fe': nexog_fe, 'nendog_nmsv': nendog_nmsv, 'nfunc': 2, 'maxstd': 2.5, 'nsreg': nsreg, 'polyweight': 10000.0}
    
    polyapp['nexog_nmsv'] = 1
    polyapp['nmsv'] = nendog_nmsv+polyapp['nsreg']-1+polyapp['nexog_nmsv']
    polyapp['npoly'] = 2*polyapp['nmsv']+1
    
    polyapp['ninnov'] = polyapp['nexog_fe']+polyapp['nexog_nmsv']
    if polyapp['nexog_fe'] != 1:
        exit('There must be one non-monetary shock (initialize_poly)')
    polyapp['ngrid'] = ngrid
    polyapp['nquad'] = np.ones(polyapp['ninnov'],dtype=int)
    polyapp['rhos'] = np.zeros(polyapp['nexog_fe'])
    polyapp['stds'] = np.zeros(polyapp['nexog_fe'])
    polyapp['nquad'][0] = 3
    polyapp['nquad'][1] = 2
    polyapp['ns'] = np.prod(polyapp['ngrid'])
    polyapp['nqs'] = np.prod(polyapp['nquad'])
    polyapp['pgrid'],polyapp['bbt'],polyapp['bbtinv'] = sparsegrid(polyapp['nmsv'],polyapp['npoly'])
    acoeff0 = np.zeros([polyapp['ns'],polyapp['nfunc']*polyapp['npoly']])
    if (nsreg == 3):
        polyapp['varlist'] = ['nr','yy','dp','lp1m1','lp3m1','p1t','p2t','p3t','eta','gamma0','epm','gamma0+epm']
        polyapp['nvars'] = len(polyapp['varlist'])
    else:
        polyapp['varlist'] = ['nr','yy','dp','lp1m1','p1t','p2t','eta','gamma0','epm','gamma0+epm','Eyy','Edp']
        polyapp['nvars'] = len(polyapp['varlist'])
    return(acoeff0,polyapp)

