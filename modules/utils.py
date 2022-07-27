import numpy as np
import scipy.stats
import scipy
import scipy.special
import sys
import os
import matplotlib.pyplot as plt
import numbers
# Generic IO
# ------------------------------------------------------------

def readPers(name, K=1):
    if not os.path.isfile(name):
        print("Not valid file")
        exit(0)
    [D, KMAX, TH, birth_all, death_all, inf_all] = np.load(name, allow_pickle=True)

    if K >= D:
        print("K value invalid:" + str(K))
        exit(0)

    return birth_all[K], death_all[K], len(inf_all[K])


def readPersAll(name, K=1):
    if not os.path.isfile(name):
        print("Not valid file")
        exit(0)
    [D, KMAX, TH, birth_all, death_all, inf_all] = np.load(name, allow_pickle=True)

    if K >= D:
        print("K value invalid:" + str(K))
        exit(0)

    return birth_all[K], death_all[K], inf_all[K]


def checkFile(name,K=1):
    a,b,_=readPers(name,K)
    return len(a)!=0


def getFileList(dirname):
    FLIST = []
    for root, dirs, files in os.walk(dirname):
        for f in files:
            if f.endswith('.npy'):
                FLIST.append(f)
    return FLIST


def getFileParams(name):
    tkns = name.split('_',maxsplit=6)
    if len(tkns) < 6:
        raise Exception()
    cmplx = tkns[0]
    gen = tkns[1]
    dim = int(tkns[2])
    kmax = int(tkns[3])
    numpts = int(tkns[4])
    id_num = int(tkns[5].split('.')[0])
    return cmplx,gen,dim,kmax,numpts,id_num

def makeFileName(cmplx,gen,dim,kmax,numpts,id_num):
    return cmplx+'_'+gen+'_'+str(dim)+"_"+str(kmax)+'_'+str(numpts)+'_'+str(id_num)+'.npy'

    
# sample at most N files from
# negative number or ommited means everything is returned 
def sampleFiles(dirname,N=-1,**kwargs):
    # all files
    prms = {}
    prms['cmplx'] = None
    prms['gen'] = None
    prms['dim'] = None
    prms['numpts'] = None
    prms['k'] = None
    prms['id'] = None
     
    for key, value in kwargs.items():
        if key not in prms.keys():
            print('Bad parameters')
            exit(0)
        prms[key] = value
        
    files = getFileList(dirname)
    filteredfiles = []
    for f in files:
        try:
            fprm = getFileParams(f)
        except:
            continue
        flag=True
        flag = False if ((prms['cmplx'] is not None) and (fprm[0]!=prms['cmplx'])) else True
        flag = False if ((prms['gen'] is not None) and (fprm[1]!=prms['gen'])) else flag 
        flag = False if ((prms['dim'] is not None) and (fprm[2]!=prms['dim'])) else flag 
        flag = False if ((prms['k'] is not None) and (fprm[3]<prms['k'])) else flag 
        flag = False if ((prms['numpts'] is not None) and (fprm[4]!=prms['numpts'])) else flag 
        flag = False if ((prms['id'] is not None) and (fprm[5]!=prms['id'])) else flag 
    
        if flag:
            if (prms['k'] is None):
                filteredfiles.append(f)  
            else:
                if(checkFile(dirname+'/'+f,prms['k'])):
                    filteredfiles.append(f)  
     
# something for the dataframes bit
    
    if (len(filteredfiles)<=N) or (N<0):
        return filteredfiles, prms   
    
    numpy.random.shuffle(filteredfiles)
    return filteredfiles[:N], prms   


# sample at most N files from
# negative number or ommited means everything is returned 
def filterFiles_old(dirname,**kwargs):
    # all files
    prms = {}
    prms['cmplx'] = None
    prms['gen'] = None
    prms['dim'] = None
    prms['numpts'] = None
    prms['k'] = None
    prms['id'] = None
     
    for key, value in kwargs.items():
        if key not in prms.keys():
            print('Bad parameters')
            exit(0)
        prms[key] = value
        
    files = getFileList(dirname)
    filteredfiles = []
    for f in files:
        try:
            fprm = getFileParams(f)
        except:
            continue
        flag = True
        flag = False if ((prms['cmplx'] is not None) and (fprm[0]!=prms['cmplx'])) else True
        flag = False if ((prms['gen'] is not None) and (fprm[1]!=prms['gen'])) else flag 
        flag = False if ((prms['dim'] is not None) and (fprm[2]!=prms['dim'])) else flag
        flag = False if ((prms['k'] is not None) and (fprm[3]<prms['k'])) else flag
        flag = False if ((prms['numpts'] is not None) and (fprm[4]!=prms['numpts'])) else flag 
        flag = False if ((prms['id'] is not None) and (fprm[5]!=prms['id'])) else flag 
        if flag:
            if (prms['k'] is None):
                filteredfiles.append(f)  
            else:
                # the below checks if we have gone all the way
                if(checkFile(dirname+'/'+f,prms['k'])):
                    filteredfiles.append(f)  
     
    # removed randomness
    return filteredfiles, prms


def filterFiles(dirname, **kwargs):
    # all files
    prms = {}
    prms['cmplx'] = None
    prms['gen'] = None
    prms['dim'] = None
    prms['numpts'] = None
    prms['k'] = None
    prms['id'] = None

    for key, value in kwargs.items():
        if key not in prms.keys():
            print('Bad parameters')
            exit(0)
        prms[key] = value

    files = getFileList(dirname)
    filteredfiles = []
    for f in files:
        try:
            fprm = getFileParams(f)
        except:
            continue
        flag = True
        flag = False if ((prms['cmplx'] is not None) and (fprm[0] != prms['cmplx'])) else True
        flag = False if ((prms['gen'] is not None) and (fprm[1] != prms['gen'])) else flag
        flag = False if ((prms['dim'] is not None) and (fprm[2] != prms['dim'])) else flag
        flag = False if ((prms['k'] is not None) and (fprm[3] < prms['k'])) else flag
        flag = False if ((prms['numpts'] is not None) and (fprm[4] != prms['numpts'])) else flag
        flag = False if ((prms['id'] is not None) and (fprm[5] != prms['id'])) else flag
        if flag:
            if (prms['k'] is None):
                 filteredfiles.append(f)
            else:
                # the below checks if we have gone all the way
                if (checkFile(dirname + '/' + f, prms['k'])):
                    filteredfiles.append(f)

                    # removed randomness
    return filteredfiles, prms


# Stats IO
# ------------------------------------------------------------
# do the fit - no processing assumed
# def fitParams(pers,DIST,*args,**kwargs):
#     params = DIST.fit(pall,*args,**kwargs)
#     return params

def makeDist(distname):
    l = locals()
    exec('DIST = scipy.stats.%s' % distname,globals(),l)
    DIST=l['DIST'] 
    return DIST

def fitParams(pers,dist,*args,**kwargs):
    params = dist.fit(pers,*args,**kwargs)
    return params

def getCDF(pers):
    pers.sort()
    cdf = np.array(range(0, len(pers))) / len(pers)
    return pers, cdf

def getCDF_uniq(vals):
    vals.sort()
    uvals,counts = np.unique(vals, return_counts=True)
    # cdf = numpy.array(range(0, len(pers))) / len(pers)
    cdf = np.cumsum(counts)/len(vals)
    return uvals, cdf

def realCDF(rv,mn,mx,nbins=1000):
    hx_est = numpy.linspace(mn,mx,nbins)
    cdf_est = rv.cdf(hx_est)    
    return hx_est,cdf_est

# def getPValue(pers,rv):
#     pval = scipy.stats.ks_1samp(pall, rv.cdf)
#     return pval[1]

def getPValue(pers,rv):
     pval = scipy.stats.ks_1samp(pers, rv.cdf)
     #print(pval)
     return pval[1]


# output functions



def qqplot(x, y, quantiles=None, interpolation='nearest', ax=None, rug=False,
           rug_length=0.05, rug_kwargs=None, **kwargs):
    """Draw a quantile-quantile plot for `x` versus `y`.

    Parameters
    ----------
    x, y : array-like
        One-dimensional numeric arrays.

    ax : matplotlib.axes.Axes, optional
        Axes on which to plot. If not provided, the current axes will be used.

    quantiles : int or array-like, optional
        Quantiles to include in the plot. This can be an array of quantiles, in
        which case only the specified quantiles of `x` and `y` will be plotted.
        If this is an int `n`, then the quantiles will be `n` evenly spaced
        points between 0 and 1. If this is None, then `min(len(x), len(y))`
        evenly spaced quantiles between 0 and 1 will be computed.

    interpolation : {‘linear’, ‘lower’, ‘higher’, ‘midpoint’, ‘nearest’}
        Specify the interpolation method used to find quantiles when `quantiles`
        is an int or None. See the documentation for numpy.quantile().

    rug : bool, optional
        If True, draw a rug plot representing both samples on the horizontal and
        vertical axes. If False, no rug plot is drawn.

    rug_length : float in [0, 1], optional
        Specifies the length of the rug plot lines as a fraction of the total
        vertical or horizontal length.

    rug_kwargs : dict of keyword arguments
        Keyword arguments to pass to matplotlib.axes.Axes.axvline() and
        matplotlib.axes.Axes.axhline() when drawing rug plots.

    kwargs : dict of keyword arguments
        Keyword arguments to pass to matplotlib.axes.Axes.scatter() when drawing
        the q-q plot.
    """
    # Get current axes if none are provided
    if ax is None:
        ax = plt.gca()
    if quantiles is None:
        quantiles = min(len(x), len(y))

    # Compute quantiles of the two samples
    if isinstance(quantiles, numbers.Integral):
        quantiles = numpy.linspace(start=0, stop=1, num=int(quantiles))
    else:
        quantiles = numpy.atleast_1d(np.sort(quantiles))
     
    x_quantiles = numpy.quantile(x, quantiles, interpolation=interpolation)
    y_quantiles = numpy.quantile(y, quantiles, interpolation=interpolation)

    # Draw the rug plots if requested
    if rug:
        # Default rug plot settings
        rug_x_params = dict(ymin=0, ymax=rug_length, c='gray', alpha=0.5)
        rug_y_params = dict(xmin=0, xmax=rug_length, c='gray', alpha=0.5)

        # Override default setting by any user-specified settings
        if rug_kwargs is not None:
            rug_x_params.update(rug_kwargs)
            rug_y_params.update(rug_kwargs)

        # Draw the rug plots
        for point in x:
            ax.axvline(point, **rug_x_params)
        for point in y:
            ax.axhline(point, **rug_y_params)

    # Draw the q-q plot
    ax.scatter(x_quantiles, y_quantiles, **kwargs)
    #ax.plot(x_quantiles, y_quantiles, **kwargs)


#########################
def gen_figure8(R1, R2, W, D, N):

    P = np.zeros((2,N))
    c = 0
    R1s = R1**2
    R1Ws = (R1 + W)**2
    R2s = R2**2
    R2Ws = (R2 + W)**2

    MX = np.max((R1, R2))
    while c < N:
        a = np.random.uniform(-2 * R2 - 3 * W / 2, 2 * R1 + 3 * W / 2)
        b = np.random.uniform(-MX - W, MX + W)
        r = (a + R2 + W / 2) ** 2 + b ** 2;
        if ((b < D) and (b > -D) and (a < R1 + W / 2) and (a > -R2 - W / 2)):
            continue

        if (r >= R2s and r < R2Ws):
            P[0,c] = a
            P[1,c] = b
            c += 1
            continue

        r = (a - R1 - W / 2) ** 2 + b ** 2
        if (r >= R1s and r < R1Ws):
            P[0,c] = a
            P[1,c] = b
            c += 1
    return P