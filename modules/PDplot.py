import utils
import scipy.stats as st
import scipy.special as sp

import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
import seaborn as sns

ALPHA = 1
RIPS = 2
TNAMES = {ALPHA:'ALPHA', RIPS:'RIPS'}
TNMAES_LABELS = {ALPHA:'Čech', RIPS:'Rips'}
PCHAR = '\pi'
# For ATMCS
# PCHAR = 'p'
AVALS = {ALPHA: 0.5, RIPS: 1}
COLS = sns.color_palette()
COLS_DICT = {(2,1):0,
    (3,1):1,
    (4,1):2,
    (5,1):3,
    (3,2):4,
    (4,2):5,
    (5,2):6,
    (4,3):7,
    (5,3):8,
    (5,4):9}

DEF_TH = 30
DEF_DELTA = 1e-3


def getSample(dirname, prms, delta=DEF_DELTA, th=DEF_TH, log_pd=True, load_all=False):
    T, G, D, K = prms
    lbl = ' / '.join(map(str, [TNMAES_LABELS[T], G, D, K]))
    F = utils.filterFiles(dirname, cmplx=TNAMES[T], gen=G, dim=D)
    F = F[0]


    # We can either load all files matching the parameters, or only the first one.
    if load_all==False:
        F = [F[0]]

    # Collecting all birth and death values from the files.
    dall = []
    ball = []
    for i,f in enumerate(F):
        fname = dirname + '/' + f
        bnew, dnew, i = utils.readPers(fname, K)
        ball += [b for b in bnew]
        dall += [d for d in dnew]

    ball = np.array(ball)
    dall = np.array(dall)

    # A patch fix for alpha-projective only.
    if T == ALPHA and G == 'projective':
            ball = ball**2
            dall = dall**2

    pall = dall / ball
    IDX = np.where((pall > 1 + delta) & (pall < th))
    ball = ball[IDX]
    dall = dall[IDX]
    pall = pall[IDX]

    if log_pd:
        ball = np.log(ball)
        dall = np.log(dall)
        pall = np.log(pall)

    print(lbl, len(ball))

    return ball, dall, pall, lbl


def plotPDS(rootname, plist, log_pd=True, num_points=0, load_all=False, th=DEF_TH, delta=DEF_DELTA, pdxl=None, fnsize_axes=24, fnsize_title = 30, psize=2):
    NC = len(plist)
    NR = 2

    if log_pd:
        xlabel = '$\log(\mathrm{birth})$'
        ylabel = '$\log(\mathrm{death})$'
        pylabel = '$\log(\mathrm{death})  -  \log(\mathrm{birth})$'
    else:
        xlabel = 'birth'
        ylabel = 'death'
        pylabel = 'death / birth'

    for i, prms in enumerate(plist):

        ball, dall, pall, lbl = getSample(rootname, prms, delta, th, log_pd)

        ax = plt.subplot(NR, NC, i+1)

        if num_points > 0:
            IDX = np.random.choice(np.arange(len(pall)), num_points, replace=False)
            sball = ball[IDX]
            sdall = dall[IDX]
        else:
            sball = ball
            sdall = dall

        # Draw persistence diagram
        ax.scatter(sball, sdall,s=psize, color=COLS[i%10])
        if pdxl!=None:
            ax.set_xlim(pdxl[i])
            ax.set_ylim(pdxl[i])
        x = ax.get_xlim()

        ax.set_xlabel(xlabel, fontsize=fnsize_axes)
        if (i==0):
            ax.set_ylabel(ylabel, fontsize=fnsize_axes)

        ax.plot(x,x,'k')
        # ax.axis('equal')
        ax.set_title(lbl, fontsize=fnsize_title)

        if load_all:
            ball, dall, pall, lbl  = getSample(rootname, prms, delta, th, log_pd, load_all=True)

        ax = plt.subplot(NR,NC,i+1 + NC)
        H, yedges, xedges = np.histogram2d(pall, ball, bins=100)
        ax.pcolormesh(xedges, yedges, np.log(H+1), cmap='turbo')
        ax.set_xlabel(xlabel, fontsize=fnsize_axes)
        if (i==0):
            ax.set_ylabel(pylabel, fontsize=fnsize_axes)

    # plt.subplots_adjust(hspace=0.5)
    plt.tight_layout()


def plotCDFS(rootname, plist, load_all=False, num_points = 0, th=DEF_TH, delta=DEF_DELTA, xl = None , pdxl=None, fnsize_axes=24, fnsize_title = 30, fnsize_legend = 14,psize=2, lnwidth=2):
    NC = len(plist)
    NR = 3

    xlabel = None
    ylabel = None
    pxlabel = '$\log('+ PCHAR + ')$'

    for i, prms in enumerate(plist):

        ball, dall, pall, lbl = getSample(rootname, prms, delta, th, load_all)

        if num_points > 0:
            IDX = np.random.choice(np.arange(len(pall)), num_points, replace=False)
            sball = ball[IDX]
            sdall = dall[IDX]
        else:
            sball = ball
            sdall = dall

        ax = plt.subplot(NR, NC, i+1)
        ax.scatter(sball, sdall, s=psize, color=COLS[i%10], label=lbl)
        if pdxl != None:
            ax.set_xlim(pdxl[i])
            ax.set_ylim(pdxl[i])

        x = ax.get_xlim()
        ax.set_xlabel(xlabel, fontsize=fnsize_axes)
        if (i==0):
            ax.set_ylabel(ylabel, fontsize=fnsize_axes)
        else:
            ax.set_ylabel(None)

        ax.plot(x,x,'k')

        ax.set_title(lbl, fontsize=fnsize_title)

        ax = plt.subplot(NR,NC,i+1 + NC)
        sns.ecdfplot(pall, ax=ax, color=COLS[i%10], linewidth=lnwidth)
        ax.set_ylim((0,1.05))
        if xl:
            ax.set_xlim(xl)
        ax.set_xlabel(pxlabel, fontsize=fnsize_axes)
        if (i==0):
            ax.set_ylabel('CDF', fontsize=fnsize_axes)
        else:
            ax.set_ylabel(None)
        ax = plt.subplot(NR, NC, 8)
        sns.ecdfplot(pall, ax=ax, color=COLS[i % 10],label=lbl, linewidth=lnwidth)
    ax = plt.subplot(NR, NC, 8)
    if xl:
        ax.set_xlim(xl)
    ax.set_ylim((0,1.05))
    ax.set_xlabel(pxlabel, fontsize=fnsize_axes)
    ax.set_ylabel('CDF', fontsize=fnsize_axes)
    ax.legend(loc='lower right', fontsize=fnsize_legend)
    # plt.subplots_adjust(hspace=0.5)
    plt.tight_layout()

# use col_base=5 for Rips in slides
def plotCDFS_merge(rootname, plist, th = np.inf, chop = 0, loglog=False, tail=False, group_by_dim = True, group_by_type=False, group_labels = True, col_base = 0, xl = None, fnsize_title=30, fnsize_axes=24, fnsize_legend=12, fnsize_tick = 12, lnwidth=3, colors=None, ax=None):

    # Setting the x-axis label.
    if loglog:
        xlabel = '$\ell = A\cdot\log\log('+PCHAR+')+B$'
    else:
        xlabel = '$\log('+PCHAR+')$'

    if ax == None:
        ax = plt.axes()

    # Reversing the plotting order.
    plist = plist.copy()
    if group_by_dim or group_by_type:
        plist.reverse()

    lbl_list = []
    for i, prms in enumerate(plist):
        _, _, pall, lbl = getSample(rootname, prms, delta=0, th=th)
        T, G, D, K = prms

        if loglog:
            pall = np.log(pall)
            pall = pall*AVALS[T]
            pall = pall - pall.mean() + st.loggamma(1).mean()
        pall.sort()
        if chop > 0:
            pall = pall[:-chop]
        col_idx = i%10
        if group_by_dim:
            col_idx = COLS_DICT[(D,K)]
            col_idx += col_base
            col_idx = col_idx % 10
            if group_labels:
                if (D, K) in lbl_list:
                    lbl = None
                else:
                    lbl = 'd=%d , k=%d' % (D,K)
                    lbl_list.append((D,K))
        if group_by_type:
            col_idx = T-1
        if xl != None:
            pall = np.append(pall, xl[1])

        if colors != None:
            col = colors[i]
        else:
            col = COLS[col_idx]
        if tail:
            sns.ecdfplot(pall, ax=ax, color=col,label=lbl, linewidth=lnwidth, complementary=True,log_scale=(False,True))
        else:
            sns.ecdfplot(pall, ax=ax, color=col, label=lbl, linewidth=lnwidth)
    #
    if not tail:
        ax.set_ylim((-0.05,1.05))
    if xl != None:
        ax.set_xlim(xl)
    ax.set_xlabel(xlabel, fontsize=fnsize_axes)

    if tail:
        ax.set_title('log(1-CDF)', fontsize=fnsize_title)
    else:
        ax.set_title('CDF', fontsize=fnsize_title)

    ax.set_ylabel(None)
    if tail:
        leg_loc = 'lower left'
    else:
        leg_loc = 'lower right'
    if fnsize_legend > 0:
        nc = 1
        if group_by_dim or group_by_type:
            handles, labels = ax.get_legend_handles_labels()
            handles = handles[::-1]
            labels = labels[::-1]
            if (group_labels == False):
                nc = 2
            ax.legend(handles, labels, fontsize=fnsize_legend,loc=leg_loc, ncol=nc)
        else:
            # nc = 2
            ax.legend(fontsize=fnsize_legend, loc=leg_loc, ncol=nc)
    plt.xticks(fontsize=fnsize_tick)
    plt.yticks(fontsize=fnsize_tick)
    plt.tight_layout()

def plotLogLogDist(rootname, plist, rv=st.gumbel_l(), chop=0, lnwidth=3, fnsize_title=20,  fnsize_legend = 14, fnsize_tick = 14, no_legend=True, xl = None, xlqq = None, group_by_dim=True, iid_test=False, death_th = 0, th=np.inf, adj_bw = None, colors=None):
    if adj_bw != None:
        adj = adj_bw
    else:
        adj = np.ones(len(plist))

    lbl_list = []


    for i, prms in enumerate(plist):
        T,G,D,K = prms
        # if i < 20:
        #      continue

        ball, dall, pall, lbl = getSample(rootname, prms, delta=0, th=th)

        if (len(pall)==0):
            print("EMPTY")
            continue

        if (death_th > 0):
            IDX = np.where(np.exp(dall) < death_th)
            pall = pall[IDX]
            print(len(IDX[0]))

        pall = np.log(pall)
        pall = pall * AVALS[T]
        pall = pall - pall.mean() + rv.mean()

        if (chop > 0):
            pall.sort()
            pall = pall[:-chop]

        if xl!=None:
            pall_cdf = np.append(pall, xl)
        else:
            pall_cdf = pall

        col_idx = i % 10
        if group_by_dim:
            col_idx = COLS_DICT[(D,K)]
            col_idx = col_idx % 10
            if (D, K) in lbl_list:
                lbl = None
            else:
                lbl = 'd=%d , k=%d' % (D,K)
                lbl_list.append((D,K))

        if no_legend:
            lbl = None

        if colors != None:
            col = colors[i]
        else:
            col = COLS[col_idx]

        ax = plt.subplot(1, 3, 1)
        sns.ecdfplot(pall_cdf, ax=ax, label=lbl, linewidth=lnwidth, color=col)

        ax = plt.subplot(1,3,2)

        print(adj[i])
        sns.kdeplot(pall, ax=ax, label=lbl, color=col, linewidth=lnwidth, clip=xl,bw_adjust=adj[i])#/prms[4]*4)
        # x, y = NaiveKDE(kernel='cosine', bw='ISJ').fit(pall).evaluate()
        # plt.plot(x,y)

        ax = plt.subplot(1,3,3)
        pp_y = sm.ProbPlot(pall, dist=rv)

        pp_y.qqplot(ax=ax, markerfacecolor=col, markeredgecolor=COLS[col_idx], markersize=5, label=lbl)

    if iid_test:
        D = np.load('../3/2samp_iid10k.npy')
        print(D.shape)
        ball = D[:,0]
        dall = D[:,1]
        pall = dall/ball
        IDX = np.where(pall > 1)
        pall = np.log(pall[IDX])
        pall = np.log(pall)
        pall = pall * AVALS[T]
        pall = pall - pall.mean() + rv.mean()

        if xl != None:
            pall_cdf = np.append(pall, xl)
        else:
            pall_cdf = pall
        lbl = 'IID'
        col_idx = (i+1) % 10

        ax = plt.subplot(1, 3, 1)
        sns.ecdfplot(pall_cdf, ax=ax, label=lbl, linewidth=lnwidth, color=COLS[col_idx])

        ax = plt.subplot(1, 3, 2)
        sns.kdeplot(pall, ax=ax, label=lbl, color=COLS[col_idx], clip=xl,
                    bw_adjust=1.5)  # ,bw_adjust=adj)# )#,bw_adjust=adj)#/prms[4]*4)

        ax = plt.subplot(1, 3, 3)
        pp_y = sm.ProbPlot(pall, dist=rv)

        pp_y.qqplot(ax=ax, markerfacecolor=COLS[col_idx], markeredgecolor=COLS[col_idx], markersize=5, label=lbl)

    ax = plt.subplot(1, 3, 1)
    ax.set(ylim=(-0.05, 1.05))

    if xl==None:
        xl = ax.get_xlim()

    x = np.linspace(xl[0],xl[1],1000)
    ax.plot(x, rv.cdf(x), 'k:', linewidth=5, label='LGumbel')
    ax.tick_params(axis='x', labelsize=fnsize_tick)
    ax.tick_params(axis='y', labelsize=fnsize_tick)

    ax.set_xlim(xl)
    ax.set_ylabel(None)

    ax.set_title('CDF', fontsize=fnsize_title)

    ax.legend(loc='upper left', fontsize=fnsize_legend)

    ax = plt.subplot(1,3,2)
    ax.plot(x, rv.pdf(x), 'k:', linewidth=5, label='LGumbel')
    ax.set_xlim(xl)
    yl = ax.get_ylim()
    ax.set_ylim([-0.05, yl[1]])
    ax.set_ylabel(None)
    ax.set_title('PDF', fontsize=fnsize_title)
    ax.tick_params(axis='x', labelsize=fnsize_tick)
    ax.tick_params(axis='y', labelsize=fnsize_tick)

    if xlqq==None:
        xlqq = xl
    ax = plt.subplot(1, 3, 3)
    ax.plot(xlqq, xlqq,'--k', linewidth=3)
    ax.set_xlim(xlqq)
    ax.set_ylim(xlqq)
    ax.set_title('QQ-Plot', fontsize=fnsize_title)
    # ax.set_xlabel('Log-Exponential Quantiles', fontsize=fnsize)
    # ax.set_ylabel('Sample Quantiles', fontsize=fnsize)
    ax.set_xlabel(None)
    ax.set_ylabel(None)
    ax.tick_params(axis='x', labelsize=fnsize_tick)
    ax.tick_params(axis='y', labelsize=fnsize_tick)

    plt.tight_layout()


def plotLogDist(rootname, plist, rv, xl = (0,10), xlqq = (0, 10)):

    cols = sns.color_palette()
    rv2 = st.loggamma(c=1,loc=0,scale=1)
    for i, prms in enumerate(plist):

        dirname = rootname + '/' + prms[1] + '/' + prms[0] + '/' + str(prms[4])
        lbl = ' / '.join(map(str,prms))
        D = prms[2]
        K = prms[3]

        ball, dall = appendAll(dirname, D, K)
        _, _, pall = preparePers(ball, dall)

        print(lbl, len(pall))

        if (len(pall)==0):
            print("EMPTY")
            continue

        if (prms[0] == 'RIPS'):
            A = 1
        else:
            A = 1/2

        # lpall = np.log(np.log(pall))
        # B = rv2.mean() - lpall.mean() * A
        # lpall = A * lpall + B
        # lpall = np.exp(lpall)
        lpall = np.log(pall) ** A
        m = lpall.mean()
        lpall = lpall/m

        ax = plt.subplot(1, 3, 1)
        sns.ecdfplot(lpall, ax=ax, label=lbl)

        ax = plt.subplot(1,3,2)
        sns.ecdfplot(lpall, ax=ax, label=lbl, log_scale=(False, True), complementary=True)

        ax = plt.subplot(1,3,3)
        pp_y = sm.ProbPlot(lpall, dist=rv)
        pp_y.qqplot(ax=ax, markerfacecolor=cols[i%10], markeredgecolor=cols[i%10], markersize=5, label=lbl)

    ax = plt.subplot(1, 3, 1)
    x = np.linspace(xl[0],xl[1],1000)
    ax.plot(x, rv.cdf(x), 'k:', label='exponential')
    ax.set_ylim((0, 1.05))
    ax.set_xlim(xl)
    ax.legend(loc='upper right', bbox_to_anchor=(-0.15, 1), fontsize=14)

    ax = plt.subplot(1,3,2)
    ax.semilogy(x[:-1], 1-rv.cdf(x[:-1]), 'k:', label='exponential')
    ax.set_xlim(xl)

    ax = plt.subplot(1, 3, 3)
    ax.plot(xlqq, xlqq,'--w', linewidth=3)
    ax.set_xlim(xlqq)
    ax.set_ylim(xlqq)

def plotLogBD(rootname, plist, rvb, rvd, xl=(-10,4), xlqq=(-10,4)):

        cols = sns.color_palette()
        rv2 = st.loggamma(c=1, loc=0, scale=1)
        for i, prms in enumerate(plist):

            dirname = rootname + '/' + prms[1] + '/' + prms[0] + '/' + str(prms[4])
            lbl = ' / '.join(map(str, prms))
            D = prms[2]
            K = prms[3]

            ball, dall = appendAll(dirname, D, K)
            ball, dall, _ = preparePers(ball, dall)
            ball = np.log(ball)
            dall = np.log(dall)
            mb = ball.mean()
            sb = ball.std()
            md = dall.mean()
            sd = dall.std()
            ball = (ball-mb)/sb
            dall = (dall-md)/sd
            print(lbl, len(ball))
            print(mb, sb, md, sd)

            ax = plt.subplot(2, 3, 1)
            sns.ecdfplot(ball, ax=ax, label=lbl)

            ax = plt.subplot(2, 3, 2)
            sns.kdeplot(ball, ax=ax, label=lbl)

            ax = plt.subplot(2, 3, 3)
            pp_y = sm.ProbPlot(ball, dist=rvb)
            pp_y.qqplot(ax=ax, markerfacecolor=cols[i % 10], markeredgecolor=cols[i % 10], markersize=5, label=lbl)

            ax = plt.subplot(2, 3, 4)
            sns.ecdfplot(dall, ax=ax, label=lbl)

            ax = plt.subplot(2, 3, 5)
            sns.kdeplot(dall, ax=ax, label=lbl)

            ax = plt.subplot(2, 3, 6)
            pp_y = sm.ProbPlot(dall, dist=rvd)
            pp_y.qqplot(ax=ax, markerfacecolor=cols[i % 10], markeredgecolor=cols[i % 10], markersize=5, label=lbl)

        ax = plt.subplot(2, 3, 1)
        ax.set_ylim((0, 1.05))
        if xl:
            ax.set_xlim(xl)
            x = np.linspace(xl[0], xl[1], 1000)
            ax.plot(x, rvb.cdf(x), 'k:', label='loggamma')

        ax.legend(loc='upper right', bbox_to_anchor=(-0.15, 1), fontsize=14)

        ax = plt.subplot(2, 3, 2)
        if xl:
            ax.set_xlim(xl)
            ax.plot(x, rvb.pdf(x), 'k:', label='loggamma')

        ax = plt.subplot(2,3,3)
        ax.plot(xlqq, xlqq, '--w', linewidth=3)
        ax.set_xlim(xlqq)
        ax.set_ylim(xlqq)

        ax = plt.subplot(2, 3, 4)
        ax.set_ylim((0, 1.05))
        if xl:
            ax.set_xlim(xl)
            ax.plot(x, rvd.cdf(x), 'k:', label='loggamma')

        ax = plt.subplot(2, 3, 5)
        if xl:
            ax.set_xlim(xl)
            ax.plot(x, rvd.pdf(x), 'k:', label='loggamma')

        ax = plt.subplot(2, 3, 6)
        ax.plot(xlqq, xlqq, '--w', linewidth=3)
        ax.set_xlim(xlqq)
        ax.set_ylim(xlqq)