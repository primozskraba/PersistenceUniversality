import array
import numpy as np
import scipy
# import numpy.random
from sklearn import neighbors
import random
import scipy.stats as st

PATCH_SIZE = 3
RMIN = 2
RMAX = 1022-PATCH_SIZE
CMIN = 2
CMAX = 1534-PATCH_SIZE

DMAT = np.array([[2, -1, 0, -1, 0, 0, 0, 0, 0],
[-1, 3, -1, 0, -1, 0, 0, 0, 0],
[0, -1, 2, 0, 0, -1, 0, 0, 0],
[-1, 0, 0, 3, -1, 0, -1, 0, 0],
[0, -1, 0, -1, 4, -1, 0, -1, 0],
[0, 0, -1, 0, -1, 3, 0, 0, -1],
[0, 0, 0, -1, 0, 0, 2, -1, 0],
[0, 0, 0, 0, -1, 0, -1, 3,   -1],
[0, 0, 0, 0, 0, -1, 0, -1, 2]])

e1 = np.array([1, 0, -1, 1, 0, -1, 1, 0, -1]) / np.sqrt(6)
e2 = np.array([1, 1, 1, 0, 0, 0, -1, -1, -1]) / np.sqrt(6)
e3 = np.array([1, -2,1,1, -2,1,1, -2,1]) / np.sqrt(54)
e4 = np.array([1, 1, 1, -2, -2, -2, 1, 1, 1]) / np.sqrt(54)
e5 = np.array([1, 0, -1, 0, 0, 0, -1, 0, 1]) / np.sqrt(8)
e6 = np.array([1,0, -1, -2,0,2,1,0, -1]) / np.sqrt(48)
e7 = np.array([1, -2,1,0,0,0, -1,2, -1]) / np.sqrt(48)
e8 = np.array([1, -2, 1, -2, 4, -2, 1, -2, 1]) / np.sqrt(216)

AMAT = np.array([e1,e2,e3,e4,e5,e6,e7,e8])
LMAT = np.diag(1/np.sum(AMAT**2,1))

#
# Getting the list of all file-indexes.
#
def get_flist():
    MX = 4212
    FL =[]
    for i in range(1,MX+1):
        fname = '../../vanhateren_iml/imk%05d.iml' % i
        # fname = 'datasets/vh/imk%05d.iml' % i
        try:
            #print(fname)
            open(fname, 'rb')
            FL = FL + [i]
        except Exception:
            #print(i)
            pass
    # print(FL)
    return FL

#
# Reading the entire image from a file.
#
def get_img(IDX):
    filename = '../../vanhateren_iml/imk%05d.iml' % IDX
    # filename = 'datasets/vh/imk%05d.iml' % IDX
    # filename = 'datasets/vh/imk%05d.iml' % IDX
    with open(filename, 'rb') as handle:
        s = handle.read()
        handle.close()
        arr = array.array('H', s)
        arr.byteswap()
    img = np.array(arr, dtype='uint16').reshape(1024, 1536)
    return img

#
#  Exrtract N random patches from image file I.
#
def get_patch(N, I):
    P = np.zeros([N, PATCH_SIZE**2])
    IMG = get_img(I)
    i = 0
    while i < N:
        PR = np.random.random_integers(RMIN, RMAX)
        PC = np.random.random_integers(CMIN, CMAX)
        # print([PR[0], PC[0], PATCH_SIZE])
        x = IMG[PR:PR+PATCH_SIZE, PC:PC+PATCH_SIZE]
        x = x.reshape(1,PATCH_SIZE**2)
        if (x.min() == x.max()):
            continue
        P[i,:] = x
        i += 1
    return P

#
# Normalizing each patch (mean & D-variance).
# Eq (1) in [LPM]
#
def norm_patch(P):
    N = P.shape[0]
    D = P.shape[1]
    PP = np.zeros([N,D])
    PN = np.zeros(N)

    for i in range(N):
        x = P[i]

        x = np.double(np.log(x+1))
        zx = x - np.mean(x)
        nx = np.sqrt(zx@DMAT@zx.T)
        if (nx==0):
            print(zx)
        if(nx>0):
            y = zx / nx
            PP[i,:] = y
            PN[i] = nx
        else:
            print("ZEROS")
            PP[i,:] = zx
            PN[i] = 0
    return [PN, PP]

#
# Extracting patches from image files.
# N = number of patches. FL = list of file indexes, K = knn parameter.
# RVAR - Percentage of top-variance patches to keep.
# RKNN - Percentage of top-knn patches to keep.
def gen_all_patches(N, FL, RKNN=0.3, RVAR = 0.2, K=15):

    # RVAR = 0.2 # Percentage of top-variance patches to keep.
    # RKNN = 0.3 # Percentage of top-knn patches to keep.
    D = 9
    NUM_FILES = len(FL)

    LOAD = 0
    if (FL == []):
        LOAD = 1
        NUM_FILES = 1
    print([N, RVAR, RKNN, NUM_FILES])
    NTOTAL = int(np.ceil(N / RVAR / RKNN))
    NFILE = int(np.ceil(NTOTAL / NUM_FILES))
    NTOTAL = int(np.ceil(NFILE * NUM_FILES))
    NKNN = int(np.ceil(N/RKNN))

    if (LOAD == 1):
        ALLP = np.load('./point_files/patches_nonconst.npy')
        print(len(ALLP))
        print(NTOTAL)
        IDX = random.sample(range(len(ALLP)), NTOTAL)
        ALLP = ALLP[IDX]

    else:
        ALLP = np.zeros([NTOTAL, D])
        for i in range(NUM_FILES):
            P = get_patch(NFILE, FL[i])
            # print(P.shape)
            ALLP[i * NFILE:(i + 1) * NFILE,:] = P

    [PN, PP] = norm_patch(ALLP)
    IDX_ORIG = np.array(IDX)
    IDX = np.argsort(-PN)
    # print(len(IDX_ORIG))
    # print(IDX.shape)
    IDX_ORIG = IDX_ORIG[IDX[:NKNN]]
    P2 = PP[IDX[:NKNN]]
    P2 = P2@AMAT.T@LMAT.T

    print(P2.shape)

    # No KNN is performed. Just take N samples.
    if (K==0):
        return P2

    # Computing KNN, and taking the top RKNN.
    nbrs = neighbors.NearestNeighbors(n_neighbors=K+1, algorithm='kd_tree').fit(P2)
    print('kdt')
    distances, indices = nbrs.kneighbors(P2)
    d = distances[:,K]
    IDX = np.argsort(d)
    # kde = st.gaussian_kde(P2.T)
    # kvals = kde(P2.T)
    # IDX = np.argsort(-kvals)

    P3 = P2[IDX[0:N],:]
    IDX_ORIG = IDX_ORIG[IDX[0:N]]
    print(P2.shape)
    print(P3.shape)
    return P3, IDX_ORIG
