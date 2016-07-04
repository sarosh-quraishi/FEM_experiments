import numpy as np


def create_Pmat(j):
    '''(nonnegative int)->matrix
    Return Pj matrix of floating point numbers with r,c rows and cols, c=no of basis functions in jth level
    Pj matrix repeats the pattern [1/2,1,1/2] in the interior
    >>> create_Pmat(0)
    array([[ 1. ,  0. ],
           [ 0.5,  0.5],
           [ 0. ,  1. ]])
    >>> create_Pmat(1)
    array([[ 1. ,  0. ,  0. ],
           [ 0.5,  0.5,  0. ],
           [ 0. ,  1. ,  0. ],
           [ 0. ,  0.5,  0.5],
           [ 0. ,  0. ,  1. ]])

    '''
    # get number of basis functions (columns) c=2j+1 for each j
    c = 2 ** j + 1
    # number of rows in extended matrix r_ext=3c-(c-1)
    r_ext = 2 * c + 1
    # initialize Pextended
    Pextended = np.zeros((r_ext, c))
    ppattern = np.array([0.5, 1.0, 0.5])
    # make a column pattern (a row vector with r entries starting with
    # ppattern and rest of entries zeros
    colpattern = np.zeros([r_ext])
    colpattern[[0, 1, 2]] = ppattern
    # roll colppatern down each column of Pextended
    for col in np.arange(c):
        Pextended[:, col] = np.roll(colpattern, 2 * col)

    Pmat = Pextended[1:-1, :]
    return Pmat


def create_Qmat(j):
    '''(nonnegative int)->matrix
    Return Qj matrix of floating point numbers with r,c rows and cols, c=no of bubble functions in jth level
    Qj matrix repeats the pattern [0.0,0.0,1.0] in the interior

    >>> create_Qmat(0)
    array([[ 0.],
           [ 1.],
           [ 0.]])

    >>> create_Qmat(1)
    array([[ 0.,  0.],
           [ 1.,  0.],
           [ 0.,  0.],
           [ 0.,  1.],
           [ 0.,  0.]])

    '''
    # get number of basis functions (columns) c=P.r-P.c for each j
    c = 2 ** j
    # get number of rows r=P.r
    r_ext = 2 ** (j + 1) + 3
    # initialize Pextended
    Qextended = np.zeros((r_ext, c))
    qpattern = np.array([0.0, 0.0, 1.0])
    # make a column pattern (a row vector with r entries starting with
    # qpattern and rest of entries zeros
    colpattern = np.zeros([r_ext])
    colpattern[[0, 1, 2]] = qpattern
    # roll colppatern down each column of Qextended
    for col in np.arange(c):
        Qextended[:, col] = np.roll(colpattern, 2 * col)

    Qmat = Qextended[1:-1, :]
    return Qmat


def create_Tmat(j):
    '''(int)->mat
    takes in resolution j and assembles Wmat=[Pmat|Qmat]
    '''
    Pmat = create_Pmat(j)
    Qmat = create_Qmat(j)
    Tmat = np.concatenate((Pmat, Qmat), axis=1)
    return Tmat


def create_Rmat(j):
    '''(int)->mat
    takes in resolution j and assembles Wmat=[Pmat|Qmat]
    '''
    Pmat = create_Pmat(j)
    dim = Pmat.shape[0]
    Qmat = np.eye(dim, dim)
    Tmat = np.concatenate((Pmat, Qmat), axis=1)
    return Tmat


def nodal2hier(K, F, W):
    '''(mat,mat,mat)->mat,mat
    (K,F,Wmat)->Ktilde,Ftilde
    '''
    aa = np.dot(W.T, K)
    Kt = np.dot(aa, W)
    Ft = np.dot(W.T, F)
    return Kt, Ft

if __name__ == "__main__":
    import doctest
    doctest.testmod()