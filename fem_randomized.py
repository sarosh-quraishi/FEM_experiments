import numpy as np
import itertools

from fem1d_classes import *
from math import *
import hierarchical_basis_helper as hier
from functools import reduce
from scipy.linalg import block_diag


rhsfn = lambda x: x*(x+3)*e**x
exactfn = lambda x: x*(1-x)*e**x

def FEM_system(level, rhsfn):
    #
    #  Set a mesh of N=2**(level) elements or 2**level+1 nodes in [ xleft, xright ].
    #

    xleft = 0.0
    xright = 1.0
    N  = 2**(level)


    mesh = Mesh ( N, xleft, xright )
    #
    #  Define the shape functions associated with any element.
    #
    sfns = Shapefns ( )
    #
    #  The mesh and shape functions define the function space V.
    #
    V = FunctionSpace ( mesh, sfns )
    #
    #  Get the coordinates of the degrees of freedom.
    #  Because we use quadratic Lagrange elements, there are 2*N+1 of them.
    #
    x = V.dofpts()
    #
    #  Evaluate the right hand side of the PDE at each node.
    #
    rhs = rhsfn ( x )
    #
    #  Compute b, the right hand side of the finite element system.
    #
    b = V.int_phi ( rhs )
    #
    #  Compute A, the stiffness matrix.
    #
    A = V.int_phi_phi ( derivative = [ True, True ] )

    return A, b, x

def BoundaryCondition(A,b,exactfn,x):
    #
    #  Modify the linear system to enforce the left boundary condition.
    #
    exact = exactfn ( x )
    A[0,0] = 1.0
    A[0,1:] = 0.0
    b[0] = exact[0]

    #
    #  Modify the linear system to enforce the right boundary condition.
    #
    A[-1,-1] = 1.0
    A[-1,0:-1] = 0.0
    b[-1] = exact[-1]
    return A,b

def getAjj(level, rhsfn, exactfn):
    A11,b1,x = FEM_system(level, rhsfn)
    A11,b1 = BoundaryCondition(A11,b1,exactfn,x)
    return A11, b1


def get_next_A(A11,b1, level, rhsfn, exactfn):
    """
    Given A11 mat from previous level, and next level find next A matrix
    """
    Av, bv = getAjj(level, rhsfn, exactfn)
    Add = Av
    Av_list=[]
    bv_list=[bv]
    alist = np.arange(level,0,-1)
    for idx, i in enumerate(alist):
        Pmat = hier.create_Pmat(i)  # order is P21,P32, ...
        Av = np.dot(Pmat.T,Av)
        Av_list.append(Av)
        bv = np.dot(Pmat.T,bv)
        bv_list.append(bv)

    Avv = list(reversed(Av_list))
    Ahh = [Av.T for Av in Avv]
    Avertical = np.vstack(Avv)

    Ahorizontal = np.hstack(Ahh)
    A22 = np.array(np.bmat([[A11, Avertical], [Ahorizontal,Add]]))
    b2 = np.array(list(itertools.chain(*bv_list)))
    return A22, b2

def get_multilevel_Ab(maxlevel, rhsfn, exactfn):
    A11,b1 = getAjj(0, rhsfn, exactfn)
    for level in np.arange(1,maxlevel):
        A11, b1 = get_next_A(A11,b1, level, rhsfn, exactfn)
    return A11, b1


def append_system(U, n):
    return block_diag(U,np.eye(n))
def extend_system(x1,b1,n):
    x1list = [x1,np.zeros(n)]
    b1list = [b1,np.zeros(n)]
    b1 = np.array(list(itertools.chain(*b1list)))
    x1 = np.array(list(itertools.chain(*x1list)))
    return x1,b1


def get_sv(A11,sthresh=1e-1):
    U1, s1, V1 = np.linalg.svd(A11)
    U1 = U1[:,s1>sthresh]
    V1 = V1[:,s1>sthresh]
    return U1,V1
def project_system(U1,V1,A11,b1):
    return np.dot(U1.T,np.dot(A11,V1)), np.dot(U1.T,b1)



def get_residual(maxlevel, rhsfn, exactfn):
    A11,b1 = get_multilevel_Ab(0, rhsfn, exactfn)
    U1,V1 = get_sv(A11)
    A11,b1 = project_system(U1,V1,A11,b1)
    print('coarse scale system '+ str(A11.shape))
    x1 = np.squeeze(np.linalg.solve(A11, b1))
    print('x1 tilde ' + str(x1.shape))

    Ulist = [U1]
    Vlist = [V1]

    for level in np.arange(2,maxlevel):
        Ajj,bj = getAjj(level-1, rhsfn, exactfn)
        n=len(Ajj)
        Ulist = [append_system(U,n) for U in Ulist]
        U1 = reduce(lambda x, y: np.dot(x,y), Ulist)

        Vlist = [append_system(U,n) for U in Vlist]
        V1 = reduce(lambda x, y: np.dot(x,y), Vlist)

        x1 = extend_system(x1,n)
        print('x_(j) tilde extended with zeros '+ str(x1.shape ) )
        A11,b1 = get_multilevel_Ab(level, rhsfn, exactfn)
        print('projected extended system ' + str(A11.shape))
        print('matrices used for projection: '+ str(U1.shape))
        print(V1.shape)
        A11,b1 = project_system(U1,V1,A11,b1)
        res = np.dot(A11,x1) - b1

        if res.all()>thresh:
            U1,V1 = get_sv(A11)
            Ulist.append(U1)
            Vlist.append(V1)
            A11,b1 = project_system(U1,V1,A11,b1)
            x1 = np.squeeze(np.linalg.solve(A11, b1))
            print('solution of next level projected system x_(j+1) tilde' + str(x1.shape))
        else:
            print('breaking at level: ' + level)
            break
    return res, Ulist, Vlist

