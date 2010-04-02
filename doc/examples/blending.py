import numpy as np
from pylab import *
from csc.divisi.tensor import DenseTensor
from csc.divisi.blend import Blend

#NUM_SAMPLES = 25
#factors = np.linspace(0, 1, NUM_SAMPLES)

#"Generate a random matrix."

#mat1 = DenseTensor(np.random.random((15,15)))
#mat2 = DenseTensor(np.random.random((15,15)))
#svd = mat1.svd()
# TODO: figure out why the svals always go 8 then a line from 2 to 0.

def overlap_matrices(mat1, mat2, row_overlap, col_overlap):
    '''
    Construct labeled tensors such that the two matrices overlap
    by the given number of rows and columns.
    '''
    r1, c1 = mat1.shape
    r2, c2 = mat2.shape
    t1 = mat1.labeled([
            map(str, range(r1)),
            map(str, range(c1))])
    t2 = mat2.labeled([
            map(str, range(r1-row_overlap, r1+r2-row_overlap)),
            map(str, range(c1-col_overlap, c1+c2-col_overlap))])
    return t1, t2


def svals_by_overlap(factors, mat1, mat2, row_overlap, col_overlap):
    blend = Blend(overlap_matrices(mat1, mat2, row_overlap, col_overlap), factor=0)
    return [blend.svals_at_factor(factor, k=15) for factor in factors]

def predicted_by_overlap(factors, mat1, mat2, row_overlap, col_overlap):
    blend = Blend(overlap_matrices(mat1, mat2, row_overlap, col_overlap), factor=0)
    return [blend.predicted_svals_at_factor(factor, num=15) for factor in factors]

def veering_by_overlap(mat1, mat2, row_overlap, col_overlap):
    t1, t2 = overlap_matrices(mat1, mat2, row_overlap, col_overlap)
    blend = Blend([t1, t2], factor=0)
    return [blend.total_veering_at_factor(factor, num=15)
            for factor in factors]

def plot_datasets(x, ys, *a, **kw):
    return [plot(x, y, *a, **kw) for y in ys]

def svals_at_factor(factor, blend_name='blend'):
    blend = globals()[blend_name]
    blend.factor = factor
    return blend.svd().svals.values()



def manual_blend_setup(rows, cols, row_overlap, col_overlap):
    np.random.seed(5)

    base = 0.001 * np.random.random((2*rows,2*cols))

    # random matrices
    mat1 = np.random.random((rows,cols))
    mat2 = np.random.random((rows,cols))

    # Fill in
    m1x = np.zeros((2*rows,2*cols))
    m1x[:rows,:cols] = mat1

    m2x = np.zeros((2*rows, 2*cols))
    m2x[rows-row_overlap:2*rows-row_overlap,
        cols-col_overlap:2*cols-col_overlap] = mat2

    return base, m1x, m2x

def manual_blend_svals(base, m1x, m2x, factor):
    m = base + (1-factor)*m1x + factor*m2x
    return np.linalg.svd(m, compute_uv=False)

def manual_blend_svec1(base, m1x, m2x, factor):
    m = base + (1-factor)*m1x + factor*m2x
    U, s, Vh = np.linalg.svd(m, compute_uv=True)
    return U[:,0]
