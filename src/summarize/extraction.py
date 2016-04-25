import  numpy as np
__author__ = 'Mengdi'

def cosine_similarity(A):
    similarity = np.dot(A, A.T)
    square_mag = np.diag(similarity)
    inv_square_mag = 1 / square_mag
    inv_square_mag[np.isinf(inv_square_mag)] = 0
    inv_mag = np.sqrt(inv_square_mag)
    cosine = similarity * inv_mag
    cosine = cosine.T * inv_mag
    return cosine

def distance_matrix_py(pts):
    """Returns matrix of pairwise Euclidean distances. Pure Python version."""
    n = len(pts)
    p = len(pts[0])
    m = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            s = 0
            for k in range(p):
                s += (pts[i,k] - pts[j,k])**2
            m[i, j] = s**0.5
    return m

def extract_summary_nopenaty(data, K, N):

    euclideanDist = distance_matrix_py(data)
    sortDist = np.sort(euclideanDist)
    kNeighborDist = sortDist[:,:K]
    kDistSum = np.sum(kNeighborDist, axis = 1)
    score = kDistSum
    sortIdex = np.argsort(score) # sort score
    sortSum = np.sort(score)
    choseIndex = sortIdex[:N] # extract N summarization

    return choseIndex