%%cython
def bs_cython(a_list):
    N = len(a_list)
    for i in range(N):
        for j in range(1, N):
            if a_list[j] < a_list[j-1]:
                a_list[j-1], a_list[j] = a_list[j], a_list[j-1]
    return a_list


%%cython
cpdef bs_cython_dtype(a_list):
    cdef int N, i, j # static type declarations
    N = len(a_list)
    for i in range(N):
        for j in range(1, N):
            if a_list[j] < a_list[j-1]:
                a_list[j-1], a_list[j] = a_list[j], a_list[j-1]
    return a_list



%%cython
cimport cython
from libc.stdlib cimport malloc, free

cpdef bs_clist(a_list):

    cdef int *c_list
    c_list = <int *>malloc(len(a_list)*cython.sizeof(int))
    cdef int N, i, j # static type declarations
    N = len(a_list)
    
    # convert Python list to C array
    for i in range(N):
        c_list[i] = a_list[i]

    for i in range(N):
        for j in range(1, N):
            if c_list[j] < c_list[j-1]:
                c_list[j-1], c_list[j] = c_list[j], c_list[j-1]

    # convert C array back to Python list
    for i in range(N):
        a_list[i] = c_list[i]
        
    free(c_list)
    return a_list



%%cython
def bs_cython2(X):
    N = len(X)
    for end in range(N, 1, -1):
        for i in range(end - 1):
            cur = X[i]
            if cur > X[i + 1]:
                tmp = X[i]
                X[i] = X[i + 1]
                X[i + 1] = tmp
