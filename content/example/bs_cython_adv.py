%%cython

cimport cython
from libc.stdlib cimport malloc, free

cpdef bs_clist(a_list):
    cdef int *c_list
    c_list = <int *>malloc(len(a_list)*cython.sizeof(int))
    cdef int N, i, j 
    N = len(a_list)
    
    for i in range(N):
        c_list[i] = a_list[i]

    for i in range(N):
        for j in range(1, N):
            if c_list[j] < c_list[j-1]:
                c_list[j-1], c_list[j] = c_list[j], c_list[j-1]

    for i in range(N):
        a_list[i] = c_list[i]
        
    free(c_list)
    return a_list
