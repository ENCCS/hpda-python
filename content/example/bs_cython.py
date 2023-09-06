%%cython

cpdef long[:] bs_cython(long[:] a_list):
    cdef int N, i, j
    N = len(a_list)
    for i in range(N):
        for j in range(1, N-i):
            if a_list[j] < a_list[j-1]:
                a_list[j-1], a_list[j] = a_list[j], a_list[j-1]
    return a_list
