from numba import jit
@jit
def bs_numba(a_list):

    N = len(a_list)
    for i in range(N):
        for j in range(1, N):
            if a_list[j] < a_list[j-1]:
                a_list[j-1], a_list[j] = a_list[j], a_list[j-1]
    return a_list
