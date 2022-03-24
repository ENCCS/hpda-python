def lap2d(u, uout):

    M = u.shape[0]     
    N = u.shape[1]    
 
    for i in range(1, M-1):
        for j in range(1, N-1):             
            uout[i, j] = 0.25 * ( u[i+1, j] + u[i-1, j] + u[i, j+1] + u[i, j-1] )     

