def heat(u, u_new, a, dt, dx, dy):

n = u.shape[0]     
m = u.shape[1]    
 
for i in range(1, n-1):
    for j in range(1, m-1):             
        u[i, j] = u[i, j] + a * dt * (  
        (u[i+1, j] - 2*u[i, j] + u[i-1, j]) /dx**2 +               
        (u[i, j+1] - 2*u[i, j] + u[i, j-1]) /dy**2 )     


u[:,:] = u[:,:]

return
