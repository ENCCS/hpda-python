# Benchmark properly
%%timeit 
d_u = numba.cuda.to_device(u)
d_unew = numba.cuda.to_device(unew)
lap2d_cuda[(16,16),(16,16)](d_u, d_unew); numba.cuda.synchronize()
d_unew.copy_to_host(unew)
