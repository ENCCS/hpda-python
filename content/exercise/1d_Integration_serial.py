import math
import time

# Grid size
n = 100000000

def integration_serial(n):
    h = 1.0 / float(n)
    mysum = 0.0
    
    for i in range(n):
        x = h * (i + 0.5)
        mysum += x ** (3/2)

    return h * mysum

if __name__ == "__main__":
    starttime = time.time()
    integral = integration_serial(n)
    endtime = time.time()

    print("Integral value is %e, Error is %e" % (integral, abs(integral - 2/5)))  # The correct integral value is 2/5
    print("Time spent: %.2f sec" % (endtime-starttime))

# 13.63 sec
