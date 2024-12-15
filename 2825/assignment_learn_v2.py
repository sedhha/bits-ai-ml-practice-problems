import numpy as np

elements = [1,2,3,4,5,6,7,8,9,0]
print(np.argmin(elements))
print(np.argmax(elements))

for (i,j) in enumerate(elements):
    print(i,j)
