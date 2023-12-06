
import numpy as np
import random
_timeOut = np.matrix('0 0 0 0 0; 0 0 0 0 0; 0 0 0 0 0; 0 0 0 0 0; 0 0 0 0 -1')
_timeOut += 1

sero = np.zeros(_timeOut.shape,dtype='uint8') + 10

#_timeOut[np.ix_([1,3],[1,3])] = np.matrix('10 10 10; 10 10 10; 10 10 10')
#print(_timeOut)
#print(_timeOut[np.ix_([1,3],[1,3])].shape)
'''
print(round(random.uniform(0.4, 0.78), 2))
print(_timeOut)
indexes = np.flatnonzero(_timeOut == _timeOut.min())
print(indexes)
print(_timeOut[int(indexes[0]/5), indexes[0]%5])
print("--------------")
print(np.isin(_timeOut, 0).any())
'''

# filas y luego columnas
A = np.matrix('0 1 2 -1; 3 4 5 -2; 6 7 8 -3')
#print(A.shape)

i = 7
tst = A > 7
print(tst.any())
#print(A[int(i/A.shape[1]), i%A.shape[1]])
#print(np.random.uniform(low=0.5, high=13.3, size=(40, 2)))