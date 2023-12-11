
import matplotlib.pyplot as plt
import numpy as np

# Cachear la función si se ejecuta siempre con los mismos valores
f = lambda x, c : np.e ** -((x**2) / (2*(c**2)))

c = 3
l = []
for i in range(10):
    for j in range(0, 10, 5):
        j = j / 10
        l.append(f(i+j, c))
l.append(f(10, c))

x = [i for i in range(len(l))]

#plt.plot(x, l)
#plt.show()
#print(l)
#print(x)


'''
    La matriz siendo de enteros pero yo queriendo que 1 en avance sea 0.5s, pues multiplico
    la cantidad de avance por 0.5, y ya pasarlo por la función.
'''
import matplotlib.cm as cm

# Aqui defino el rango del mapa
delta = 0.025
x = np.arange(-3, 3, 0.5) # np.arange(0, 10.0, 1) 
y = np.arange(-3, 3, 0.5)
X, Y = np.meshgrid(x, y)
Z1 = np.exp(-X**2 - Y**2)
Z2 = np.exp(-(X - 1)**2 - (Y - 1)**2)
Z = (Z1 - Z2) * 2
# --------------------------------
# Y aqui ahora hago el plot como tal
#m = np.matrix('1 2 3 4 5 0 0 0 0 0;1 2 3 4 5 0 0 0 0 0;1 2 3 4 5 0 0 0 0 0;1 2 3 4 5 0 0 0 0 0;1 2 3 4 5 0 0 0 0 0;1 2 3 4 5 0 0 0 0 0;1 2 3 4 5 0 0 0 0 0;1 2 3 4 5 0 0 0 0 0;1 2 3 4 5 0 0 0 0 0;1 2 3 4 5 0 0 0 0 0')
#m = np.random.rand(len(x), len(y))
m = np.zeros((len(x), len(y)))
#m[3, 3] = 1;m[2,2] = 1;m[2, 3] = 1;m[3, 2] = 1
m[3, 0] = 1;m[1, 1] = 1;m[5, 3] = 1
#m[3, 3] = 1;m[2,2] = 1;m[2, 3] = 1;m[3, 2] = 1; m[2, 4] = 1;m[4, 2] = 1; m[3, 4] = 1;m[4, 3] = 1; m[4, 4] = 1
#m[3, 3] = 1;m[2, 3] = 1;m[3, 2] = 1;m[4, 3] = 1;m[3, 4] = 1

fig, ax = plt.subplots()
#CS = ax.contour(1-m, origin=None, corner_mask=True)
CS = ax.contour(X, Y, 1-m, corner_mask=True, algorithm=None, levels=1)
ax.clabel(CS, inline=True, fontsize=10)
ax.set_title('Simplest default with labels')
#print(1-m)
#print(CS)
#plt.show()
#print(Z)

#plt.pcolormesh(1-m)
#plt.show()

def t():
    return 1, 2

a = np.array([1,2,3,4,5, 1, 0])
b = t()
#print(len(np.unique(a)))
#print(b)
#print([1, 2] == [1, 2])


ar = [[1], [2], [1, 2], [2, 3], [4, 5, 6], [5, 7], [4, 7]]
SAT = []
for i in range(len(ar)):
    incluir = True
    for j in SAT:
        if set(j).issubset(ar[i]): # es j un subset de i?
            incluir = False
            break
    if incluir:   
        for j in range(i+1,len(ar)):
            if set(ar[j]).issubset(ar[i]): # es j un subset de i?
                incluir = False
                break
    if incluir:
        SAT.append(ar[i])

print(SAT)
#print(bool([1] in [1,2]))
print(set([1]).issubset(set([1, 2])))