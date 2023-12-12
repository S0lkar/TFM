
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






#ar = [[0], [1], [2], [3,4,5], [4,6]]
SAT = []
def satin():
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

#print(SAT)
#print(bool([1] in [1,2]))
#print(set([1]).issubset(set([1, 2])))

rel = np.matrix("1 0 0 0 0 0 0; 0 1 0 0 0 0 0; 0 0 1 1 0 0 0; 0 0 1 1 0 0 0; 0 0 0 0 1 0 0; 0 0 0 0 0 1 0; 0 0 0 0 0 0 1")

class A():
    ptr = 0
    def __init__(self, name):
        self.name = name
        if A.ptr == 0:
            A.ptr = self
            
    def __str__(self):
        return self.name
            
a = A('a')
b = A.ptr
c = A('c')
#print({a, b, c})
#print(set([1,2,3,4,3]))

# -------------------------------------
# pip install python-sat
from pysat.solvers import Glucose3
ar = [[2, 3], [1]]
g = Glucose3()
for i in ar:
    g.add_clause(i)

for i in g.enum_models():
    print(i)
    
    
ar = [[1], [2], [3]]
#g = Glucose3()
g.__init__() # Tengo que 'regargar' despues de cada SATeada

for i in ar:
    g.add_clause(i)

for i in g.enum_models():
    print(i)
    

print('-----------') 
import copy
# TO PROD AS IS
SAT_SOLVER = Glucose3()
def Solve_SAT(ar):
    sol = []
    for i in ar:
        SAT_SOLVER.add_clause(i)

    for i in SAT_SOLVER.enum_models():
        sol.append(i)
    
    SAT_SOLVER.__init__()
    return sol

def Activation_Combinations(list_cover): # list of nodes assign to each area in need. [[],[],[],...]
    d = dict() # contains a hash to rename the nodes' IDs. {real_value: virtual_value}
    cont = 1
    for i in list_cover:
        for j in i:
            if j not in d.keys():
                d[j] = cont
                cont += 1

    clause = copy.deepcopy(list_cover) # The clause that will be solved, using the renamed nodes.
    
    for i in range(len(clause)):
        for j in range(len(clause[i])):
            clause[i][j] = d[clause[i][j]]
            
    d = {v: k for k, v in d.items()} # Inverse of the map. {virtual_value: real_value}
    sol_sat = Solve_SAT(clause) # List of all combinations which satisfy the necessity.

    candidates = [] # list of candidate combinations.
    for sol in sol_sat:
        act = []
        for i in sol:
            if i > 0: # This node is set to true (it needs to be activated)
                act.append(d[i]) # We append the real value to know which node it is.
        candidates.append(act)
    return candidates

print(Activation_Combinations([[0], [23], [1, 2], [2, 3], [4, 5, 6], [5, 7], [4, 7]]))
print("--------------------------------")

clause = [[0], [23], [1, 2], [2, 3], [4, 5, 6], [5, 7], [4, 7]]
d = dict()
cont = 1
for i in clause:
    for j in i:
        if j not in d.keys():
            d[j] = cont
            cont += 1

import copy

clause_aux = copy.deepcopy(clause)
for i in range(len(clause_aux)):
    for j in range(len(clause_aux[i])):
        clause_aux[i][j] = d[clause_aux[i][j]]

d = {v: k for k, v in d.items()} # Inverse of the map
#print(clause, " - ", clause_aux)
#print("\n" + str(d))
sol_sat = Solve_SAT(clause_aux)

sol = sol_sat[0]
#print(sol)
activar = []
for i in sol:
    if i > 0: # Es un nodo activado de la combinacion
        activar.append(d[i])

#print("ACTIVAR: " + str(activar))
'''
index = set()
for i in clause:
    for j in i:
        index.add(j)

index = list(index)
print(index)
print(index[8])
'''

print(":------------------------:")
marks = [0,1,2,3,4,5,10, 10]

print(marks.index(max(marks)))