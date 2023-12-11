import seaborn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import random;  random.seed(348573985)
from functools import lru_cache


#%% ------------------- GLOBAL SETTINGS  -------------------
# Show_step() window size
WIN_SIZE = (13, 5)

# Visualices each step
VERBOSE = False

# Equivalent between simulation steps and time
STEP_TIME_RATE = 0.5 # each step equals to 0.5 TU

# Slope of the Gaussian function used to calculate uncertainty over time
c = 3

# The threshold used in the necessity criterion, (max_ImT, max_ImA).
THRESHOLD = (0.6, 0.9)


#%% ------------------- DATA STRUCTURES & TIME -------------------
_T = 0
_Board = ''

def Decay():
    global _Board, _T; _Board += 1; _T += 1

# Steps2Uncertainty(_Board[(i, j)])
@lru_cache(maxsize=None)
def Steps2Uncertainty(n_steps):
    t = n_steps * STEP_TIME_RATE
    return np.e ** -((t**2) / (2*(c**2)))


class Area():
    RoE = 0 # Radio of Effect of each node
    NodesCoords = [] # List of the coordinates where each node is located.
    Nodes = [] # List of the % of battery of each node.
    NodesRelations = [] # List of areas that a node is associated with.
    Defined_Areas = [] # List of pointers to each defined area
    
    def __init__(self, nodes_assigned):
        '''
        An area is defined by the cells it englobes and by which nodes can be activated.
        '''
        self.cells = [] # list of cells that compose the area. Each cell defined by (i,j) coords (tuples for better performance).
        self.nodes_assigned = nodes_assigned # nodes that can measure the area
        Area.Defined_Areas.append(self) # registers this area into the list
        self.Umean = 0 # mean of all uncertainty values of this area.
        self.Umax = 0 # maximum of all uncertainty values of this area.
        for i in nodes_assigned: # Add the pointer of the current area to each of the nodes assigned
            Area.NodesRelations[i].append(self)
        
    @staticmethod
    def check_existence(nodes_needed):
        for i in Area.Defined_Areas:
            if nodes_needed == i.nodes_assigned:
                return i
        return None
     
    def Uncertainty(self): # Updates all relevant statistics of Uncertainty
        list_of_Uvalues = []
        for i in self.cells:
            list_of_Uvalues.append(Steps2Uncertainty(_Board[i]))
        # return mean(list), max(list)
        self.Umean = sum(list_of_Uvalues) / len(list_of_Uvalues)
        self.Umax  = max(list_of_Uvalues)
        return
    
    # Local to each area. It is possible to define a global one.
    def Necessity_Criterion(self):
        return (self.Umean > THRESHOLD[0]) or (self.Umax > THRESHOLD[1])
    
    # Can be remodeled to consider a higher effectiveness on the node's coords and gradually
    # worse removal of uncertainty with distance (we would need to change the Board datatype to float)
    def Activate(self):
        global _Board
        for i in self.cells:
            _Board[i] = 0 # Measures the area, clearing all uncertainty within.
        return 
    
    def paint_area(self):
        '''
            Just for testing pourposes. Prints in the board the area (by inserting the nodes that represent it).
            BEWARE: If the area is an intersection between node 0 and any other, it will NOT show that first '0'
            due to the board being an integer matrix.
        '''
        global _Board
        m = ''.join(map(str, self.nodes_assigned))
        for i in self.cells:
            _Board[i] += 1
            
    
#%% ------------------- FUNCTIONS  -------------------

def dist_eu(a, b):
    '''
        Returns the euclidean distance between numpy arrays a, b.
        Far more efficient than the regular formula. This can be made slightly faster by not performing the assert.
    '''
    assert a.shape == b.shape, "vectors must be of equal shape."
    return np.linalg.norm(a - b)

# UPGRADE: We could use an image instead of plain text for better manipulation.
def load_instance(Filename):
    '''
        Initializes Area class with all of the areas and nodes that exists in the instance defined in Filename.
        Also initializes the board as defined in Filename.
        The structure of filename is as follows;
            radius of effect
            list of coordinates of each node, in the format 'x,y'
            separator (~)
            matrix representing the initial state of the field, with the same format that np.matrix accepts
    '''
    global _Board
    f = open(Filename, 'r')
    
    # -- Reading of radius of each node --
    RoE = float(f.readline()[0:-1])
    Area.RoE = RoE
    
    # -- Reading of the coordinates of each node --
    coord = f.readline()
    count = 0
    list_ = []
    while coord != '~\n':
        list_.append(tuple(map(int, coord[0:-1].split(','))))
        coord = f.readline()
        count += 1
    Area.NodesCoords = list_ # Stores the list of coordinates
    Area.Nodes = np.zeros(count, 'int8') # Assuming all batteries on 0% used energy
    Area.NodesRelations = [[] for _ in range(len(Area.Nodes))]
    
    # Reading of the initial state of the board
    m = ''
    for i in f.readlines():
        m += i
    f.close()
    _Board = np.matrix(m)
    
    # For each cell we must check if it is in bounds of any node. If that is the case, a new area must be defined (if needed)
    # to include that cell.
    
    # x -> rows, y -> cols, n -> nodes. Assuming (x == y >> n);
    # O(x*y*n) ~= O(x^2)
    
    for i in range(_Board.shape[0]):
        for j in range(_Board.shape[1]):
            nodes_needed = []
            for k in range(len(list_)):
                if dist_eu(np.array((i, j)), np.array(list_[k])) <= RoE: # Are coords (i, j) in bounds of node k?
                    nodes_needed.append(k)
            
            if nodes_needed: # After checking all k nodes, is there any that covers this cell?
                node = Area.check_existence(nodes_needed)
                if node: # Is there already an Area covered by the same nodes?
                    node.cells.append((i, j)) # If so, we join this cell into that previously defined area
                else:
                    a = Area(nodes_needed) # We define such new area
                    a.cells.append((i, j))

               
# ar is a list of lists of values.
# ex: ar = [[1], [2], [1, 2], [2, 3], [4, 5, 6], [5, 7]]    ->  SAT = [[1], [2], [4, 5, 6], [5, 7]]
def Get_independent_sets(ar):
    SAT = []
    for i in range(len(ar)):
        include = True
        for j in SAT:
            if set(j).issubset(ar[i]): # is j a subset of i?
                include = False
                break
        if include:   
            for j in range(i+1,len(ar)):
                if set(ar[j]).issubset(ar[i]): # is j a subset of i?
                    include = False
                    break
        if include:
            SAT.append(ar[i])
    return SAT

# Must also return if a battery has reaches 100%.
# Must also turn on the areas associated with the activated nodes
def Activation_Criterion(active_areas):
    # Get list of nodes per area
    # [[], [], [], []]...
    SAT = []
    for i in active_areas:
        SAT.append(i.nodes_assigned)
    
    # Filter the list, having only independent sets which need to be activated
    SAT = Get_independent_sets(SAT)
    
    # get all combinations that solve the situation (SAT jr solver)
    # nodes: {1,2,3,4,...,i}
    nodes_activated = []
    
    # Per combination, there needs to be a procedure to choose the best one
    
    
    # per node, activate all areas that it is associated with
    for i in Area.Defined_Areas:
        nodes = i.nodes_assigned
        # Ahora ver si en los nodos asignados al area i figura cualquiera de los activados
        # puedo concatenar las listas y si hay menos valores únicos que elementos, voilà
    
    # take out those nodes' batteries.
    for i in nodes_activated:
        Area.Nodes[i] += 1 # consumes 1% of battery on usage
        if Area.Nodes[i] == 100:
            return False
    
    return True


def Visualize_Contour():
    aux = np.array(_Board * STEP_TIME_RATE)
    aux = np.power(np.e, -(np.power(aux, 2) / (2*(c**2))))

    fig, ax = plt.subplots()
    CS = ax.contour(aux, origin=None, levels=len(np.unique(aux))-1)
    ax.clabel(CS, inline=True, fontsize=10)
    ax.set_title('Beta')
    plt.show()
    return



#%% ------------------- MAIN -------------------

#TODO: I may want to print the metrics; final status and through time.
if __name__ == '__main__':
    load_instance('./Correccion/Settings.txt')
    for i in Area.Defined_Areas:
        print(i.nodes_assigned)
    print(Area.NodesRelations)
    
    '''
    all_batteries_charged = True
    while all_batteries_charged
        areas_activated = []
        Area.Reset_Nodes_Stats() # Each node will have its statistics updated
        for i in Area.Defined_Areas:
            i.Uncertainty() # Updates relevant statistics values
            if i.Necessity_Criterion():
                areas_activated.append(i)
        
        if areas_activated:
            all_batteries_charged = Activation_Criterion(areas_activated)
            
        if VERBOSE:
            Visualize_Contour()
            
        Decay()
    '''
    Visualize_Contour()