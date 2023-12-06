import seaborn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import random;  random.seed(348573985)



#%% ------------------- GLOBAL SETTINGS  -------------------
# Show_step() window size
WIN_SIZE = (13, 5)

# Size of the field (board)
FIELD_SIZE = (40, 40)


#%% ------------------- DATA STRUCTURES  -------------------

_Board = ''
class Area():
    RoE = 0 # Radio of Effect of each node
    NodesCoords = [] # List of the coordinates where each node is located.
    Nodes = '' # List of the % of battery of each node.
    Defined_Areas = [] # List of pointers to each defined area
    
    def __init__(self, nodes_assigned):
        '''
        An area is defined by the cells it englobes and by which nodes can be activated.
        '''
        self.cells = [] # list of cells that compose the area. Each cell defined by (i,j) coords (tuples for better performance).
        self.nodes_assigned = nodes_assigned # nodes that can measure the area
        Area.Defined_Areas.append(self) # registers this area into the list
        
    @staticmethod
    def check_existence(nodes_needed):
        for i in Area.Defined_Areas:
            if nodes_needed == i.nodes_assigned:
                return i
        return None
    
    def paint_area(self):
        '''
            Just for testing pourposes. Prints in the board the area (by inserting the nodes that represent it).
            BEWARE: If the area is an intersection between node 0 and any other, it will NOT show that first '0'
            due to the board being an integer matrix.
        '''
        global _Board
        m = ''.join(map(str, self.nodes_assigned))
        for i in self.cells:
            _Board[i] = m
    
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
                    



#%% ------------------- MAIN -------------------
import time

if __name__ == '__main__':
    load_instance('./Correccion/Settings.txt')
    #print(_Board)
    #print(Area.NodesCoords)
    #print(Area.Nodes)
    #print(Area.Defined_Areas)
    
    for i in Area.Defined_Areas:
        i.paint_area()
        print(i.nodes_assigned)
    print(_Board)
    
    pass