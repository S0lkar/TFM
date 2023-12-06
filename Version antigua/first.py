import seaborn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import random;  random.seed(348573985)

#%% ------------------- VARIABLES GLOBALES -------------------
# Suponemos ixj nodos, establecidos en una forma de malla.
TAM_SIM  = (20, 10)
# Tamaño de la ventana de show_step()
WIN_SIZE = (13, 5)

# Rango representante de la batería gastada por medición (%).
BAT_DRAIN = (10, 20)#(0.8, 1.5) #(0.8, 2.5)

OPC = True

# Cada celda tiene el 'cansancio' de la batería (100 - %batería). Inicialmente
# suponemos un estado de desgaste bajo.
_bat = np.random.randint(low=0, high=10, size=TAM_SIM)

# Información relativa a la tasa de muestreo (rango de time-out y estado actual)
T_LIMIT  = [10, 20]#[0.5*60, 20*60] # Min == 10 min, Max == 20 min. Tiempo en segundos.
_timeOut = np.matrix('0 0 0 0 0; 0 0 0 0 0; 0 0 0 0 0; 0 0 0 0 0; 0 0 0 0 0') 

# Instante actual de la simulación
_T = 0



#%% ------------------- FUNCIONES IMPLEMENTADAS -------------------

def move_figure(f, x, y):
    """_summary_
    
    > Precondición: Ninguna.
    > Postcondición: Fuerza a que la figura f se desplaze a las coordenadas (x,y)
    de la pantalla.
    """
    backend = matplotlib.get_backend()
    if backend == 'TkAgg':
        f.canvas.manager.window.wm_geometry("+%d+%d" % (x, y))
    elif backend == 'WXAgg':
        f.canvas.manager.window.SetPosition((x, y))
    else:
        f.canvas.manager.window.move(x, y)

def show_step():
    """_summary_
    
    > Precondición: Ninguna.
    > Postcondición: Se muestra una ventana con información relativa al estado
    de la simulación en el instante _T actual. La ejecución es bloqueante, finaliza
    al cerrar la ventana.
    """
    global _bat, _timeOut, T_LIMIT, _T, WIN_SIZE
    plt.close('all')
    fig, ax = plt.subplots(1, 2, figsize=WIN_SIZE)
    seaborn.heatmap(_bat, vmin=0, vmax=100, annot=True, ax=ax[0]).set_title('Batería')
    seaborn.heatmap(_timeOut, vmin=T_LIMIT[0], vmax=T_LIMIT[1], annot=True, ax=ax[1]).set_title('Time-out')
    print('\n'*10, "\tTIEMPO DE SIMULACIÓN:\t", _T)
    move_figure(fig, 50, 50)
    plt.show()


def time_forward(nodo_i):
    """_summary_

    > Precondición: nodo_i es un array de los nodos con el mínimo tiempo de espera.
    > Postcondición: Avanza T_ y actualiza el timeOut de todos los nodos hasta el instante
    en el que se se activa aquel nodo con la menor espera. Devuelve un array con los nodos
    de espera mínima
    """
    global _T, _timeOut
    
    # Lo localizamos en la malla y observamos su tiempo de espera.
    t_espera = _timeOut[int(nodo_i[0]/TAM_SIM[1]), nodo_i[0]%TAM_SIM[1]]
    
    # Avanzamos hasta dicha posición en el tiempo y actualizamos los nodos.
    _T += t_espera 
    _timeOut -= t_espera
    
    # Devuelve el array de nodos cuya espera sea la mínima.
    # Aunque...necesitaría asignar el tiempo de espera antes de esto...
    return np.flatnonzero(_timeOut == _timeOut.min())


def battery_substraction(nodo_i):
    """_summary_

    > Precondición: nodo_i es un array de los nodos que han realizado una medida.
    > Postcondición: substrae una cantidad aleatoria de energía de las baterías de los nodos
    que realizaron una medición.
    """
    global _bat, TAM_SIM, BAT_DRAIN
    for i in nodo_i:
        _bat[int(i/TAM_SIM[1]), i%TAM_SIM[1]] += round(random.uniform(BAT_DRAIN[0], BAT_DRAIN[1]), 2)
        if _bat[int(i/TAM_SIM[1]), i%TAM_SIM[1]] > 100:
            _bat[int(i/TAM_SIM[1]), i%TAM_SIM[1]] = 100
    

def init_timeOut():
    """_summary_

    > Precondición: Ninguna.
    > Postcondición: Inicializa _timeOut con el estado inicial de las baterías.
    """
    global _bat, _timeOut, T_LIMIT
    
    '''
    Le resto el mínimo para que al interpolar decirle a las baterias menos cansadas
    que tienen que trabajar a tope, no vale que por estar al 80% del cansancio que
    se empiecen a relajar.
    '''
    
    min_cansancio = _bat.min()
    _copiaBat = _bat - min_cansancio   # Copia de _bat con el rango para interpolar
    _postRep = np.zeros(_copiaBat.shape) # Estado posterior al reparto
    MAX_C = 100 - min_cansancio # Rango de cansancios en la batería (0, 100-min_c)
    
    # Reparto del cansancio para los nodos interiores.
    for i in range(1, TAM_SIM[0]-2):
        for j in range(1, TAM_SIM[1]-2):
            c = _copiaBat[i, j] / 9
            _postRep[i, j] += c
            _postRep[i, j+1] += c
            _postRep[i+1, j] += c
            _postRep[i, j-1] += c
            _postRep[i-1, j] += c
            _postRep[i+1, j-1] += c
            _postRep[i-1, j+1] += c
            _postRep[i-1, j-1] += c
            _postRep[i+1, j+1] += c
    
    # Para los bordes voy a hacer que no tengan fatiga, para forzar su uso continuado.
    # Reparto del cansancio para los nodos superiores e inferiores
    for j in range(TAM_SIM[1]):
        _postRep[0, j] = _copiaBat[0, j] / 9 # Superior
        _postRep[TAM_SIM[0]-1, j] = _copiaBat[TAM_SIM[0]-1, j] / 9 # Inferior
    
    # Reparto del cansancio para los nodos laterales
    for i in range(TAM_SIM[0]):
        _postRep[i, 0] = _copiaBat[i, 0] / 2 # izquierda
        _postRep[i, TAM_SIM[1]-1] = _copiaBat[i, TAM_SIM[1]-1] / 2 # derecha
    
    # Teniendo el reparto estimado, se aplican los escalados acorde al
    # intervalo de medición mínimo-máximo y el estado de cada batería.
    # En otras palabras, se inicializa _timeOut.
    
    # Se realiza la inicialización de _timeOut
    _timeOut = ((_postRep / MAX_C) * (T_LIMIT[1]-T_LIMIT[0])) + T_LIMIT[0]
    
    return np.flatnonzero(_timeOut == _timeOut.min()) # Devuelve los nodos que se actualizarán antes


def act_nodo(nodo_i):
    """_summary_

    > Precondición: nodo_i es un array de aquellos nodos despiertos que realizaron una
    medición.
    > Postcondición: actualiza sus tiempos de espera en base a la fatiga propia (bordes)
    y también la de sus vecinos (interiores)
    """
    global _bat, TAM_SIM, _timeOut
    min_cansancio = _bat.min()
    MAX_C = 100 - min_cansancio # Rango de cansancios en la batería (0, 100-min_c)
    
    # Cada batería se hace consciente de su entorno y actúa según la fatiga propia
    # y la vecina.
    for i in nodo_i:
        c = int(i/TAM_SIM[1]); f = i%TAM_SIM[1]
        cansancio = 0
        # Es una celda de los bordes
        if c == 0 or c == TAM_SIM[0]-1 or f == 0 or f == TAM_SIM[1]-1:
            cansancio = (_bat[c, f] - min_cansancio) / 9 # Igual que en init_timeOut
        else: # Es una celda interior
            cansancio = (_bat[c, f] - min_cansancio)/9+(_bat[c, f+1] - min_cansancio)/9
            +(_bat[c+1, f] - min_cansancio)/9 +(_bat[c, f-1] - min_cansancio)/9
            +(_bat[c-1, f] - min_cansancio)/9+(_bat[c+1, f-1] - min_cansancio)/9
            +(_bat[c-1, f+1] - min_cansancio)/9+(_bat[c-1, f-1] - min_cansancio)/9
            +(_bat[c+1, f+1] - min_cansancio)/9
        # Se actualizan de la misma forma que al inicio
        _timeOut[c, f] = ((cansancio / MAX_C) * (T_LIMIT[1]-T_LIMIT[0])) + T_LIMIT[0]
    
    return np.flatnonzero(_timeOut == _timeOut.min()) # Devuelve los nodos que se actualizarán antes
        


#%% ------------------- MAIN -------------------
if __name__ == '__main__':
    # Asignación inicial según la batería
    ids = init_timeOut()
    show_step()
    
    # Evolución de la simulación
    if OPC: # Con gestión de energía
        while not np.isin(_bat, 100).any(): # Cuanto una batería se gaste, se para la ejecución.
            ids = time_forward(ids)
            battery_substraction(ids)
            ids = act_nodo(ids)
            show_step()
    else: # Sin gestión de energía
        _bat = _bat.astype('float64')
        tst = _bat >= 100
        while not tst.any():
            _bat += np.random.uniform(low=BAT_DRAIN[0], high=BAT_DRAIN[1], size=_bat.shape)
            tst = _bat >= 100
        
    

    show_step()
    print('\n'*10, "\t\t.: SIMULACIÓN TERMINADA EN T =", _T, ':.\n\n')
    
    print('> ' + str(_bat.mean()))