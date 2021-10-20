#@title MFDFA main
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 16:12:49 2021

@author: Erick Muñiz Morales

Modules to image segmentation by multifractal detrendted
fluctiation analysis.


"""
from skimage.exposure import equalize_adapthist
import numpy as np
import tqdm
import threading
import networkx as nx

#inversa = None


"""
------- BEGIN MODULAR FUNCTIONS
"""

def get_windows_size_w_by_pixel(img, i,j,w=11):
    """


    Parameters
    ----------
    img : TYPE
        DESCRIPTION.
    i : TYPE
        DESCRIPTION.
    j : TYPE
        DESCRIPTION.
    w : TYPE, optional
        DESCRIPTION. The default is 11.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    N = img.shape[0]
    M = img.shape[1]
    min_x = max([ i-w, 0 ])
    max_x = min([ i+w, N ])

    min_y = max([ i-w, 0 ])
    max_y = min([ i+w, M ])

    return img[min_x:max_x, min_y:max_y]


def get_window_size_s(img,i,j,s):
    """
    Parameters
    ----------
    img : skimae.io.imread (ndarray); the whole image
        the whole image
    s : size of the window of fluctiation
        DESCRIPTION.
    i : row index
        is less of N_s
    j : column index
        is less of M_s

    Returns
    -------
    TYPE
        The image window of size s with index i,j in the new fluctuation.

    """
    min_x = i*s
    max_x = (i+1)*s
    min_y = j*s
    max_y = (j+1)*s
    return img[min_x:max_x, min_y:max_y]

def get_cumulative_matrix(img):
    """
    Get the sum cumulative matrix of the squared matrix (img).

    Parameters
    ----------
    img : Matrix of values.
        DESCRIPTION.

    Returns
    -------
    None.

    """
    N = img.shape[0]

    matrix_ones = np.ones((N,N))

    rigth_matrix = np.triu(matrix_ones)
    left_matrix = np.tril(matrix_ones)

    return left_matrix @ img @ rigth_matrix

def make_var_least_squared_cum_matrix(cum_matrix,matrix_a = None, inverse_ls = None, vectors = False):
    """
    A linear regresion of the window

    Parameters
    ----------
    cum_matrix : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """

    N = cum_matrix.shape[0]

    inverse_matrix = None

    A = []
    y = []

    for i in range(N):
        for j in range(N):
            renglon = [i + 1, j + 1, 1]
            A.append(renglon)
            y.append([cum_matrix[i, j]])

    A = np.array(A)
    y = np.array(y)

    inverse_matrix = inverse_ls

    alpha = np.dot( inverse_matrix , y)
    #alpha = np.dot((np.dot(np.linalg.inv(np.dot(A.T,A)),A.T)),y)

    y_pred = A @ alpha
    y_real = y

    if vectors:
        return y_real.ravel(), y_pred.ravel(), y_pred.reshape((N,N))
    else:
        return np.mean((y_real.ravel() - y_pred.ravel())**2)

def measure_fractal(array_var, q = 2):
    """
    For the other measures.

    Parameters
    ----------
    array_var : TYPE
        DESCRIPTION.
    q : TYPE, optional
        DESCRIPTION. The default is 2.

    Returns
    -------
    None.

    """
    array_var = np.array(array_var)
    return np.mean( array_var**(q/2) )**(1/q)


def get_leas_squared_array(x,y):
    """
    The most simple least squered of two arrays. The model is in the form
                y = ax + b
    Parameters
    ----------
    x : TYPE
        DESCRIPTION.
    y : TYPE
        DESCRIPTION.

    Returns
    -------
    a,b in the above formula.

    """

    x = np.array(x).reshape((len(x),1))
    y = np.array(y).reshape((len(y),1))
    A = np.concatenate( [x, np.ones((x.shape[0],1))] , axis=1)


    h = np.dot((np.dot(np.linalg.inv(np.dot(A.T,A)),A.T)),y)

    return h.ravel()


"""
------- END MODULAR FUNCTIONS
"""

"""
------- BEGIN PRINCIPAL FUNCTIONS
"""

def make_mfdfa_individual_window(window, s, q=2, dict_inverse = None):
    """


    Parameters
    ----------
    window : TYPE
        DESCRIPTION.
    s : TYPE
        DESCRIPTION.
    q : TYPE, optional
        DESCRIPTION. The default is 2.

    Returns
    -------
    Fluctuation

    """

    N,M = window.shape

    N_s = int(N / s)
    M_s = int(M / s)


    F_list_q = []
    for i in range(N_s):
        for j in range(M_s):
            X_mn = get_window_size_s(window, i, j, s)

            G_mn = get_cumulative_matrix(X_mn)

            inverse_least_squared = dict_inverse[s]

            y_real, y_pred, G_mn_pred = make_var_least_squared_cum_matrix(G_mn,
                                                                          vectors=True,
                                                                          inverse_ls = inverse_least_squared)

            y_mn = y_real - y_pred

            f_2mn = np.mean(y_mn**2)

            F_list_q.append(f_2mn)

    F_q = np.mean(np.array(F_list_q)**(q/2))**(1/q)

    return F_q


def get_h_q_mfdfa(window, min_s = 6, max_s = None, q = 2, dict_inverse = None):
    """
    Esta es la que debemos paralelizar

    Parameters
    ----------
    window : TYPE
        DESCRIPTION.
    min_s : TYPE, optional
        DESCRIPTION. The default is 6.
    max_s : TYPE, optional
        DESCRIPTION. The default is None.
    q : TYPE, optional
        DESCRIPTION. The default is 2.

    Returns
    -------
    None.

    """
    s_ls = None
    y_ls = []

    N,M = window.shape

    if max_s == None:
        s_ls = range(min_s, int(min([N,M])/2) + 1 ,2)
    else:
        s_ls = range(min_s, max_s + 1 )

    for s in s_ls:
        #print(s)
        h = make_mfdfa_individual_window(window,
                                         s=s,
                                         q = 2,
                                         dict_inverse = dict_inverse)
        y_ls.append(h)


    h_q = get_leas_squared_array(np.log(s_ls), np.log(y_ls))

    return h_q[0]



#"""
#------- END PRINCIPAL FUNCTIONS
#"""

"""
------------------- ------------------- ------------------- ------------------- 
                            BEGIN - TRANFORM IMAGES 
------------------- ------------------- ------------------- -------------------
"""


def make_graph_segmentation(binaryimg, k=1, lmd=3.5):
    N, M = binaryimg.shape
    foo = binaryimg.copy()
    s, t = N * M, N * M + 1
    G = nx.DiGraph()
    # Aqui generamos la red
    for idx, row in enumerate(binaryimg):
        for jdx, pixel in enumerate(row):
            # Polinomio de redireccionamiento
            px_id = (idx * len(row)) + jdx

            # Están arriba
            if idx == 0:  # or idx == N-1:
                # equina izquierda
                if jdx == 0:  # or jdx == M-1:
                    source = px_id
                    # Derecha
                    target = (idx * len(row)) + (jdx + 1)
                    G.add_edge(source, target, weight=k)
                    # Abajo
                    target = ((idx + 1) * len(row)) + jdx
                    G.add_edge(source, target, weight=k)
                # esquina Derecha
                elif jdx == M - 1:
                    source = px_id
                    # Izquierda
                    target = (idx * len(row)) + (jdx - 1)
                    G.add_edge(source, target, weight=k)
                    # Abajo
                    target = ((idx + 1) * len(row)) + jdx
                    G.add_edge(source, target, weight=k)
                # En medio
                else:
                    source = px_id
                    # Derecha
                    target = (idx * len(row)) + (jdx + 1)
                    G.add_edge(source, target, weight=k)
                    # Izquierda
                    target = (idx * len(row)) + (jdx - 1)
                    G.add_edge(source, target, weight=k)
                    # Abajo
                    target = ((idx + 1) * len(row)) + jdx
                    G.add_edge(source, target, weight=k)
            # Hasta abajo
            elif idx == N - 1:
                # equina izquierda
                if jdx == 0:  # or jdx == M-1:
                    source = px_id
                    # Derecha
                    target = (idx * len(row)) + (jdx + 1)
                    G.add_edge(source, target, weight=k)
                    # Arriba
                    target = ((idx - 1) * len(row)) + jdx
                    G.add_edge(source, target, weight=k)
                # esquina Derecha
                elif jdx == M - 1:
                    source = px_id
                    # Izquierda
                    target = (idx * len(row)) + (jdx - 1)
                    G.add_edge(source, target, weight=k)
                    # Arriba
                    target = ((idx - 1) * len(row)) + jdx
                    G.add_edge(source, target, weight=k)
                # En medio
                else:
                    source = px_id
                    # Derecha
                    target = (idx * len(row)) + (jdx + 1)
                    G.add_edge(source, target, weight=k)
                    # Izquierda
                    target = (idx * len(row)) + (jdx - 1)
                    G.add_edge(source, target, weight=k)
                    # Arriba
                    target = ((idx - 1) * len(row)) + jdx
                    G.add_edge(source, target, weight=k)
            # En medio
            else:
                source = px_id

                # Derecha
                target = (idx * len(row)) + (jdx + 1)
                G.add_edge(source, target, weight=k)

                # Izquierda
                target = (idx * len(row)) + (jdx - 1)
                G.add_edge(source, target, weight=k)

                # Arriba
                target = ((idx - 1) * len(row)) + jdx
                G.add_edge(source, target, weight=k)

                # Abajo
                target = ((idx + 1) * len(row)) + jdx
                G.add_edge(source, target, weight=k)

            # Asociamos con s,t

            if pixel > 0:
                source = s
                target = px_id
                G.add_edge(source, target, weight=lmd)
            else:
                source = px_id
                target = t
                G.add_edge(source, target, weight=lmd)
    return G, s, t

def get_lungs(fooimg):

    col_index = np.where(np.sum(fooimg, axis=0) != 0)[0]
    row_index = np.where(np.sum(fooimg, axis=1) != 0)[0]

    min_row = row_index[0]
    max_row = row_index[-1]

    min_col = col_index[0]
    max_col = col_index[-1]

    return fooimg[min_row:max_row, min_col:max_col]

def union_image_by_graph(lung_image):
    k = 1
    lmd = 2.5
    binaryimg = lung_image.copy()
    N, M = binaryimg.shape
    G, s, t = make_graph_segmentation(binaryimg, k=k, lmd=lmd)
    cvalue, partition = nx.minimum_cut(G, _s=s, _t=t, capacity='weight')
    real_pixels = np.array(list(partition[0]))
    array_pixels = np.arange(s)
    segment_img = np.isin(array_pixels, real_pixels).reshape(N, M)
    return segment_img


"""
------------------- ------------------- ------------------- ------------------- 
                            BEGIN - TRANFORM IMAGES 
------------------- ------------------- ------------------- -------------------
"""

def get_h_image(img, q = 2):
    """


    Parameters
    ----------
    img : TYPE
        DESCRIPTION.
    q : TYPE, optional
        DESCRIPTION. The default is 2.

    Returns
    -------
    None.

    """

    s_ls = range(6, 17 + 1 )
    y_ls = []
    for s in s_ls:
        #print(s)
        h = make_mfdfa_individual_window(img,s=s, q = 2)
        y_ls.append(h)

    h_q = get_leas_squared_array(np.log(s_ls), np.log(y_ls))

    return h_q[0]

class MFDFAImage():
    def __init__(self, image, q = -10, nworkers = 3, windowsize = 2):
        """

        :param image: The original image to segment.
        :param q: The value of the multifractal.
        :param nworkers: Numbers of threads.
        """
        self.image = image
        self.nworkers = nworkers
        self.image_transform = equalize_adapthist( image ,clip_limit=0.02 )
        self.q = q
        self.windowsize = windowsize
        self.inversematrix = None
        self.dict_inverses = {}

        for s in range(2, 4):
            A = []
            for i in range(s):
                for j in range(s):
                    renglon = [i + 1, j + 1, 1]
                    A.append(renglon)
            A = np.array(A)
            A = (np.dot(np.linalg.inv(np.dot(A.T, A)), A.T))
            self.dict_inverses[s] = A

    def run(self):
        """
        Hacemos la rutina sobre los hilos.

        :return :
        """
        N, M = self.image.shape
        #step = int(N/self.nworkers)

        mfdfa_image = None
        mfdfa_image = np.zeros((N,M))

        list_workers = []
        list_index_lung = []

        for i in range(N):
            for j in range(M):
                if self.image[i,j] > 0:
                    list_index_lung.append( (i,j) )

        n_list = len(list_index_lung)
        step = int( n_list / self.nworkers )

        for w in range(self.nworkers):
            list_worker = None
            if w == self.nworkers - 1:
                min_w = step * w
                max_w = n_list
                list_worker = list_index_lung[min_w:max_w]
            else:
                min_w = step * w
                max_w = step * (w+1)
                list_worker = list_index_lung[min_w:max_w]

            worker = WorkerMFDFA(orginal_image=self.image_transform,
                                 final_image=mfdfa_image,
                                 list_worker=list_worker,
                                 windowsize=self.windowsize,
                                 q=self.q,
                                 id= w,
                                 inversematrix = self.dict_inverses)
            worker.start()
            list_workers.append(worker)

        for worker in list_workers:
            worker.join()

        return mfdfa_image

class WorkerMFDFA(threading.Thread):
    def __init__(self,orginal_image,final_image, list_worker, id, windowsize = 2,q = -10,inversematrix = None ):
        self.original_image = orginal_image
        self.final_image = final_image
        self.list_worker = list_worker
        self.id = id
        self.windowsize = windowsize
        self.q = q
        self.inversematrix = inversematrix
        threading.Thread.__init__(self)


    def run(self):
        """

        :return:
        """
        N,M = self.final_image.shape

        min_x = 0
        max_x = 0
        min_y = 0
        max_y = 0

        for i,j in self.list_worker:
            min_x = max([i - self.windowsize, 0])
            max_x = min([i + self.windowsize, N])
            min_y = max([j - self.windowsize, 0])
            max_y = min([j + self.windowsize, M])

            h = get_h_q_mfdfa(self.original_image[min_x:max_x, min_y:max_y] + 1,
                                 min_s=2,
                                 max_s=3,
                                 q=self.q,
                                 dict_inverse = self.inversematrix)

            self.final_image[i,j] = h

