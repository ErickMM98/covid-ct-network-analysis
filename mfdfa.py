#@title MFDFA main
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 16:12:49 2021

@author: Erick Muñiz Morales

Modules to image segmentation by multifractal detrendted
fluctiation analysis.


"""

import numpy as np
import tqdm
import threading


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

def make_var_least_squared_cum_matrix(cum_matrix, vectors = False):
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

    A = []
    y = []
    for i in range(N):
        for j in range(N):
            renglon = [i+1,j+1,1]
            A.append(renglon)
            y.append([cum_matrix[i,j]])

    A = np.array(A)
    y = np.array(y)
    #print(A,y)
    alpha = np.dot((np.dot(np.linalg.inv(np.dot(A.T,A)),A.T)),y)

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

def make_mfdfa_individual_window(window, s, q=2):
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

            y_real, y_pred, G_mn_pred = make_var_least_squared_cum_matrix(G_mn, vectors=True)

            y_mn = y_real - y_pred

            f_2mn = np.mean(y_mn**2)

            F_list_q.append(f_2mn)

    F_q = np.mean(np.array(F_list_q)**(q/2))**(1/q)

    #print(F_q)

    return F_q


def get_h_q_mfdfa(window, min_s = 6, max_s = None, q = 2):
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
        h = make_mfdfa_individual_window(window,s=s, q = 2)
        y_ls.append(h)

    h_q = get_leas_squared_array(np.log(s_ls), np.log(y_ls))

    return h_q[0]



#"""
#------- END PRINCIPAL FUNCTIONS
#"""

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
    def __init__(self, image, q = -10):
        self.image = image

    def run(self):
        """
        The most importan function in the class.
        :return:
        """


class WorkerMFDFA(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)