from skimage.io import imread
from skimage.exposure import equalize_adapthist
import matplotlib.pyplot as plt
import mfdfa
import numpy as np
from time import time

"""
GENERAL VALUES
"""
q = -10
nworkers = 10


img = imread('images/0018.jpg')


parallel_mfdfa = mfdfa.MFDFAImage(image=img,nworkers=nworkers)
ti = time()
foo = parallel_mfdfa.run()
t2 = time() - ti
print(t2)

fig, ax = plt.subplots()
ax.imshow(foo)
ax.axis(False)
fig.show()

#print(np.unique(foo.ravel()))

