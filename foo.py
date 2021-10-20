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
mean = np.array(foo.ravel())
mean = mean[mean > 0]
mean = np.mean(mean)

mask_foo = foo.copy() > mean
lungs = mfdfa.get_lungs(foo)
good_lungs = mfdfa.get_lungs(mask_foo)
union_lungs = mfdfa.union_image_by_graph(good_lungs)
t2 = time() - ti
print(t2)


fig, axes = plt.subplots(2,2, sharex=True, sharey=True)
ax = axes.ravel()
ax[0].imshow(mfdfa.get_lungs( equalize_adapthist(img, clip_limit=0.02)))
ax[1].imshow(lungs, cmap='gray' )
ax[2].imshow(good_lungs)
ax[3].imshow(union_lungs)

ax[0].axis(False)
ax[1].axis(False)
ax[2].axis(False)
ax[3].axis(False)

fig.show()



#print(np.where(np.sum( foo > 0 , axis=0) != 0)[0])
#print(np.unique(foo.ravel()))
