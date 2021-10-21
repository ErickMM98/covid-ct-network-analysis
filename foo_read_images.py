import matplotlib.pyplot as plt
import numpy as np
import os

path_image_normal = 'D:\\servicio\\binary_images\\binary_arrays\\normal\\1865\\images'

list_img = os.listdir(path_image_normal)[0]

img = np.load( os.path.join( path_image_normal, list_img ) )

plt.imshow(img)
plt.show()