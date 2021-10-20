import os
import mfdfa
from natsort import natsorted, ns
from skimage.io import imread, imsave
from skimage.transform import resize
from skimage.exposure import equalize_adapthist
import numpy as np
import matplotlib.pyplot as plt

normal_path = 'D:\\servicio\\binary_images\\Normal'
covid_path = 'D:\\servicio\\binary_images\\NCP'

path_finally_covid = 'D:\\servicio\\binary_images\\binary_arrays\\covid'
path_finally_normal = 'D:\\servicio\\binary_images\\binary_arrays\\normal'

id_normal = os.listdir(normal_path)
id_covid = os.listdir(covid_path)

nworkers = 15

# ---------------- Iteramos los normales
for id in id_normal[:]:
    path_cts = os.path.join(normal_path, id)

    list_binary_images = []

    path_folder_id = os.path.join(path_finally_normal, id)

    os.makedirs(path_folder_id)
    path_images_id = os.path.join(path_folder_id, 'images')
    os.makedirs(path_images_id)

    final_path_stack = os.path.join(path_folder_id, id)

    for ct in os.listdir(path_cts):
        path_final_cts = os.path.join(path_cts, ct)
        raw_images = os.listdir(path_final_cts)
        N = len(raw_images)
        i = 0
        for img in natsorted(raw_images)[:]:
            print(i, N)
            i += 1
            path_img = os.path.join(path_final_cts, img)
            image = imread(path_img)

            #Hacemos MFDFA  en la imagen.
            parallel_mfdfa = mfdfa.MFDFAImage(image=image,
                                              nworkers=nworkers)
            foo = parallel_mfdfa.run()

            mean = np.array(foo.ravel())
            mean = mean[mean > 0]
            mean = np.mean(mean)

            mask_foo = foo.copy() > mean
            lungs = mfdfa.get_lungs(foo)
            good_lungs = mfdfa.get_lungs(mask_foo)
            union_lungs = mfdfa.union_image_by_graph(good_lungs)


            image_resized = resize(union_lungs, (128, 128),
                                   anti_aliasing=False)
            image_original_resize = resize(mfdfa.get_lungs(image), (128, 128),
                                   anti_aliasing=False)

            image_mfdfa = resize(mfdfa.get_lungs(foo), (128, 128),
                                 anti_aliasing=False)

            list_binary_images.append(image_resized)

            path_final_img = os.path.join(path_images_id,img)

            imsave(path_final_img,
                   image_mfdfa)


    stack_id = np.stack(list_binary_images)

    np.save(final_path_stack,
            stack_id)




# for id in id_covid[:]:
#     path_cts = os.path.join(covid_path, id)
#
#     list_binary_images = []
#
#     path_folder_id = os.path.join(path_finally_covid, id)
#
#     os.makedirs(path_folder_id)
#     path_images_id = os.path.join(path_folder_id, 'images')
#     os.makedirs(path_images_id)
#
#     final_path_stack = os.path.join(path_folder_id, id)
#
#     for ct in os.listdir(path_cts):
#         path_final_cts = os.path.join(path_cts, ct)
#         raw_images = os.listdir(path_final_cts)
#         N = len(raw_images)
#         i = 0
#         for img in natsorted(raw_images)[:]:
#             print(i, N)
#             i += 1
#             path_img = os.path.join(path_final_cts, img)
#             image = imread(path_img)
#
#             #Hacemos MFDFA  en la imagen.
#             parallel_mfdfa = mfdfa.MFDFAImage(image=image,
#                                               nworkers=nworkers)
#             foo = parallel_mfdfa.run()
#
#             mean = np.array(foo.ravel())
#             mean = mean[mean > 0]
#             mean = np.mean(mean)
#
#             mask_foo = foo.copy() > mean
#             lungs = mfdfa.get_lungs(foo)
#             good_lungs = mfdfa.get_lungs(mask_foo)
#             union_lungs = mfdfa.union_image_by_graph(good_lungs)
#
#
#             image_resized = resize(union_lungs, (128, 128),
#                                    anti_aliasing=False)
#             image_original_resize = resize(mfdfa.get_lungs(image), (128, 128),
#                                    anti_aliasing=False)
#
#             image_mfdfa = resize(mfdfa.get_lungs(foo), (128, 128),
#                                  anti_aliasing=False)
#
#             list_binary_images.append(image_resized)
#
#             path_final_img = os.path.join(path_images_id,img)
#
#             imsave(path_final_img,
#                    image_mfdfa)
#
#
#     stack_id = np.stack(list_binary_images)
#
#     np.save(final_path_stack,
#             stack_id)
#
