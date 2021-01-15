import matplotlib.pyplot as plt
import os
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

base_dir = 'C:/Users/Dawid/Desktop/baza_danych_augmentacja/inne'
aug = 'C:/Users/Dawid/Desktop/baza_danych_augmentacja/inne2'

datagen = ImageDataGenerator(
    rescale = 1./255,
    rotation_range= 40,
    width_shift_range= 0.05,
    height_shift_range= 0.05,
    shear_range= 0.1,
    zoom_range= 0.1,
    horizontal_flip= True,)

fnames = [os.path.join(base_dir, fname) for fname in os.listdir(base_dir)]
i = 0
j = 0

for img_path in fnames:
    img = image.load_img(img_path, target_size = (150,150))
    x = image.img_to_array(img)
    x = x.reshape((1,) + x.shape)
    i = i + 1
    j = 0
    if i == (len(fnames)+1):
        break
    for batch in datagen.flow(
        x,
        batch_size=32,
        save_to_dir = aug,
        save_prefix= 'random',
        save_format= 'jpeg'):
        j = j + 1
        if j % 8 == 0:
            break
    plt.show()