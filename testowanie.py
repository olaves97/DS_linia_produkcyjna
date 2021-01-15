import cv2
import tensorflow as tf
import os



CATEGORIES = ["hammer", "screwdriver"]
hammer = 0
screwdriver = 0
counter = 0

def prepare(filepath):
    IMG_SIZE = 150  # 50 in txt-based
    img_array = cv2.imread(filepath, cv2.IMREAD_COLOR)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 3)

model = tf.keras.models.load_model("hammer_screwdriver_4.h5")

directory = 'C:/Users/Dawid/Desktop/baza_danych_aug/test_data/srubokret'
for filename in os.listdir(directory):
    if filename.endswith(".jpg"):
        data = os.path.join(directory, filename)
        prediction = model.predict(prepare(data))
        print(prediction)
        if prediction[0][0] > 0.5:
            counter = counter + 1
        if prediction[0][1] > 0.5:
            hammer = hammer + 1
        if prediction[0][2] > 0.5:
            screwdriver = screwdriver + 1

#prediction = model.predict(prepare('C:/Users/Dawid/Desktop/baza_danych_aug/test_data/srubokret/screwdriver (242)'))

print("W tym zbiorze ilosc mlotkow wynosi: ",hammer, " i stanowi to ",(hammer/len((os.listdir(directory))))*100, "% zbioru")
print("W tym zbiorze ilosc srubokretow wynosi: ",screwdriver," i stanowi to ",(screwdriver/len((os.listdir(directory))))*100, "% zbioru")
print("W tym zbiorze ilosc inne wynosi: ",counter," i stanowi to ",(counter/len((os.listdir(directory))))*100, "% zbioru")


#print(prediction)



