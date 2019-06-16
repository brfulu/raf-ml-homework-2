import cv2
import keras
import numpy as np
import sys

# Prvi i jedini argument komandne linije je indeks test primera
if len(sys.argv) != 2:
    print("Neispravno pozvan fajl, koristiti komandu \"python3 main.py X\" za pokretanje na test primeru sa brojem X")
    exit(0)

tp_idx = sys.argv[1]
img = cv2.imread('tests/{}.png'.format(tp_idx))

#################################################################################
# U ovoj sekciji implementirati obradu slike, ucitati prethodno trenirani Keras
# model, i dodati bounding box-ove i imena klasa na sliku.
# Ne menjati fajl van ove sekcije.

# Ucitavamo model
#model = keras.models.load_model('model.h5')

# TODO
solution = img.copy()
solution1 = solution
cv2.fastNlMeansDenoisingColored(solution, solution1, 80, 21, 7, 14)

gray = cv2.cvtColor(solution1, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 1)
thresh = cv2.threshold(blurred, 220, 255, cv2.THRESH_BINARY)[1]
thresh = cv2.bitwise_not(thresh)

contours, hierachy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

cropImageList = []
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    croppedImage = (img[y : y + h, x : x + w]).copy()
    cropImageList.append(cv2.resize(croppedImage, (28, 28)))
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

#u cropImageList se nalaze sve izdvojene slike koje imaju su dimenzije 28x28

#################################################################################

# Cuvamo resenje u izlazni fajl

cv2.imwrite("tests/{}_out.png".format(tp_idx), img)