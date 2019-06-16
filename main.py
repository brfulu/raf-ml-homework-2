import cv2
import keras
import numpy as np
import sys
import matplotlib.pyplot as plt

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
model = keras.models.load_model('fashion.h5')

item_labels = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

solution = img.copy()
solution1 = solution
cv2.fastNlMeansDenoisingColored(solution, solution1, 70, 21, 7, 12)

gray = cv2.cvtColor(solution1, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 1)
thresh = cv2.threshold(blurred, 225, 255, cv2.THRESH_BINARY)[1]
thresh = cv2.bitwise_not(thresh)

contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

copy_img = img.copy()

for contour in contours:
	x, y, w, h = cv2.boundingRect(contour)

	if w * h < 100:
		continue

	cropped_img = (copy_img[y: y + h, x: x + w]).copy()

	s = max(cropped_img.shape[0:2])
	square_img = np.full((s, s, 3), 255, dtype='uint8')
	ax, ay = (s - cropped_img.shape[1]) // 2, (s - cropped_img.shape[0]) // 2
	square_img[ay:cropped_img.shape[0] + ay, ax:ax + cropped_img.shape[1]] = cropped_img

	resized_img = cv2.resize(square_img, (28, 28))

	input_img = cv2.bitwise_not(resized_img)
	input_img = input_img.astype('float32') / 255
	input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)

	# plt.imshow(input_img, cmap='gray', vmin=0, vmax=1)
	# plt.show()

	input_img = input_img.reshape(1, 28, 28, 1)
	probabilities = model.predict(input_img)
	prediction = np.argmax(probabilities)
	label = item_labels[prediction]

	font = cv2.FONT_HERSHEY_SIMPLEX
	bottom_left_corner = (x - 8, y - 4)
	font_scale = 0.4
	font_color = (0, 0, 255)
	line_type = 1

	cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
	cv2.putText(img, item_labels[prediction], bottom_left_corner, font, font_scale, font_color, line_type)

#################################################################################

# Cuvamo resenje u izlazni fajl

cv2.imwrite("tests/{}_out.png".format(tp_idx), img)
