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

# TODO
# Define the text labels
fashion_mnist_labels = ["T-shirt/top",  # index 0
						"Trouser",  # index 1
						"Pullover",  # index 2
						"Dress",  # index 3
						"Coat",  # index 4
						"Sandal",  # index 5
						"Shirt",  # index 6
						"Sneaker",  # index 7
						"Bag",  # index 8
						"Ankle boot"]  # index 9

solution = img.copy()
solution1 = solution
cv2.fastNlMeansDenoisingColored(solution, solution1, 70, 21, 7, 12)

gray = cv2.cvtColor(solution1, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 1)
thresh = cv2.threshold(blurred, 225, 255, cv2.THRESH_BINARY)[1]
thresh = cv2.bitwise_not(thresh)

contours, hierachy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

fashion_items = []
delta_x = 3
delta_y = 3

for contour in contours:
	x, y, w, h = cv2.boundingRect(contour)

	x = x - delta_x
	y = y - delta_y

	if w * h < 80:
		continue

	w = int (w +  delta_x)
	h = int (h +  delta_y)
	cropped_img = (img[y: y + h, x: x + w]).copy()

	# Getting the bigger side of the image
	s = max(cropped_img.shape[0:2])

	# Creating a dark square with NUMPY
	square_img = np.full((s, s, 3), 255, dtype='uint8')

	# Getting the centering position
	ax, ay = (s - cropped_img.shape[1]) // 2, (s - cropped_img.shape[0]) // 2

	# Pasting the 'image' in a centering position
	square_img[ay:cropped_img.shape[0] + ay, ax:ax + cropped_img.shape[1]] = cropped_img

	resized_img = cv2.resize(square_img, (28, 28))
	fashion_items.append(resized_img)
	cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

	input_img = cv2.bitwise_not(resized_img)
	input_img = input_img.astype('float32') / 255
	# input_img = 1 - input_img
	input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)

	# plt.imshow(square_img, cmap='gray', vmin=0, vmax=255)
	# plt.show()

	#plt.imshow(input_img, cmap='gray', vmin=0, vmax=1)
	#plt.show()

	input_img = input_img.reshape(1, 28, 28, 1)
	probabilities = model.predict(input_img)
	# print(probabilities)
	prediction = np.argmax(probabilities)
	# print(prediction)
	label = fashion_mnist_labels[prediction]
	print(label)

# u fashion_items se nalaze sve izdvojene slike koje imaju su dimenzije 28x28

#################################################################################

# Cuvamo resenje u izlazni fajl

cv2.imwrite("tests/{}_out.png".format(tp_idx), img)
