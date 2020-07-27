from PIL import Image as Picture
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from routines import store, min_max_mean_hist, movement, cross_correlation, median


class Image(object):

	def __init__(self, path):
		self.content = Picture.open(path).convert('L')
		self.width, self.height = self.content.size
		self.min, self.max, self.mean, self.hist = min_max_mean_hist(self.content, self.width, self.height)


	# Luminance and contrast

	def luminance(self):
		return self.mean


	def contrast(self, method):
		assert (method in ["std", "variation"]), "Methode inconnu."
		
		# Ecart-type des variations de nveau de gris.
		if method == "std":
			result = 0
			for x in range(self.width):
				for y in range(self.height):
					result += pow(self.content.getpixel((x, y)) - self.mean, 2)
			return sqrt(result / (self.height*self.width))

		# Variation entre niveaux de gris min et max.
		elif method == "variation":
			return (self.max - self.min) / (self.max + self.min)


	# Intensity profil and histogram

	def histogram(self, histogram_result_path):

		# Display histogram
		plt.bar(np.array(list(range(256))), self.hist, 1.0, color='b')
		plt.xlabel("Niveau de gris")
		plt.ylabel("Nombre de pixels")
		plt.title("Histogram")
		plt.savefig(histogram_result_path, dpi=600 , format=histogram_result_path.split(".")[-1])
		plt.show()


	def intensity_profil(self, point1, point2, color, result_path, intensity_path):		
		assert (255 >= color or self.width >= point1[0] and self.width >= point2[0] and self.height >= point1[1] and self.height >= point2[1]), "Au moins un point n'est pas contenu dans l'image."
		
		# Find parameter of the line: slope and y-intercept
		y = np.transpose(np.array([point1[1], point2[1]]))
		x = np.array([[point1[0], 1], [point2[0], 1]])

		parameter = np.dot(np.linalg.inv(x),y)

		# Find points on the line
		def line(x):
			return int(x*parameter[0] + parameter[1])

		Lvec = np.vectorize(line)
		values_for_x = list(range(point1[0], point2[0]+1)) if (point2[0] > point1[0]) else list(range(point2[0], point1[0]+1))
		values_for_y = list(Lvec(np.array(values_for_x)))

		# Convert image to numpy array
		numpy_matrix = np.array(self.content, dtype=np.uint8)

		# Draw the line
		intensities = []
		for (x, y) in zip(values_for_x, values_for_y):
			intensities.append(numpy_matrix[x, y])
			numpy_matrix[y, x] = color

		# Plot
		plt.plot(intensities)
		plt.xlabel("Pixels")
		plt.ylabel("Niveau de gris")
		plt.title("Line profil")
		plt.savefig(intensity_path, dpi=600 , format=intensity_path.split(".")[-1])
		plt.show()

		# Store the result
		store(numpy_matrix, result_path)

	
	#  Addition, soustraction, mutiplipar un scalaire.

	def multiplication(self, scalar, result_path):
		assert (scalar >= 0), "Donner un scalaire positif."

		result = np.ones((self.height, self.width), dtype=np.uint8)
		for x in range(self.width):
			for y in range(self.height):
				result[y, x] = min(scalar*self.content.getpixel((x, y)), 255)
		return store(result, result_path)


	def addition(self, image, result_path):
		assert (self.width == image.width and self.height == image.height), "Les deux images ne sont pas de meme taille"

		result = np.ones((self.height, self.width), dtype=np.uint8)
		for x in range(self.width):
			for y in range(self.height):
				result[y, x] = min(self.content.getpixel((x, y)) + image.content.getpixel((x, y)), 255)
		return store(result, result_path)


	def soustraction(self, image, result_path):
		assert (self.width == image.width and self.height == image.height), "Les deux images ne sont pas de meme taille"

		result = np.ones((self.height, self.width), dtype=np.uint8)
		for x in range(self.width):
			for y in range(self.height):
				result[y, x] = max(self.content.getpixel((x, y)) - image.content.getpixel((x, y)), 0)
		return store(result, result_path)


	#  Negation, conjontion et disjonction logique

	def negation(self, result_path):
		result = np.ones((self.height, self.width), dtype=np.uint8)
		for x in range(self.width):
			for y in range(self.height):
				result[y, x] = 255 - self.content.getpixel((x, y))
		return store(result, result_path)


	def conjunction(self, image, result_path):
		assert (self.width == image.width and self.height == image.height), "Les deux images ne sont pas de meme taille"

		result = np.ones((self.height, self.width), dtype=np.uint8)
		for x in range(self.width):
			for y in range(self.height):
				result[y, x] = min(self.content.getpixel((x, y)), image.content.getpixel((x, y)))
		return store(result, result_path)


	def disjunction(self, image, result_path):
		assert (self.width == image.width and self.height == image.height), "Les deux images ne sont pas de meme taille"

		result = np.ones((self.height, self.width), dtype=np.uint8)
		for x in range(self.width):
			for y in range(self.height):
				result[y, x] = max(self.content.getpixel((x, y)), image.content.getpixel((x, y)))
		return store(result, result_path)


	# Ameliorer le contraste

	def enhance_contraste(self, func, result_path):
		result = np.ones((self.height, self.width), dtype=np.uint8)

		LUT = list(range(256))
		for i in range(256):
			LUT[i] = func(i)

		for x in range(self.width):
			for y in range(self.height):
				result[y, x] = LUT[self.content.getpixel((x, y))]
		return store(result, result_path)


	def histogram_equalization(self, result_path):
		normalized_histogram = self.hist / (self.width*self.height)
		
		density = np.array(range(256))
		for i in range(256):
			density[i] = np.sum(normalized_histogram[0:i+1])
		
		result = np.ones((self.height, self.width), dtype=np.uint8)
		for x in range(self.width):
			for y in range(self.height):
				result[y, x] = np.round(density[self.content.getpixel((x, y))]*255)
		return store(result, result_path)


	# Interpollation
	# Ca ne marche pas
	def interpollation(self, size, method, result_path):
		if method == "knn":
			result = np.ones((size*self.height, size*self.width), dtype=np.uint8)
			for x in range(self.width):
				for y in range(self.height):
					result[y:y+size, x:x+size] = self.content.getpixel((x, y))
			if x == self.width-1  and y == self.height-1:
				return store(result, result_path)


	def convolution(self, filter, stride, method, result_path):
		size = filter.shape[0]
		assert size == filter.shape[1], "Votre matrice vous servant de filtre n'est pas carre"
		assert type(stride) == int, "Le stride doit etre un entier"
		assert method in ["VALID", "SAME"], "Methode invalide, vous devez choisir entre SAME et VALID"

		if method == "VALID":
			numpy_image = np.asarray(self.content)
			new_width = round((self.width - size) / stride)
			new_height = round((self.height - size) / stride)
			result = np.ones((new_width, new_height), dtype=np.uint8)
			for x in range(new_width):
				for y in range(new_height):
					a = movement(x, stride, size)
					b = movement(y, stride, size)
					result[x, y] = cross_correlation(numpy_image[b-size: b, a-size: a], filter)
			return store(result, result_path)
		
		if method == "SAME":

			new_width = round((self.width - size) / stride) + 1
			new_height = round((self.height - size) / stride) + 1
			result = np.ones((new_width, new_height), dtype=np.uint8)

			numpy_image = np.zeros((movement(new_width, stride, size), movement(new_height, stride, size)))
			numpy_image[0:self.width, 0:self.height] = np.transpose(np.asarray(self.content))
			
			for x in range(new_width):
				for y in range(new_height):
					a = movement(x, stride, size)
					b = movement(y, stride, size)
					result[x, y] = cross_correlation(numpy_image[a-size: a, b-size: b], filter)
			return store(result, result_path)


	def filter_median(self, size, stride, result_path):
		assert type(stride) == int, "Le stride doit etre un entier"
		assert type(size) == int,  "La taille doit etre un entier"

		numpy_image = np.asarray(self.content)
		new_width = round((self.width - size) / stride)
		new_height = round((self.height - size) / stride)
		result = np.ones((new_width, new_height), dtype=np.uint8)
		for x in range(new_width):
			for y in range(new_height):
				a = movement(x, stride, size)
				b = movement(y, stride, size)
				result[x, y] = median(numpy_image[b-size: b, a-size: a])
		return store(result, result_path)