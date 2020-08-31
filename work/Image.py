import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from routines import store, min_mean_hist, movement, cross_correlation, median, read_pgm


class Image(object):

	def __init__(self, path):
		self.content, self.max = read_pgm(path)
		self.height, self.width = self.content.shape
		self.min, self.mean, self.hist = min_mean_hist(self.content, self.width, self.height)


	# Luminance and contrast

	def luminance(self):
		return self.mean


	def contrast(self, method):
		assert (method in ["std", "variation"]), "Methode inconnu."
		
		# Ecart-type des variations de nveau de gris.
		if method == "std":
			result = 0
			for x in range(self.height):
				for y in range(self.width):
					result += pow(self.content[x, y] - self.mean, 2)
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
		
		if point1[0] == point2[0]:
			values_for_y = list(range(point1[1], point2[1]+1)) if (point2[1] > point1[1]) else list(range(point2[1], point1[1]+1))
			values_for_x = len(values_for_y)*[point1[0]]
		
		elif point1[1] == point2[1]:
			values_for_x = list(range(point1[0], point2[0]+1)) if (point2[0] > point1[0]) else list(range(point2[0], point1[0]+1))
			values_for_y = len(values_for_x)*[point1[1]]
		
		else: 
			# Find parameter of the line: slope and y-intercept
			y = np.transpose(np.array([point1[1], point2[1]]))
			x = np.array([[point1[0], 1], [point2[0], 1]])

			slope = (point2[1] - point1[1]) / (point2[0] - point1[0])
			y_intercept = point1[1] - point1[0]*slope

			# Find points on the line
			def line(x):
				return int(x*slope + y_intercept)

			Lvec = np.vectorize(line)

			values_for_x = list(range(point1[0], point2[0]+1)) if (point2[0] > point1[0]) else list(range(point2[0], point1[0]+1))
			values_for_y = list(Lvec(np.array(values_for_x)))

		# Convert image to numpy array
		numpy_matrix = np.array(self.content, dtype=np.uint8)
		
		# Draw the line
		intensities = []
		for (x, y) in zip(values_for_x, values_for_y):
			intensities.append(numpy_matrix[y, x])
			numpy_matrix[y, x] = color

		# Plot
		#intensities = intensities.reverse()
		plt.plot(intensities[::-1])
		plt.xlabel("Pixels")
		plt.ylabel("Gray scale")
		plt.title("Line profil")
		plt.savefig(intensity_path, dpi=600 , format=intensity_path.split(".")[-1])
		plt.show()

		# Store the result
		store(numpy_matrix, result_path)

	
	#  Addition, soustraction, mutiplipar un scalaire.

	def multiplication(self, scalar, result_path):
		assert (scalar >= 0), "Donner un scalaire positif."

		result = np.ones((self.height, self.width), dtype=np.uint8)
		for x in range(self.height):
			for y in range(self.width):
				result[x, y] = min(scalar*self.content[x, y], 255)
		
		store(result, result_path)
		return Image(result_path)


	def addition(self, image, result_path):
		assert (self.width == image.width and self.height == image.height), "Les deux images ne sont pas de meme taille"

		result = np.ones((self.height, self.width), dtype=np.uint8)
		for x in range(self.height):
			for y in range(self.width):
				result[x, y] = min(self.content[x, y] + image.content[x, y], 255)
		store(result, result_path)
		return Image(result_path)


	def subtraction(self, image, result_path):
		assert (self.width == image.width and self.height == image.height), "Les deux images ne sont pas de meme taille"

		result = np.ones((self.height, self.width), dtype=np.uint8)
		for x in range(self.height):
			for y in range(self.width):
				result[x, y] = max(self.content[x, y] - image.content[x, y], 0)
		
		store(result, result_path)
		return Image(result_path)



	#  Negation, conjontion et disjonction logique

	def negation(self, result_path):
		result = np.ones((self.height, self.width), dtype=np.uint8)
		for x in range(self.height):
			for y in range(self.width):
				result[x, y] = 255 - self.content[x, y]
		
		store(result, result_path)
		return Image(result_path)


	def conjunction(self, image, result_path):
		assert (self.width == image.width and self.height == image.height), "Les deux images ne sont pas de meme taille"

		result = np.ones((self.height, self.width), dtype=np.uint8)
		for x in range(self.height):
			for y in range(self.width):
				result[x, y] = min(self.content[x, y], image.content[x, y])
		
		store(result, result_path)
		return Image(result_path)


	def disjunction(self, image, result_path):
		assert (self.width == image.width and self.height == image.height), "Les deux images ne sont pas de meme taille"

		result = np.ones((self.height, self.width), dtype=np.uint8)
		for x in range(self.height):
			for y in range(self.width):
				result[x, y] = max(self.content[x, y], image.content[x, y])
		
		store(result, result_path)
		return Image(result_path)



	# Ameliorer le contraste

	def enhance_contrast(self, func, result_path):
		result = np.zeros((self.height, self.width), dtype=np.uint8)

		LUT = list(range(256))
		for i in range(256):
			LUT[i] = func(i)

		for x in range(self.height):
			for y in range(self.width):
				result[x, y] = LUT[self.content[x, y]]
		
		store(result, result_path)
		return Image(result_path)


	def enhance_contrast_linearly(self, result_path):
		result = np.zeros((self.height, self.width), dtype=np.uint8)

		def func(i):
			return 255*(i - self.min)/(self.max - self.min)

		LUT = list(range(256))
		for i in range(256):
			LUT[i] = func(i)

		for x in range(self.height):
			for y in range(self.width):
				result[x, y] = LUT[self.content[x, y]]
		
		store(result, result_path)
		return Image(result_path)


	def enhance_contrast_with_saturation(self, smin, smax, result_path):
		result = np.zeros((self.height, self.width), dtype=np.uint8)

		def func(i):
			return 255*(i - smin)/(smax - smin)

		LUT = list(range(256))
		for i in range(256):
			LUT[i] = func(i)

		for x in range(self.height):
			for y in range(self.width):
				result[x, y] = LUT[self.content[x, y]]
		
		store(result, result_path)
		return Image(result_path)

		
	def histogram_equalization(self, result_path):
		normalized_histogram = self.hist / (self.width*self.height)

		density = np.array(range(256))
		for i in range(256):
			density[i] = np.sum(normalized_histogram[0, 0:i+1])
		print(density)
		result = np.zeros((self.height, self.width), dtype=np.uint8)
		for x in range(self.height):
			for y in range(self.width):
				result[x, y] = np.round(density[self.content[x, y]]*255)
		
		store(result, result_path)
		return Image(result_path)



	# Interpollation
	# Ca ne marche pas
	def interpollation(self, size, method, result_path):
		if method == "knn":
			result = np.zeros((size*self.height, size*self.width), dtype=np.uint8)

			index_x = 0
			for x in range(self.height):
				index_y = 0
				for y in range(self.width):
					result[index_x:index_x+size, index_y:index_y+size] = self.content[x, y]*np.ones((size, size))
					index_y += size
				index_x += size
			store(result, result_path)
			return Image(result_path)
		else:
			print("Les autres méthodes ne marchent pas.")


	def convolution(self, filter, stride, method, result_path):
		size = filter.shape[0]
		assert size == filter.shape[1], "Votre matrice vous servant de filtre n'est pas carre"
		assert type(stride) == int, "Le stride doit etre un entier"
		assert method in ["VALID", "SAME"], "Methode invalide, vous devez choisir entre SAME et VALID"

		if method == "VALID":
			numpy_image = np.asarray(self.content)
			new_width = round((self.width - size) / stride)
			new_height = round((self.height - size) / stride)
			result = np.zeros((new_height, new_width), dtype=np.uint8)
			for x in range(new_height):
				for y in range(new_width):
					a = movement(x, stride, size)
					b = movement(y, stride, size)
					result[x, y] = cross_correlation(numpy_image[a-size: a, b-size: b], filter)
			
			store(result, result_path)
			return Image(result_path)
		
		if method == "SAME":

			new_width = round((self.width - size) / stride) + 1
			new_height = round((self.height - size) / stride) + 1
			result = np.zeros((new_height, new_width), dtype=np.uint8)

			numpy_image = np.zeros((self.height, self.width), dtype=np.uint8)
			numpy_image[0:self.height, 0:self.width] = np.asarray(self.content)
			
			for x in range(new_height):
				for y in range(new_width):
					a = movement(x, stride, size)
					b = movement(y, stride, size)
					result[x, y] = cross_correlation(numpy_image[a-size: a, b-size: b], filter)

			store(result, result_path)
			return Image(result_path)


	def median_filter(self, size, stride, method, result_path):
		assert type(stride) == int, "Le stride doit etre un entier"
		assert type(size) == int,  "La taille doit etre un entier"


		if method == "VALID":
			numpy_image = np.asarray(self.content)
			new_width = round((self.width - size) / stride)
			new_height = round((self.height - size) / stride)
			result = np.zeros((new_height, new_width), dtype=np.uint8)
			
			for x in range(new_height):
				for y in range(new_width):
					a = movement(x, stride, size)
					b = movement(y, stride, size)
					result[x, y] = median(numpy_image[a-size: a, b-size: b])
			
			store(result, result_path)
			return Image(result_path)

		if method == "SAME":
			new_width = round((self.width - size) / stride) + 1
			new_height = round((self.height - size) / stride) + 1
			result = np.zeros((new_height, new_width), dtype=np.uint8)

			numpy_image = np.zeros((self.height, self.width))
			numpy_image[0:self.height, 0:self.width] = np.asarray(self.content)
			
			for x in range(new_height):
				for y in range(new_width):
					a = movement(x, stride, size)
					b = movement(y, stride, size)
					result[x, y] = median(numpy_image[a-size: a, b-size: b])

			store(result, result_path)
			return Image(result_path)


	def gradiant_contour(self, filter, stride, method, result_path):
		imx = self.convolution(filter, stride, method, "x"+result_path)
		imy = self.convolution(np.transpose(filter), stride, method, "y"+result_path)

		return imx.addition(imy, result_path)


	def laplacian(self, stride, method, result_path):
		filter = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
		return self.convolution(filter, stride, method, result_path)