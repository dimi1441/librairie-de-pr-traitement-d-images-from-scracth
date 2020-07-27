from PIL import Image as Picture
import numpy as np


def read_pgm(file_name):
	"""Return a raster of integers from a PGM as a list of lists."""
	pgmf = open(file_name, "r")

	assert pgmf.readline() == 'P2\n', "Magic number incompatible"
	second_line = pgmf.readline()
	if '#' in second_line:
		(width, height) = [int(i) for i in pgmf.readline().split()]
	else:
		(width, height) = [int(i) for i in second_line.split()]
	depth = int(pgmf.readline())
	assert depth <= 255, "Image de 8 bits par pixels."
	
	raster = []
	
	for y in range(height):
		row = []
		print(pgmf.readline())
		"""
		for y in range(width):
			row.append(pgmf.read(1))
		raster.append(row)
		"""
	pgmf.close()
	
	return raster


def store(result, path):
	""" Transform numpy array into image and store the result """
	extension = path.split(".")[-1]
	image = Picture.fromarray(result).save(path, extension)
	image = Picture.open(path).convert('L')	
	image.show()
	return image


def min_max_mean_hist(content, width, height):
	""" Compute min, max, mean and hist of image pixels """
	min = 255
	max = 0
	sum = 0
	hist = np.array(list(range(256)))
	for x in range(width):
		for y in range(height):
			hist[content.getpixel((x, y))] += 1
			sum += content.getpixel((x, y))
			if min > content.getpixel((x, y)):
				min = content.getpixel((x, y))
			if content.getpixel((x, y)) > max:
				max = content.getpixel((x, y))
	return min, max, sum/(width*height), hist


def cross_correlation(matrix1, matrix2):
	""" Compute cross entropy """
	result = np.sum(matrix1*matrix2)
	if result > 255:
		return 255
	elif result < 0:
		return 0
	else:
		return result


def movement(x, stride, size):
	""" Deplacement in image which we want to convoluate """
	return stride*x + size


def median(matrix):
	""" Calculate the median for a flatten matrix """
	vector = np.sort(matrix.reshape(1, -1)[0])
	middle = round(len(vector)/2)
	return vector[middle]