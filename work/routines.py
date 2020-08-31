import numpy as np
import matplotlib.pyplot as plt
from collections import Counter


def read_pgm(file_name):
	"""Return a raster of integers from a PGM as a list of lists."""
	pgmf = open(file_name, "r")

	assert pgmf.readline() == 'P2\n', "Magic number incompatible"
	
	second_line = pgmf.readline()
	
	if '#' in second_line:
		(width, height) = [int(i) for i in pgmf.readline().split()]
	else:
		(width, height) = [int(i) for i in second_line.split()]

	max_pixel_value = int(pgmf.readline())
	assert max_pixel_value <= 255, "Image de 8 bits par pixels."

	rows = []
	for line in pgmf.readlines():
		for number in line.split():
			rows.append(int(number))

	result = np.array(rows, dtype=np.uint8)
	result = np.reshape(result, (height, width))

	pgmf.close()

	return result, max_pixel_value


def store(result, path):
	""" Transform numpy array into image and store the result """
	extension = path.split(".")[-1]
	
	height, width = result.shape
	result_intermediate = np.reshape(result, (1, width*height))

	with open(path, "w") as f:
		f.write("P2\n")
		f.write("#\n")
		f.write(str(width)+" "+str(height)+"\n")
		f.write(str(np.max(result))+"\n")
		for i in result_intermediate[0]:
			f.write(str(i)+"\n")

	plt.figure()
	plt.imshow(result, cmap="gray")
	plt.show()


def min_mean_hist(content, width, height):
	""" Compute min, max, mean and hist of image pixels """
	min = 255
	sum = 0
	hist = np.zeros((1, 256))
	for x in range(height):
		for y in range(width):
			hist[0, content[x, y]] += 1
			sum += content[x, y]
			if min > content[x, y]:
				min = content[x, y]
	return min, sum/(width*height), hist


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