from PIL import Image as Picture
import numpy as np

def store(result, path):
	"Transform un numpy array en image que l'on stock et affiche à l'écran"
	extension = path.split(".")[-1]
	image = Picture.fromarray(result).save(path, extension)
	image = Picture.open(path).convert('L')	
	image.show()
	return image


def min_max_mean_hist(content, width, height):
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
	return np.sum(matrix1*matrix2)


def movement(x, stride, size):
	return stride*x + size