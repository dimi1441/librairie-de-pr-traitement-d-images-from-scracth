## Ce progtamme ne manipule que des images en noir sur blanc.

import Image
import numpy as np

im1 = Image.Image("data/input/tour2.jpeg")
filtre = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])

#a = filtre.reshape(1, -1)
#print(a)
#print(a[0].sort())

im1.filter_median(3, 2, "data/output/med.jpeg")
