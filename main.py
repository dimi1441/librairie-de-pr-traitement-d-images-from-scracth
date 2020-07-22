## Ce progtamme ne manipule que des images en noir sur blanc.

import Image
import numpy as np

im1 = Image.Image("data/tour2.jpeg")
filtre = np.ones((3,3))
print(im1.convolution(filtre, 2, method="SAME"))
