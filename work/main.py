## Ce progtamme ne manipule que des images en noir sur blanc.

import Image
import numpy as np


filter = np.transpose(np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]]))

im2 = Image.Image("../../../data/input/lena.pgm")
#im3 = im2.multiplication(1.5, "mult.pgm")

filter = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]]) 

im2.histogram_equalization("im.pgm")

#im2.histogram("hist.pdf")

#im2.intensity_profil((200, 400), (200, 100), 5, "result.pgm", "intensity.pdf")
