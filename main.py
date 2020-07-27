## Ce progtamme ne manipule que des images en noir sur blanc.

import Image
import numpy as np
from routines import read_pgm

im1 = Image.Image("data/input/cerf.jpeg")
filter = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
im1.interpollation(2, "knn", "data/output/result.jpeg")