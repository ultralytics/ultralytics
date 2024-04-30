from validator import BaseValidator
import numpy as np

iouv = [1,2,3]
iou = np.array((5,2))
correct = np.zeros((5, 3)).astype(bool)


BaseValidator._numba_match_predictions( iouv, iou, correct)