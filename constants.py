import numpy as np

HEIGHT = 874
WIDTH = 1164
PP = np.array([WIDTH / 2., HEIGHT / 2.], dtype=np.float32)

FOCAL_LENGTH = 910

MAX_YAW = np.tan(FOCAL_LENGTH / WIDTH / 2.)
MAX_PITCH = np.tan(FOCAL_LENGTH / HEIGHT / 2.)

CAMERA_MAT = np.array([
    [FOCAL_LENGTH, 0, PP[0]],
    [0, FOCAL_LENGTH, PP[1]],
    [0, 0, 1]],
dtype=np.float32)