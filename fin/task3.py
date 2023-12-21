import numpy as np

ori = np.array([[33, 152, 19, 153], [162, 62, 165, 78], [46, 127, 153, 13]])
mask = np.array([[64, 9, 133, 45], [135, 30, 150, 9], [11, 131, 107, 160]])

# let new = alpha * mask + (1 - alpha) * ori
alpha = 1.6

new = alpha * mask + (1 - alpha) * ori

print(new)
