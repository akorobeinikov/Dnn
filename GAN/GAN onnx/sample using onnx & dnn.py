import cv2
import numpy as np
import matplotlib.pyplot as plt


net = cv2.dnn.readNetFromONNX("A:/GAN models/model.onnx")
noise = np.random.normal(loc=0, scale=1, size=[100, 100])
net.setInput(noise)
generated_images = net.forward()
generated_images = generated_images.reshape(100, 28, 28)
figsize=(10,10)
dim=(10,10)
plt.figure(figsize=figsize)
for i in range(generated_images.shape[0]):
    plt.subplot(dim[0], dim[1], i+1)
    plt.imshow(generated_images[i], interpolation='nearest', cmap='gray_r')
    plt.axis('off')
plt.tight_layout()
plt.show()