from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt

def plot_generated_images(generator, examples=100, dim=(10,10), figsize=(10,10)):
    noise= np.random.normal(loc=0, scale=1, size=[examples, 100])
    generated_images = generator.predict(noise)
    generated_images = generated_images.reshape(100, 28, 28)
    plt.figure(figsize=figsize)
    for i in range(generated_images.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(generated_images[i], interpolation='nearest', cmap='gray_r')
        plt.axis('off')
    plt.tight_layout()
    return plt

a = input()
path = 'gan_generator ' + str(a) + '.h5'
model = load_model(path)
plot = plot_generated_images(model)
plot.show()
