'''
This code is to implement color distortion method proposed in 
https://arxiv.org/abs/2002.05709,
as its image augmentation method, using TensorFlow.
'''

import tensorflow as tf
import cv2
import random
import matplotlib.pyplot as plt

def get_color_distortion(imgs, s=1.0):
    '''
    input: 
    imgs: batch of tensor RGB images [batch, H, W, 3] in range [0, 1]
    s: distortion strength in [0, 1]

    output: 
    a color distorted image [batch, H, W, 3]
    '''
    p = random.uniform(0, 1)
    if p < 0.5:
        imgs = tf.image.random_brightness(imgs, 0.8*s) # [-brightness, brightness]
        imgs = tf.image.random_contrast(imgs, max(0, 1 - 0.8*s), 1 + 0.8*s) # [max(0, 1 - contrast), 1 + contrast]
        imgs = tf.image.random_saturation(imgs, max(0, 1 - 0.8*s), 1 + 0.8*s) # [max(0, 1 - saturation), 1 + saturation]
        imgs = tf.image.random_hue(imgs, 0.2*s) # [-hue, hue]
    else:
        p = random.uniform(0, 1)
        if p < 0.2:
            imgs = tf.image.rgb_to_grayscale(imgs)
            imgs = tf.image.grayscale_to_rgb(imgs)
        else:
            pass

    return imgs

# unit testing
if __name__ == '__main__':
    # input path
    test_image_path = '../test_images/0009.jpg'

    # image I/O and processing
    img_origin = cv2.imread(test_image_path)
    img_origin = cv2.cvtColor(img_origin, cv2.COLOR_BGR2RGB)
    img = tf.convert_to_tensor(img_origin, dtype=tf.float32) # [H, W, 3]
    imgs = tf.expand_dims(img, axis=0) # [1, H, W, 3]
    imgs = imgs / 255 # Normalization
    imgs_aug = get_color_distortion(imgs, s=1.0) # output is a image tensor
    imgs_aug = imgs_aug.numpy()[0]

    # figure plot
    f, axarr = plt.subplots(1, 2, figsize=(15,15))
    axarr[0].imshow(img_origin)
    axarr[1].imshow(imgs_aug)
    plt.show()