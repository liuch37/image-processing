'''
This code is to visualize attention map on an image.
Ref: https://github.com/facebookresearch/mmf/issues/145
'''

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image

cmap = matplotlib.cm.get_cmap('viridis')
#cmap.set_bad(color="k", alpha=0.0)

def attention_blend_interpolation(im, att, factor):
    softmax = att

    img_h, img_w = im.shape[:2]
    opacity = cv2.resize(softmax, (img_w, img_h))
    opacity = np.minimum(opacity, 1)

    opacity = opacity[..., np.newaxis]

    vis_im = np.array(Image.fromarray(cmap(opacity, bytes=True), 'RGBA'))
    vis_im = vis_im.astype(im.dtype)
    vis_im = cv2.addWeighted(im, factor[0], vis_im, factor[1], 0)
    vis_im = vis_im.astype(im.dtype)
    
    return vis_im

def attention_grid_interpolation(im, att):
    softmax = att

    opacity = cv2.resize(softmax, (im.shape[1], im.shape[0]))
    opacity = opacity[..., np.newaxis]
    opacity = opacity*0.95+0.05

    vis_im = opacity*im + (1-opacity)*255
    vis_im = vis_im.astype(im.dtype)
    return vis_im

def visualize_pred(im, att_map, factor, mode='blend'):
    '''
    im: an RGB image
    att_map: an attention map
    factor: a tuple of weights for original image and attention map (alpha, beta) used in blend mode
    mode: support 'blend' or 'grid'
    '''
    im = cv2.cvtColor(im, cv2.COLOR_RGB2RGBA)
    if mode == 'blend':
        im_ocr_att = attention_blend_interpolation(im, att_map, factor)
    elif mode == 'grid':
        im_ocr_att = attention_grid_interpolation(im, att_map)
    else:
        print("ERROR: no supported blending mode for {}".format(mode))
        im_ocr_att = None

    return im_ocr_att

# unit testing
if __name__ == '__main__':
    # input path
    test_image_path = '../test_images/0009.jpg'

    im = cv2.imread(test_image_path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    # fake attention map
    x, y = np.meshgrid(np.linspace(-10, 10, 100), np.linspace(-10, 10, 100))
    dst = np.sqrt(x*x+y*y)
    # initializing sigma and muu
    sigma = 2
    muu = 0.000
    # calculating Gaussian array
    att_map = np.exp(-( (dst-muu)**2 / ( 2.0 * sigma**2 ) ) )
    # creating blended image
    img_blend = visualize_pred(im, att_map, (0.5, 0.7))
    plt.imshow(img_blend)
    plt.show()
