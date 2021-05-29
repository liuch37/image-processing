'''
This code is to implement mosaic method proposed in
https://arxiv.org/abs/2004.10934
modified from https://github.com/jason9075/opencv-mosaic-data-aug
as its image augmentation method, using opencv and numpy.
'''

import numpy as np
import random
import cv2
import matplotlib.pyplot as plt
import pdb

def mosaic_augmentation(imgs, labels, output_size, scale_range, filter_scale=0.0):
    '''
    input:
    imgs: [batch=4, H, W, 3]
    labels: [batch=4, anchor, Class+4]
    output_size: tuple of final mosaic image output size
    scale_range: scaling range for each image (low, high) in [0, 1]
    filter_scale: bounding box filtering threshold in pixels
    output:
    mosaic image: [H, W, 3]
    new_anno: new labels [anchor*batch, Class+4]
    '''
    output_img = np.zeros([output_size[0], output_size[1], 3], dtype=np.uint8)
    scale_x = scale_range[0] + random.random() * (scale_range[1] - scale_range[0])
    scale_y = scale_range[0] + random.random() * (scale_range[1] - scale_range[0])
    divid_point_x = int(scale_x * output_size[1])
    divid_point_y = int(scale_y * output_size[0])

    new_anno = []
    
    # top-left
    img = cv2.resize(imgs[0], (divid_point_x, divid_point_y))
    output_img[:divid_point_y, :divid_point_x, :] = img
    for bbox in labels[0]:
        xmin = bbox[1] * scale_x * output_size[1] / imgs[0].shape[1]
        ymin = bbox[2] * scale_y * output_size[0] / imgs[0].shape[0]
        xmax = bbox[3] * scale_x * output_size[1] / imgs[0].shape[1]
        ymax = bbox[4] * scale_y * output_size[0] / imgs[0].shape[0]
        new_anno.append([bbox[0], xmin, ymin, xmax, ymax])

    # top-right
    img = cv2.resize(imgs[1], (output_size[1] - divid_point_x, divid_point_y))
    output_img[:divid_point_y, divid_point_x:output_size[1], :] = img
    for bbox in labels[1]:
        xmin = divid_point_x + bbox[1] * (1 - scale_x) * output_size[1] / imgs[1].shape[1]
        ymin = bbox[2] * scale_y * output_size[0] / imgs[1].shape[0]
        xmax = divid_point_x + bbox[3] * (1 - scale_x) * output_size[1] / imgs[1].shape[1]
        ymax = bbox[4] * scale_y * output_size[0] / imgs[1].shape[0]
        new_anno.append([bbox[0], xmin, ymin, xmax, ymax])
    
    # bottom-left
    img = cv2.resize(imgs[2], (divid_point_x, output_size[0] - divid_point_y))
    output_img[divid_point_y:output_size[0], :divid_point_x, :] = img
    for bbox in labels[2]:
        xmin = bbox[1] * scale_x * output_size[1] / imgs[2].shape[1]
        ymin = divid_point_y + bbox[2] * (1 - scale_y) * output_size[0] / imgs[2].shape[0]
        xmax = bbox[3] * scale_x * output_size[1] / imgs[2].shape[1]
        ymax = divid_point_y + bbox[4] * (1 - scale_y) * output_size[0] / imgs[2].shape[0]
        new_anno.append([bbox[0], xmin, ymin, xmax, ymax])
    
    # bottom-right
    img = cv2.resize(imgs[3], (output_size[1] - divid_point_x, output_size[0] - divid_point_y))
    output_img[divid_point_y:output_size[0], divid_point_x:output_size[1], :] = img
    for bbox in labels[3]:
        xmin = divid_point_x + bbox[1] * (1 - scale_x) * output_size[1] / imgs[3].shape[1]
        ymin = divid_point_y + bbox[2] * (1 - scale_y) * output_size[0] / imgs[3].shape[0]
        xmax = divid_point_x + bbox[3] * (1 - scale_x) * output_size[1] / imgs[3].shape[1]
        ymax = divid_point_y + bbox[4] * (1 - scale_y) * output_size[0] / imgs[3].shape[0]
        new_anno.append([bbox[0], xmin, ymin, xmax, ymax])

    if filter_scale > 0:
        new_anno = [anno for anno in new_anno if
                    filter_scale < (anno[3] - anno[1]) and filter_scale < (anno[4] - anno[2])]

    return output_img, new_anno

# unit testing
if __name__ == '__main__':
    # input path
    test_image1_path = '../test_images/0009.jpg'
    test_image2_path = '../test_images/0029.jpg'
    test_image3_path = '../test_images/1005.jpg'
    test_image4_path = '../test_images/1011.jpg'
    OUTPUT_SIZE = (600, 600)  # Height, Width
    SCALE_RANGE = (0.3, 0.7)
    FILTER_TINY_SCALE = OUTPUT_SIZE[0] / 50 # if height or width lower than this scale, drop it.
    # label array
    labels = np.array([[[1, 409, 152, 973, 621], [1, 443, 1648, 917, 1933], [1, 867, 1333, 1732, 1926]],
                       [[1, 409, 415, 1223, 1311], [1, 781, 58, 1781, 890], [1, 961, 573, 2020, 1611]],
                       [[1, 22, 86, 115, 191], [1, 123, 101, 203, 199], [1, 201, 119, 272, 208]],
                       [[1, 3, 25, 96, 305], [1, 119, 53, 288, 290], [1, 328, 48, 471, 291]],])
    # image IO processing
    img1 = cv2.imread(test_image1_path)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.imread(test_image2_path)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    img3 = cv2.imread(test_image3_path)
    img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
    img4 = cv2.imread(test_image4_path)
    img4 = cv2.cvtColor(img4, cv2.COLOR_BGR2RGB)
    imgs = [img1, img2, img3, img4]

    mosaic_img, mosaic_labels = mosaic_augmentation(imgs, labels, OUTPUT_SIZE, SCALE_RANGE, FILTER_TINY_SCALE)
    
    for label in mosaic_labels:
        label = [int(l) for l in label]
        mosaic_img = cv2.rectangle(mosaic_img, (label[1], label[2]), (label[3], label[4]), (255, 0, 0), 4)

    plt.imshow(mosaic_img)
    plt.show()
