'''
This code is to implement color distortion method proposed in 
https://arxiv.org/abs/2002.05709,
as its image augmentation method, using PyTorch.
'''

from torchvision import transforms
import cv2
from PIL import Image
import matplotlib.pyplot as plt

def get_color_distortion(s=1.0):
    # input: s as distortion strength
    # output: a composed color distortion function
    color_jitter = transforms.ColorJitter(brightness=0.8*s, contrast=0.8*s, saturation=0.8*s, hue=0.2*s)
    rnd_color_jitter = transforms.RandomApply([color_jitter])
    rnd_gray = transforms.RandomGrayscale(p=0.2)
    color_distort = transforms.Compose([rnd_color_jitter, rnd_gray])

    return color_distort

# unit testing
if __name__ == '__main__':
    # input path
    test_image_path = '../test_images/0009.jpg'

    # image I/O and processing
    img_origin = cv2.imread(test_image_path)
    img_origin = cv2.cvtColor(img_origin, cv2.COLOR_BGR2RGB)
    img_origin = Image.fromarray(img_origin) # transform to PIL array
    img_aug = get_color_distortion(s=1.0)(img_origin) # output is a PIL image

    # figure plot
    f, axarr = plt.subplots(1, 2, figsize=(15,15))
    axarr[0].imshow(img_origin)
    axarr[1].imshow(img_aug)
    plt.show()