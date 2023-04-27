import math
import numpy as np
import matplotlib.pyplot as plt
import scipy
import skimage.transform as sktr

# Take 4 allignment pairs manually
def get_points(im1, im2):
    print('Please select 4 points in each image for alignment.')
    plt.imshow(im1)
    p1, p2, p3, p4 = plt.ginput(4)
    plt.close()
    plt.imshow(im2)
    p5, p6, p7, p8 = plt.ginput(4)
    plt.close()
    return (p1, p2, p3, p4, p5, p6, p7, p8)

# Read images
def Read(id, path):
    #remove * 255 jpg files
    source = plt.imread(path) * 255
    source = np.clip(source, 0, 255)
    return source

# Setting up the input output paths and the parameters
inputDir = '../Images/'
outputDir = '../Samples/'

# 1. Collect correspondce pairs


# 2. Generate homography H


# 3. Determine bounds of the new combined mage


# 4. Perform inverse warp


# 5. Overlay remaining image content onto the warped image content