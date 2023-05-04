import math
import numpy as np
import matplotlib.pyplot as plt
import scipy
import skimage.transform as sktr
import cv2

# Take 4 allignment pairs manually
def get_points(im1, im2):
    print('Please select 4 points in each image for alignment.')
    plt.imshow(im1)
    p1, p2, p3, p4 = plt.ginput(4)
    plt.close()
    plt.imshow(im2)
    p5, p6, p7, p8 = plt.ginput(4)
    plt.close()
    src = np.array([p1, p2, p3, p4])
    dst = np.array([p5, p6, p7, p8])
    return src.astype(int), dst.astype(int)

# Read images
def Read(path):
    source = plt.imread(path) / 255
    source = np.clip(source, 0, 255)
    return source

# Setting up the input output paths and the parameters
inputDir = '../Images/'
outputDir = '../Samples/'

# 1. Collect correspondence pairs
im1 = Read(inputDir + "d_2.jpeg")
im2 = Read(inputDir + "d_1.jpeg")
src, dst = get_points(im1, im2)


# 2. Generate homography H
# to test whether the computed H is correct: sample, mask = cv2.findHomography(src, dst, mask=None)
A = [[src[0][0], src[0][1], 1, 0, 0, 0, -src[0][0]*dst[0][0], -src[0][1]*dst[0][0]],
     [src[1][0], src[1][1], 1, 0, 0, 0, -src[1][0]*dst[1][0], -src[1][1]*dst[1][0]],
     [src[2][0], src[2][1], 1, 0, 0, 0, -src[2][0]*dst[2][0], -src[2][1]*dst[2][0]],
     [src[3][0], src[3][1], 1, 0, 0, 0, -src[3][0]*dst[3][0], -src[3][1]*dst[3][0]],
     [0, 0, 0, src[0][0], src[0][1], 1, -src[0][0]*dst[0][1], -src[0][1]*dst[0][1]],
     [0, 0, 0, src[1][0], src[1][1], 1, -src[1][0]*dst[1][1], -src[1][1]*dst[1][1]],
     [0, 0, 0, src[2][0], src[2][1], 1, -src[2][0]*dst[2][1], -src[2][1]*dst[2][1]],
     [0, 0, 0, src[3][0], src[3][1], 1, -src[3][0]*dst[3][1], -src[3][1]*dst[3][1]],
    ]
A = np.array(A)
b = np.array([dst[0][0], dst[1][0], dst[2][0], dst[3][0], dst[0][1], dst[1][1], dst[2][1], dst[3][1]])
H = np.reshape(np.concatenate((np.linalg.solve(A, b), [1])), (3, 3))


# 3. Determine bounds of the new combined mage
left_top = np.dot(H, [0, 0, 1])
left_top = np.array([left_top[0]/left_top[2], left_top[1]/left_top[2]]).astype(int)
right_top = np.dot(H, [len(im1)-1, 0, 1])
right_top = np.array([right_top[0]/right_top[2], right_top[1]/right_top[2]]).astype(int)
left_bot = np.dot(H, [0, len(im1[0])-1, 1])
left_bot = np.array([left_bot[0]/left_bot[2], left_bot[1]/left_bot[2]]).astype(int)
right_bot = np.dot(H, [len(im1)-1, len(im1[0])-1, 1])
right_bot = np.array([right_bot[0]/right_bot[2], right_bot[1]/right_bot[2]]).astype(int)

x_min = min(0, left_top[0], left_bot[0])
x_max = max(len(im1)-1, right_top[0], right_bot[0])
y_min = min(0, left_top[1], right_top[1])
y_max = max(len(im1[0])-1, left_bot[1], right_bot[1])

width = x_max - x_min + 1
height = y_max - y_min + 1
tmp = np.array(np.zeros((height, width, 3), dtype=np.float32))

# 4. Perform inverse warp
# Fix this part to do proper inverse warping
H_inverse = np.linalg.inv(H)

# 5. Overlay remaining image content onto the warped image content
# start = [-x_min, -y_min]
# for i in range(0, len(im2) - 1):
#     for j in range(0, len(im2[0]) -1):
#         tmp[start[1] + i][start[0] + j] = im2[i][j]

plt.imsave(outputDir + "/" + "" + "_result.jpg", tmp)



# # bilinear interpolation
# a = original[0] - math.floor(original[0])
# b = original[1] - math.floor(original[1])

# bottom_left = (math.floor(original[0]), math.floor(original[1])) 
# bottom_right = (math.ceil(original[0]), math.floor(original[1]))
# top_left = (math.floor(original[0]), math.ceil(original[1]))
# top_right = (math.ceil(original[0]), math.ceil(original[1]))

# color = (1 - a)*(1 - b)*im1[bottom_left] + a*(1 - b)*im1[bottom_right] + a*b*im1[top_right] + (1 - a)*b*im1[top_left]
# tmp[i, j] = color