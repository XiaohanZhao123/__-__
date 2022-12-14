import cv2
import numpy as np
from matplotlib import pyplot as plt

def blur(image_arr, **kwargs):
    if kwargs['algorithm'] == 'gaussian':
        return cv2.GaussianBlur(image_arr, kwargs['ksize'], kwargs['sigmaX'], kwargs['sigmaY'])
    if kwargs['algorithm'] == 'bilateral':
        return cv2.bilateralFilter(image_arr, kwargs['d'], kwargs['sigmaColor'], kwargs['sigmaSpace'])

canny = cv2.Canny

def watershed(img_arr,):
    def get_marker(img_arr):
        contours = cv2.findContours(img_arr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[1]
        # Create the marker image for the watershed algorithm
        markers = np.zeros(img_arr.shape, dtype=np.int32)
        # Draw the foreground markers
        for i in range(len(contours)):
            cv2.drawContours(markers, contours, i, (i + 1), -1)
        return markers
    marker = get_marker(img_arr)
    return cv2.watershed(img_arr, marker)

if __name__ == '__main__':
    filepath = 'D:/data/Image1.jpg'
    img_arr = cv2.imread(filepath)
    img_blur = blur(img_arr, algorithm='gaussian', ksize=5, sigmaX=0, sigmaY=0)
    img_canny = canny(img_blur, 80, 120)
    cv2.imshow(img_arr,'original')
    cv2.imshow(img_blur, 'img blur')
    cv2.imshow(img_canny, 'img canny')
    plt.show()
    cv2.waitKey(0)