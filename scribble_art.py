import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
img = cv2.imread('Bald-Eagle.jpg')

# Convert the image from RGB to Grayscale
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#Apply median blur to reduce noise
gray_image = cv2.medianBlur(gray_image, 5)
plt.imshow(gray_image, cmap="gray")
plt.axis('off')
plt.title("After Median Blurring")
plt.show()

# Detect edges using adaptive thresholding
edges = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
plt.imshow(edges, cmap="gray")
plt.axis('off')
plt.title("After Adaptive Thresholding")
plt.show()

# Removing the noise 
gray_image=cv2.bilateralFilter(gray_image, 2, 450, 650)
plt.imshow(gray_image, cmap="gray")
plt.axis('off')
plt.title("After Bilateral Filtering")
plt.show()

#Convert the image to color
color = cv2.stylization(img, sigma_s=150, sigma_r=0.25)
plt.imshow(color)
plt.axis('off')
plt.title("After Stylization")
plt.show()

# Display the catoon image 
cv2.imshow("Color", color)
cv2.waitKey(0)
cv2.destroyAllWindows()
