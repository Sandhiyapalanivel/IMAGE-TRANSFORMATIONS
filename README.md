# IMAGE-TRANSFORMATIONS


## Aim
To perform image transformation such as Translation, Scaling, Shearing, Reflection, Rotation and Cropping using OpenCV and Python.

## Software Required:
Anaconda - Python 3.7

## Algorithm:
### Step1:


Import necessary libraries such as OpenCV, NumPy, and Matplotlib for image processing and visualization.

### Step2:

Read the input image using cv2.imread() and store it in a variable for further processing.


### Step3:

Apply various transformations like translation, scaling, shearing, reflection, rotation, and cropping by defining corresponding functions:

1.Translation moves the image along the x or y-axis.
2.Scaling resizes the image by scaling factors.
3.Shearing distorts the image along one axis.
4.Reflection flips the image horizontally or vertically.
5.Rotation rotates the image by a given angle.

### Step4:
Display the transformed images using Matplotlib for visualization. Convert the BGR image to RGB format to ensure proper color representation.

### Step5:
Save or display the final transformed images for analysis and use plt.show() to display them inline in Jupyter or compatible environments.

## Program:
```python
Developed By: RAHINI A
Register Number: 212223230165


import cv2
import numpy as np
import matplotlib.pyplot as plt

# Function to display image using Matplotlib
def display_image(image, title):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for proper color display
    plt.imshow(image_rgb)
    plt.title(title)
    plt.axis('off')
    plt.show()

# Load an image
image = cv2.imread('tree.png')
display_image(image, 'Original Image')


# i) Image Translation
def translate(img, x, y):
    M = np.float32([[1, 0, x], [0, 1, y]])
    translated = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
    return translated

translated_image = translate(image, 100, 50)
display_image(translated_image, 'Translated Image')

# ii) Image Scaling
def scale(img, scale_x, scale_y):
    scaled = cv2.resize(img, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
    return scaled

scaled_image = scale(image, 1.5, 1.5)
display_image(scaled_image, 'Scaled Image')

# iii) Image Shearing
def shear(img, shear_factor):
    rows, cols, _ = img.shape
    M = np.float32([[1, shear_factor, 0], [0, 1, 0]])
    sheared = cv2.warpAffine(img, M, (cols, rows))
    return sheared

sheared_image = shear(image, 0.5)
display_image(sheared_image, 'Sheared Image')

# iv) Image Reflection
def reflect(img):
    reflected = cv2.flip(img, 1)  # 1 for horizontal flip
    return reflected

reflected_image = reflect(image)
display_image(reflected_image, 'Reflected Image')

# v) Image Rotation
def rotate(img, angle):
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h))
    return rotated

rotated_image = rotate(image, 45)
display_image(rotated_image, 'Rotated Image')

# vi) Image Cropping
def crop(img, start_row, start_col, end_row, end_col):
    cropped = img[start_row:end_row, start_col:end_col]
    return cropped

cropped_image = crop(image, 50, 50, 200, 200)
display_image(cropped_image, 'Cropped Image')

```
## Output:
### Original Image

![Screenshot 2025-04-24 133039](https://github.com/user-attachments/assets/2b519074-d17e-4e52-876f-2fc5ececbff9)

### i)Image Translation

![Screenshot 2025-04-24 133058](https://github.com/user-attachments/assets/1246bdc0-7abe-4c57-9f1f-af699060dc5a)

### ii) Image Scaling

![Screenshot 2025-04-24 133107](https://github.com/user-attachments/assets/911d471b-093f-41cc-b87f-8d0768a6e1d7)

### iii)Image shearing

![Screenshot 2025-04-24 133115](https://github.com/user-attachments/assets/e3cccf71-6efd-49c1-9b0e-33b09b562e9b)


### iv)Image Reflection

![Screenshot 2025-04-24 133125](https://github.com/user-attachments/assets/14244e67-95a0-4bd7-a6ed-8da15e8aa9d7)


### v)Image Rotation

![Screenshot 2025-04-24 133134](https://github.com/user-attachments/assets/64b4973d-4164-45b9-b2dd-c66df1a99ede)

### vi)Image Cropping
![Screenshot 2025-04-24 133143](https://github.com/user-attachments/assets/504d40ed-877d-46dc-9a91-ade0124b4d3f)


## Result: 

Thus the different image transformations such as Translation, Scaling, Shearing, Reflection, Rotation and Cropping are done using OpenCV and python programming.
