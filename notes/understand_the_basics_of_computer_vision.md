## 🧠 1. What Is an Image to a Computer?

To a human, an image is a visual.  
To a computer, it's just **a matrix of numbers** — where each number represents **pixel intensity**.

### 🔸 Types of Images:

|Type|Structure|Description|
|---|---|---|
|Grayscale|2D array (H x W)|1 channel (black to white)|
|RGB / BGR|3D array (H x W x 3)|3 channels (Red, Green, Blue or BGR)|
|Binary|2D array (0 or 1)|Black and white|

> 🧪 Try it: A 100x100 RGB image = `100 x 100 x 3` = 30,000 pixel values!

---

## 🔧 2. Setting Up Your Environment

Install OpenCV and NumPy:

bash

CopyEdit

`pip install opencv-python numpy`

Import the basics:

python

CopyEdit

`import cv2 import numpy as np`

---

## 🖼️ 3. Loading and Displaying Images

### 📥 Load an image

python

CopyEdit

`image = cv2.imread('face.jpg')  # Loads in BGR by default`

### 👁️ Display the image

python

CopyEdit

`cv2.imshow('Image Window', image) cv2.waitKey(0)  # Waits for any key to be pressed cv2.destroyAllWindows()`

### 📏 Check image shape

python

CopyEdit

`print(image.shape)  # Output: (height, width, channels)`

---

## 🎨 4. Color Spaces and Channels

### Convert BGR to Grayscale:

python

CopyEdit

`gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)`

### Split channels (B, G, R):

python

CopyEdit

`b, g, r = cv2.split(image)`

### Merge channels:

python

CopyEdit

`merged = cv2.merge((b, g, r))`

---

## ✂️ 5. Basic Operations

### 🔍 Resize image

python

CopyEdit

`resized = cv2.resize(image, (300, 300))`

### 📐 Crop image

python

CopyEdit

`cropped = image[50:200, 100:300]  # [y1:y2, x1:x2]`

### 🖍️ Draw shapes and text

python

CopyEdit

`cv2.rectangle(image, (50, 50), (200, 200), (0, 255, 0), 2)  # Green box cv2.circle(image, (100, 100), 50, (255, 0, 0), 3)           # Blue circle cv2.putText(image, 'Hello', (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)`

---

## 🔄 6. Rotate and Flip

### 🔁 Rotate (90 degrees clockwise)

python

CopyEdit

`rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)`

### ↔️ Flip horizontally

python

CopyEdit

`flipped = cv2.flip(image, 1)  # 1 for horizontal, 0 for vertical`

---

## 🔎 7. Accessing Pixel Values

python

CopyEdit

`pixel = image[100, 150]  # Returns a list: [B, G, R] print(f"Pixel at (100, 150): {pixel}")`

### Modify a pixel:

python

CopyEdit

`image[100, 150] = [255, 255, 255]  # Set to white`

---

## 💡 Quick Summary: Important OpenCV Functions

|Function|Description|
|---|---|
|`cv2.imread()`|Load image from file|
|`cv2.imshow()`|Display image|
|`cv2.cvtColor()`|Convert color space|
|`cv2.resize()`|Resize an image|
|`cv2.rectangle()`|Draw a rectangle|
|`cv2.putText()`|Write text on image|
|`cv2.flip()`|Flip image vertically/horizontally|

---

## 🧪 Practice Exercise

Try this script:

python

CopyEdit

`import cv2  # Load and display image image = cv2.imread('face.jpg') cv2.imshow('Original', image)  # Convert to grayscale and show gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) cv2.imshow('Grayscale', gray)  # Draw a rectangle and put text cv2.rectangle(image, (50, 50), (250, 250), (0, 255, 0), 2) cv2.putText(image, 'Face', (60, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)  cv2.imshow('With Rectangle', image) cv2.waitKey(0) cv2.destroyAllWindows()`

---

## 📚 What's Next?

Once you're comfortable with this:

- 📸 Work with webcam streams using `cv2.VideoCapture()`
    
- 🤖 Use face detection (`cv2.CascadeClassifier`) for real-time detection
    
- Then move into **face recognition**
