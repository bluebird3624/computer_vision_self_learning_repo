# 🧠 Computer Vision Learning Repository

Welcome to my personal learning journey into **Computer Vision using Python** and tools like **OpenCV**, **NumPy**, and others. This repository is structured for clarity and long-term scalability as I progress from foundational concepts to advanced techniques like face recognition, object detection, and deep learning-based vision systems.

---

## 📁 Project Structure

```bash
.
├── human_faces_and_object_dataset/
│   ├── images/
│   │   ├── female_faces/
│   │   ├── male_faces/
│   │   ├── objects/
│   └── image_labels.csv         # Contains image paths and type labels
│
├── notes/
│   └── learning.md              # My structured learning notes and summaries
│
├── understanding_basics_of_compviz/
│   ├── intro_to_images.ipynb    # Working with pixels, channels, shapes
│   ├── draw_shapes_text.ipynb   # Drawing and annotation
│   ├── resize_crop_rotate.ipynb # Image manipulation
│   └── ...                      # More notebooks as I learn
│
└── README.md                    # You're here!
```

#### 🎯 Learning Goals
- Understand how computers see images (pixels, channels, color spaces)

- Learn to manipulate images: resize, crop, rotate, flip

- Learn to draw shapes and annotate images

- Access and modify pixel values efficiently

- Detect and isolate features (faces, edges, etc.)

- Train simple models for recognition

- Build interactive tools with OpenCV

#### 📚 Topics Covered So Far
| Topic                            | Status        | Notes Location                                          |
| -------------------------------- | ------------- | ------------------------------------------------------- |
| Basics of computer vision        | ✅ In Progress | `notes/learning.md`, `understanding_basics_of_compviz/` |
| Image formats, pixels & channels | ✅             | `intro_to_images.ipynb`                                 |
| Resizing, cropping, rotating     | ✅             | `resize_crop_rotate.ipynb`                              |
| Drawing & annotating images      | ✅             | `draw_shapes_text.ipynb`                                |
| Pixel access & manipulation      | ✅             | `intro_to_images.ipynb`                                 |
| Dataset organization             | ✅             | `human_faces_and_object_dataset/`                       |

#### 🧾 Dataset Description
The `human_faces_and_object_dataset/` folder contains a simple dataset for experimentation.

```bash

images/
├── female_faces/     # Images of female human faces
├── male_faces/       # Images of male human faces
├── objects/          # Images of miscellaneous objects
image_labels.csv      # Format: filepath, type (male/female/object)
```

This dataset will be used in practical exercises like:

- Face detection

- Object cropping

- Classification and labeling

## 🛠️ Tools & Technologies

- **Python 3.x**
    
- OpenCV (`cv2`)
    
- NumPy
    
- Jupyter Notebooks (`.ipynb`)
    
- Markdown for documentation

## 🚀 How to Use This Repo

1. Clone the repo:
    
    bash
    
    CopyEdit
    
    `git clone <repo link> cd computer_vision_self_learning_repo`
    
2. Open the notebooks:
    
    bash
    
    CopyEdit
    
    `jupyter lab # or jupyter notebook`
    
3. Explore `notes/learning.md` for a structured reading guide.
    
4. Use `human_faces_and_object_dataset` for testing concepts.

## About This Project

This is a **self-guided educational project**, driven by curiosity and the need to build a solid foundation in computer vision — especially for applications in facial recognition, automation, and intelligent systems.

## 📌 Coming Up

- Face and object detection using Haar Cascades
    
- Feature extraction (HOG, SIFT)
    
- Face embeddings with deep learning (FaceNet, Dlib)
    
- CNNs for image classification
    
- Building a real-time face recognition app

## 💬 License

This project is open for educational use and contribution. Images are sourced or generated for academic purposes only.

> **“You don’t learn to walk by following rules. You learn by doing, and by falling over.” — Richard Branson**

Happy learning! 🎓