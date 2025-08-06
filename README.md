# Drone Image : Semantic Segmentation with ResNet34-UNet

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue)](https://www.python.org/)  [![Build Status](https://img.shields.io/badge/Notebook-Ready-yellow)]()

---

## Project Overview

This repository implements a **semantic segmentation** pipeline using a **UNet** model with a **ResNet34** backbone (pre-trained on ImageNet). The goal is to segment images into multiple classes (e.g., roads, vegetation, buildings, etc.) based on provided annotation masks.

Key steps include:

1. **Environment Setup & Dependencies**  
2. **Data Loading & Preprocessing**  
3. **Mask Encoding & One-Hot Conversion**  
4. **Training/Test Split**  
5. **Model Definition (ResNet34-UNet)**  
6. **Training Loop & Callbacks**  
7. **Evaluation & Visualization**  
8. **Saving & Reloading the Model**

---

## Repository Structure

```text
├── Drone_Image_Segmenter.ipynb   # Main Jupyter notebook
└── README.md                   # This file
```

> **Note:** The `dataset/` folder is **not** included in the repo. Download from [Dataset Link](http://dronedataset.icg.tugraz.at/).   
> Please mount or symlink your data in the notebook (e.g., via Google Drive).

---

## Environment Setup

1. **Clone the repository**  
   ```bash
   git clone https://github.com/Akshit-Soni/Drone-Image-Segmentation.git
   cd Drone-Image-Segmentation
   ```

2. **Create & activate a virtual environment**  
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

4. **pip dependencies install**
   ```bash
   !pip install patchify keras==2.15.0 tensorflow==2.15.0 opencv-python tqdm pillow scikit-learn random pickle keras-segmentation segmentation_models==1.0.1 efficientnet==1.1.1 
   ```

---

## Data Loading & Preprocessing

Open `Drone_Image_Segmenter.ipynb` and run:

1. **Mount your drive** (for Colab):
   ```python
   from google.colab import drive
   drive.mount('/content/gdrive')
   ```

2. **Load Images & Masks**:
   ```python
   from patchify import patchify
   import cv2, os

   def data_loader(folder_dir):
       image_dataset = []
       file_list = sorted(os.listdir(folder_dir))
   
       for images in file_list:
           image_path = os.path.join(folder_dir, images)
           image = cv2.imread(image_path, 1)
           if image is not None:
               image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
               image = cv2.resize(image, (128, 128))
               image = Image.fromarray(image)
               image = np.array(image)
               image_dataset.append(image)
           else:
               print(f"Warning: Could not load {image_path}")
   
       print(f"Loaded {len(image_dataset)} images from {folder_dir}")
       return image_dataset
   
   image_dataset = np.array(data_loader('/content/gdrive/MyDrive/Project/dataset/Images'))
   mask_dataset  = np.array(data_loader('/content/gdrive/MyDrive/Project/dataset/Annotations'))
   ```

3. **Visual Sanity Check**:
   ```python
   import matplotlib.pyplot as plt
   image_number = random.randint(0, len(mask_dataset)-1)
   plt.figure(figsize=(12, 6))
   plt.subplot(121)
   plt.title(f'Original Image {image_number}')
   plt.imshow(image_dataset[image_number])
   plt.subplot(122)
   plt.title(f'Corresponding Mask {image_number}')
   plt.imshow(mask_dataset[image_number])
   plt.show()
   ```

---

## Mask Encoding & One-Hot Conversion

1. **Load Class-RGB Mapping**:
   ```python
   import pandas as pd
   mask_labels = pd.read_csv('/content/gdrive/MyDrive/Project/class_dict.csv')
   ```

2. **Convert RGB Masks to Label Indices**:
   ```python
   def rgb_to_labels(img, mask_labels):
       label_seg = np.zeros(img.shape, dtype=np.uint8)
       for i in range(mask_labels.shape[0]):
           label_seg[np.all(img == list(mask_labels.iloc[i, [1,2,3]]), axis=-1)] = i
       label_seg = label_seg[:,:,0]
       return label_seg

   labels = []
   for i in range(mask_dataset.shape[0]):
       label = rgb_to_labels(mask_dataset[i], mask_labels)
       labels.append(label)
       
   labels = np.array(labels)
   labels = np.expand_dims(labels, axis=3)
   
   print("Unique labels in dataset:", np.unique(labels))   
   ```

3. **One-Hot Encode**:
   ```python
   # One-hot encoding
   n_classes = len(np.unique(labels))
   labels_cat = to_categorical(labels, num_classes=n_classes)
   ```

---

## Train/Test Split

   ```python
   from sklearn.model_selection import train_test_split
   
   X_train, X_test, y_train, y_test = train_test_split(
       image_dataset, labels_cat, test_size=0.20, random_state=42
   )
   ```

---

## Model Definition & Training

1. **Set Backbone & Preprocessing**:
   ```python
   import segmentation_models as sm
   sm.set_framework('tf.keras')
   BACKBONE = 'resnet34'
   preprocess_input = sm.get_preprocessing(BACKBONE)

   X_train_prepr = preprocess_input(X_train)
   X_test_prepr = preprocess_input(X_test)
   ```

2. **Build UNet**:
   ```python
   model = sm.Unet(
       BACKBONE,
       encoder_weights='imagenet',
       classes=n_classes,
       activation='softmax'
   )
   ```

3. **Compile & Train**:
   ```python
   model.compile(
       optimizer='adam',
       loss='categorical_crossentropy',
       metrics=['accuracy']
   )

   history = model_resnet_backbone.fit(
       X_train_prepr,
       y_train,
       batch_size=16,
       epochs=100,
       verbose=1,
       validation_data=(X_test_prepr, y_test)
   )
   ```

4. **Training Curves**:
   ```python
   import matplotlib.pyplot as plt
   plt.plot(history.history['accuracy'], label='train_acc')
   plt.plot(history.history['val_accuracy'], label='val_acc')
   plt.legend(); plt.show()
   ```

---

## Evaluation & Inference

1. **Load Best Model**:
   ```python
   from tensorflow.keras.models import load_model
   model = load_model('models/resnet_backbone.hdf5')
   ```

2. **Predict & Visualize**:
   ```python
   idx = np.random.randint(len(X_test))
   y_pred = model_resnet_backbone.predict(X_test_prepr)
   y_pred_argmax = np.argmax(y_pred, axis=3)
   y_test_argmax = np.argmax(y_test, axis=3)
   
   plt.figure(figsize=(15,5))
   plt.subplot(1,3,1); plt.imshow(X_test[idx]);       plt.title('Input')
   plt.subplot(1,3,2); plt.imshow(y_true_idx);       plt.title('Ground Truth')
   plt.subplot(1,3,3); plt.imshow(y_pred_idx);       plt.title('Prediction')
   plt.show()
   ```

---

## Results & Metrics

- **Training Accuracy**: typically ~93% on the dataset (monitor for overfitting).  
- **Validation Accuracy**: best recorded at ~80% (varies by data).  
- **MeanIoU**: can be computed via:
  ```python
  from tensorflow.keras.metrics import MeanIoU
  miou = MeanIoU(num_classes=n_classes)
  miou.update_state(y_test_argmax.flatten(), y_pred_argmax.flatten())
  print("Mean IoU =", miou.result().numpy())
  ```

---

## Usage
  
1. **Adjust** paths in the notebook (e.g., mount points).  
2. **Run** cells in `Project_Exhibition_1.ipynb` sequentially.  
3. **Monitor** GPU usage—training can be accelerated on a GPU runtime (e.g., Colab, AWS).

---

## Contributing

Feel free to submit issues or pull requests. Fork the repo, create a branch, and open a PR.

---
