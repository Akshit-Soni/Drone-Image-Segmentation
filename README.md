# Drone Image : Semantic Segmentation with ResNet34-UNet

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue)](https://www.python.org/)  [![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)  [![Build Status](https://img.shields.io/badge/Notebook-Ready-yellow)]()

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
├── Project_Exhibition_1.ipynb   # Main Jupyter notebook
├── requirements.txt            # Python dependencies
├── models/                     # Trained model checkpoints
│   └── resnet_backbone.hdf5
├── dataset/                    # (Not included)  
│   ├── Images/                 # Raw input images
│   └── Annotations/            # Corresponding segmentation masks
├── class_dict.csv              # CSV mapping class names to RGB values
├── README.md                   # This file
└── LICENSE                     # Project license (MIT)
```

> **Note:** The `dataset/` folder is **not** included in the repo.  
> Please mount or symlink your data in the notebook (e.g., via Google Drive).

---

## Environment Setup

1. **Clone the repository**  
   ```bash
   git clone https://github.com/yourusername/semantic-segmentation.git
   cd semantic-segmentation
   ```

2. **Create & activate a virtual environment**  
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**  
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Additional pip installs** (if not in `requirements.txt`):  
   ```bash
   pip install patchify keras tensorflow opencv-python keras-segmentation segmentation_models
   ```

---

## Data Loading & Preprocessing

Open `Project_Exhibition_1.ipynb` and run:

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
       images = []
       for filename in os.listdir(folder_dir):
           img = cv2.imread(os.path.join(folder_dir, filename))
           img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
           images.append(img)
       return images

   image_dataset = np.array(data_loader('/content/gdrive/MyDrive/Project/dataset/Images'))
   mask_dataset  = np.array(data_loader('/content/gdrive/MyDrive/Project/dataset/Annotations'))
   ```

3. **Visual Sanity Check**:
   ```python
   import matplotlib.pyplot as plt
   idx = np.random.randint(len(image_dataset))
   plt.subplot(1,2,1); plt.imshow(image_dataset[idx])
   plt.subplot(1,2,2); plt.imshow(mask_dataset[idx]); plt.show()
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
   def rgb_to_labels(img, label_df):
       label_seg = np.zeros(img.shape[:2], dtype=np.uint8)
       for i, row in label_df.iterrows():
           rgb = list(row[['r','g','b']])
           label_seg[np.all(img == rgb, axis=-1)] = i
       return label_seg

   labels = np.array([rgb_to_labels(m, mask_labels) for m in mask_dataset])
   labels = labels[..., np.newaxis]
   ```

3. **One-Hot Encode**:
   ```python
   from tensorflow.keras.utils import to_categorical
   n_classes = len(np.unique(labels))
   labels_cat = to_categorical(labels, num_classes=n_classes)
   ```

---

## Train/Test Split

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    image_dataset, labels_cat,
    test_size=0.2,
    random_state=42
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

   X_train_p = preprocess_input(X_train)
   X_test_p  = preprocess_input(X_test)
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

   history = model.fit(
       X_train_p, y_train,
       validation_data=(X_test_p, y_test),
       epochs=100,
       batch_size=16,
       callbacks=[
           EarlyStopping(patience=10, restore_best_weights=True),
           ModelCheckpoint('models/resnet_backbone.hdf5', save_best_only=True)
       ]
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
   y_pred = model.predict(X_test[idx][None, ...])
   y_pred_idx = np.argmax(y_pred[0], axis=-1)
   y_true_idx = np.argmax(y_test[idx], axis=-1)

   plt.figure(figsize=(15,5))
   plt.subplot(1,3,1); plt.imshow(X_test[idx]);       plt.title('Input')
   plt.subplot(1,3,2); plt.imshow(y_true_idx);       plt.title('Ground Truth')
   plt.subplot(1,3,3); plt.imshow(y_pred_idx);       plt.title('Prediction')
   plt.show()
   ```

---

## Results & Metrics

- **Training Accuracy**: typically > 90% on small datasets (monitor for overfitting).  
- **Validation Accuracy**: best checkpoint saved at ~X% (varies by data).  
- **MeanIoU**: can be computed via:
  ```python
  from tensorflow.keras.metrics import MeanIoU
  miou = MeanIoU(num_classes=n_classes)
  miou.update_state(y_true_idx.flatten(), y_pred_idx.flatten())
  print("Mean IoU =", miou.result().numpy())
  ```

---

## Usage

1. **Populate** `dataset/Images` and `dataset/Annotations`.  
2. **Adjust** paths in the notebook (e.g., mount points).  
3. **Run** cells in `Project_Exhibition_1.ipynb` sequentially.  
4. **Monitor** GPU usage—training can be accelerated on a GPU runtime (e.g., Colab, AWS).

---

## Contributing

Feel free to submit issues or pull requests. Fork the repo, create a branch, and open a PR.

---

## License

Released under the **MIT License**. See [LICENSE](LICENSE) for details.
