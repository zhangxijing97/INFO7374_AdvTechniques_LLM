# INFO7374_AdvTechniques_LLM

## Table of Contents

1. [Lecture 1](#lecture-1)
2. [Lecture 2](#lecture-2)
3. [Lecture 3](#lecture-3)
4. [Lecture 4](#lecture-4)
5. [Lecture 5](#lecture-5)
5. [Lecture 6](#lecture-6)

# Lecture 1

## Supervised Learning and K-Nearest Neighbors (KNN)

1. **Training the Model with `train_x` and `train_y`**:
   - Using `knn.fit(train_x, train_y)`, we store the `train_x` data points and their corresponding labels from `train_y`. KNN doesn’t learn parameters but simply stores the training data.

2. **Predicting Labels for `test_x`**:
   - When calling `knn.predict(test_x)`, each point in `test_x` is compared to all points in `train_x` to find its \(k\) nearest neighbors.
   - **Classification Decision**:
     - Based on the labels of these \(k\) nearest neighbors in `train_x`, the model assigns the most common label to the `test_x` point.

### Summary
- `train_x` serves as the reference set to find neighbors.
- `test_x` is the set of new data points for which we predict labels based on the majority vote among the \(k\) nearest neighbors in `train_x`.

### K-Nearest Neighbors (KNN) Example

Suppose we have a dataset of animals, each labeled as either a "Dog" or "Cat," based on their height and weight.

#### 1. Dataset
- **Features**: Height (cm) and Weight (kg)
- **Labels**: Dog or Cat

Our dataset includes:

| Height (cm) | Weight (kg) | Label |
|-------------|-------------|-------|
| 50          | 8           | Dog   |
| 55          | 9           | Dog   |
| 60          | 11          | Dog   |
| 45          | 5           | Cat   |
| 48          | 6           | Cat   |
| 52          | 7           | Cat   |

#### 2. New Data Point
We have a new animal with a height of 53 cm and a weight of 8 kg. We want to classify it as either a Dog or Cat using **K=3**.

#### 3. Calculate Distances
Calculate the Euclidean distance between the new point and each existing point in the dataset:

- Distance to (50, 8) = \( \sqrt{(53-50)^2 + (8-8)^2} = 3 \)
- Distance to (55, 9) = \( \sqrt{(53-55)^2 + (8-9)^2} \approx 2.24 \)
- Distance to (60, 11) = \( \sqrt{(53-60)^2 + (8-11)^2} \approx 7.62 \)
- Distance to (45, 5) = \( \sqrt{(53-45)^2 + (8-5)^2} \approx 8.54 \)
- Distance to (48, 6) = \( \sqrt{(53-48)^2 + (8-6)^2} \approx 5.39 \)
- Distance to (52, 7) = \( \sqrt{(53-52)^2 + (8-7)^2} \approx 1.41 \)

#### 4. Find Nearest Neighbors
The 3 nearest points to (53, 8) are:
- (52, 7) labeled "Cat"
- (55, 9) labeled "Dog"
- (50, 8) labeled "Dog"

#### 5. Prediction
With \( K=3 \), the most common class among the nearest neighbors is "Dog" (2 out of 3). Therefore, we classify the new data point as a **Dog**.

---

### Key Points of KNN
- **Interpretability**: KNN is highly interpretable as it bases predictions directly on existing data.
- **Non-Parametric**: It makes no assumptions about the underlying data distribution.
- **Computational Cost**: KNN can be slow with large datasets as it computes distances for each prediction.

KNN is ideal for smaller datasets where relationships are straightforward, and similarity in feature space is an effective way to predict labels.


```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

#Input Dateset
org_df = pd.read_csv("/Users/zhangxijing/MasterNEU/INFO7374_AdvTechniques_LLM/Datasets/heart_attack.csv")

#Define features and label for KNN
label_df =  org_df.loc[:,org_df.columns == 'heart_attack']
feat_df =  org_df.loc[:,org_df.columns != 'heart_attack']

#Normalize Data
feat_df = (feat_df - feat_df.mean()) / feat_df.std()

#Seperate test and train data
train_x,test_x,train_y,test_y = train_test_split(feat_df,label_df,test_size=0.25)

# KNN with 3 neighbors
knn_3 = KNeighborsClassifier(n_neighbors=3)
knn_3.fit(train_x, train_y)
test_pred_y_3 = knn_3.predict(test_x)

# Accuracy for KNN-3
accuracy_knn_3 = accuracy_score(test_y, test_pred_y_3)
print(f"Accuracy of KNN-3: {accuracy_knn_3:.4f}")
```

## Lecture 2