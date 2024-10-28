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
   - Classification Decision: The model assigns each `test_x` point the most common label among its \(k\) nearest neighbors in `train_x`. For example, The 3 nearest points in example is Dog (2/3), so we classify the new data point as a Dog.

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

# Lecture 2

## Logistic Regression

Logistic Regression is a statistical method for **binary classification** tasks, where the outcome is categorical (e.g., spam or not spam). It’s particularly useful for situations where the dependent variable is binary.

#### Key Concepts
- **Sigmoid Function**: Maps output values to a range between 0 and 1, making it ideal for representing probabilities. The sigmoid function is defined as:

  `σ(z) = 1 / (1 + e^(-z))`

  where `z` is a linear combination of input features.

- **Binary Classification**: Logistic regression can predict the probability `P(Y=1|x)` that an input `x` belongs to the positive class. A threshold (e.g., 0.5) is used for classification:
  - If the probability > 0.5, classify as Class 1.
  - Otherwise, classify as Class 0.

#### Example of Logistic Regression
Suppose we want to predict whether an email is spam based on the frequency of certain keywords. Features could include:
- **Feature 1**: Count of the word "free"
- **Feature 2**: Count of the word "win"

After training, the model might give the equation:

`Spam = σ(0.5 * Free_Count + 0.8 * Win_Count - 1.0)`

For classification:
1. **If `σ(1.6) ≈ 0.832`** (for input values `Free_Count=2` and `Win_Count=1`), classify as **spam** (since 0.832 > 0.5).
2. **If `σ(-0.5) ≈ 0.377`** (for input values `Free_Count=1` and `Win_Count=0`), classify as **not spam** (since 0.377 < 0.5).

This example demonstrates how logistic regression can be used to calculate the probability of belonging to a certain class and classify based on a threshold.

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the dataset
org_df = pd.read_csv("/Users/zhangxijing/MasterNEU/INFO7374_AdvTechniques_LLM/Datasets/heart_attack.csv")

#Define features and label
label_df =  org_df.loc[:,org_df.columns == 'heart_attack']
feat_df =  org_df.loc[:,org_df.columns != 'heart_attack']

# Split dataset into training (75%) and test (25%) data
train_x, test_x, train_y, test_y = train_test_split(feat_df, label_df, test_size=0.25, random_state=42)

# Create and train the Logistic Regression model
model = LogisticRegression(max_iter=10000)  # Increase max_iter if convergence issues arise
model.fit(train_x, train_y)

# Make predictions on the test data
test_pred_y = model.predict(test_x)

# Calculate accuracy of the model for test data
test_accuracy = accuracy_score(test_y, test_pred_y)
print("Test Accuracy:", test_accuracy)
```