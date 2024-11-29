# INFO7374_AdvTechniques_LLM

## Table of Contents

1. [Lecture 1](#lecture-1)
2. [Lecture 2](#lecture-2)
3. [Lecture 3](#lecture-3)
4. [Lecture 4](#lecture-4)
5. [Lecture 5](#lecture-5)
6. [Lecture 6](#lecture-6)
7. [Lecture 7](#lecture-7)
8. [Lecture 8](#lecture-8)
9. [Lecture 9](#lecture-9)

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

# Lecture 3

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Input

# Input Dataset
org_df = pd.read_csv("/Users/zhangxijing/MasterNEU/INFO7374_AdvTechniques_LLM/Datasets/income_ds.csv")

# Labels and Features
label_df = org_df.loc[:, org_df.columns == 'High_Income']
feat_df = org_df.loc[:, org_df.columns != 'High_Income']

# Encode Categorical Variables
feat_df = pd.get_dummies(feat_df, dtype='int')

# Normalize Features
feat_df = (feat_df - feat_df.mean()) / feat_df.std()

# Split Train and Test Data (75% Train, 25% Test)
x_train, x_test, y_train, y_test = train_test_split(feat_df, label_df, test_size=0.25, random_state=42)

# Model 1: One Hidden Layer with 6 Units
model_1 = Sequential()
model_1.add(Dense(units=6, activation='relu', input_shape=(x_train.shape[1],)))
model_1.add(Dense(units=1, activation='sigmoid'))

# Set Optimizer, Loss Function, and Metrics
model_1.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train Model 1
model_1.fit(x_train, y_train, epochs=500, verbose=1)
print(model_1.summary())

# Accuracy of Model 1 on Test Data
loss_1, accuracy_1 = model_1.evaluate(x_test, y_test)
print('Model 1 - accuracy=', accuracy_1, ' , loss=', loss_1)

# Compare the Results
print('Comparison:')
print('Model 1: One Hidden Layer with 6 Units - accuracy=', accuracy_1, ', loss=', loss_1)
```

# Lecture 4

### Byte-Pair Encoding (BPE) Example

BPE is a tokenization technique that merges frequent character pairs in text to create a compact vocabulary, allowing NLP models to handle complex and rare words more efficiently.

#### Example Process
1. **Initial Representation**: Start with individual characters as tokens for each word.
   - Training data: ["low", "lower", "lowest"]
   - Initial tokens:
     - `low` = `[l, o, w]`
     - `lower` = `[l, o, w, e, r]`
     - `lowest` = `[l, o, w, e, s, t]`

2. **Merge Frequent Pairs**:
   - The most frequent pair `[l, o]` is merged to form `lo`:
     - `low` = `[lo, w]`
     - `lower` = `[lo, w, e, r]`
     - `lowest` = `[lo, w, e, s, t]`

   - Next, `[lo, w]` is the most frequent, so merge it to form `low`:
     - `low` = `[low]`
     - `lower` = `[low, e, r]`
     - `lowest` = `[low, e, s, t]`

3. **Repeat Until Threshold**:
   - Continue merging pairs until reaching a maximum vocabulary size or no pairs meet the frequency threshold.

#### Purpose
BPE builds a vocabulary of common subwords, making it efficient for models to handle large vocabularies and recognize parts of rare words by breaking them into familiar subunits.

min_frequency=2: Only merges character pairs that appear at least twice in the corpus.<br>
vocab_size=30: Limits the vocabulary size to 30 tokens, meaning only the most frequent pairs are retained.<br>

```python
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers import CharBPETokenizer

# Instantiate tokenizer
tokenizer = CharBPETokenizer()

tokenizer.train([ "/Users/zhangxijing/MasterNEU/INFO7374_AdvTechniques_LLM/Datasets//sample.txt"],min_frequency=2, vocab_size=30)

print(tokenizer.get_vocab())

output = tokenizer.encode("highest")
print(output.tokens)

output = tokenizer.encode("newer")
print(output.tokens)
print(output.ids)

output = tokenizer.encode("higher is better")
print(output.tokens)
```

# Lecture 5

**Word2Vec** is a technique that creates vector representations of words, capturing semantic relationships based on context. It enables words with similar meanings to have similar vectors, useful for NLP tasks.

### Key Concepts

- **Models**:
  - **CBOW**: Predicts target words from surrounding words, faster and good for common words.
  - **Skip-Gram**: Predicts surrounding words from a target word, effective for rare words.
  
- **Window Size**: Determines the context range. Smaller sizes capture local context, larger ones capture broader relationships.

- **Advantages**:
  - **Dense Vectors**: Efficient, capturing rich semantics.
  - **Semantic Similarity**: Similar words are close in vector space, revealing relationships.

- **Limitations**: Static embeddings; each word has one vector regardless of context (e.g., "bank" as a place vs. riverbank).

Word2Vec is used in NLP tasks like sentiment analysis and document similarity for its ability to represent word relationships.

### The instruction is to train a **Word2Vec model on the text8 dataset** using different combinations of these hyper-parameters:

- **`win_size`**: Context window size around each word, with values `[3, 7, 13, 25]`.
  - **Smaller values** (3) capture close word relationships.
  - **Larger values** (25) capture broader context.

- **`vector_size`**: Dimensionality of word embeddings, with values `[20, 70, 100, 300]`.
  - **Smaller sizes** (20) are computationally cheaper but less detailed.
  - **Larger sizes** (300) capture richer semantics but need more resources.

With 4 values for each parameter, **16 combinations** are tested to find the best setup for the text8 dataset.
```python
from gensim.models.word2vec import Word2Vec
import gensim.downloader as api
from sklearn.cluster import KMeans
import numpy as np

corpus = api.load('text8')

# Train Word2Vec model
model = Word2Vec(sentences=corpus, window=5, vector_size=70)

print("Embedding vector for Paris is: ", model.wv['paris'])

print('Similar to France: ', model.wv.similar_by_vector (model.wv['france'],topn=3))
print('Similar to Paris: ', model.wv.similar_by_vector (model.wv['paris'],topn=3))

# Find most similar embeddings to a transformed embedding
transform = model.wv['france'] - model.wv['paris']
print('Transform: ', model.wv.similar_by_vector ( transform + model.wv['madrid'] ,topn=3))

# Some word embeddings
embeddings =np.array([
model.wv['paris'] , model.wv['he'],
model.wv['vienna'] , model.wv['she']
])

# K-means clustering
kmeans = KMeans(n_clusters=2)
kmeans.fit(embeddings)

# Print cluster assignments
for i, label in enumerate(kmeans.labels_):
    print("Embedding ", i, " is in cluster ", label)
```

# Lecture 6

### Example with Starting Phrase: "The cat"

#### Greedy Search
- **Process**: Selects the highest probability word at each step without looking ahead.
- **Steps**:
  - Step 1: “is” (0.5)
  - Step 2: “on” (0.6)
  - Step 3: “the” (0.7)
  - Step 4: “mat” (0.8)
- **Result**: “The cat is on the mat.”

#### Beam Search (Beam Width = 2)
- **Process**: Keeps top 2 sequences at each step, considering cumulative probabilities.
- **Steps**:
  - Step 1: Top paths - “is” (0.5), “sat” (0.3)
  - Step 2: Continue with best paths - “is on” (0.3), “sat on” (0.18)
  - Step 3: “is on the” (0.21), “sat on the” (0.126)
  - Step 4: “is on the mat” (0.168), “sat on the mat” (0.1008)
- **Result**: “The cat is on the mat.”

**Summary**:
- **Greedy Search**: Fast but short-sighted.
- **Beam Search**: More thorough, considering multiple paths, often yielding more coherent sequences.

# Lecture 8

## Evolution of LLMs and Techniques

### **1. GPT-1 (117M Parameters)**
- **Architecture**: Transformer decoder with 12 layers.
- **Dataset**: Trained on the BooksCorpus dataset (~4.6GB of text, over 7,000 unique books).
- **Training Approach**:
  - Self-supervised pretraining for predicting the next word in a sequence.
  - Fine-tuned on specific natural language processing (NLP) tasks.
- **Key Achievements**:
  - Demonstrated the potential of large-scale language modeling as an effective pretraining technique for downstream tasks such as question answering and sentiment analysis.

### **2. GPT-2 (1.5B Parameters)**
- **Improvements**:
  - Parameter count increased from 117M to 1.5B.
  - Training dataset grew from ~4GB to ~40GB (WebText corpus of internet text).
- **Capabilities**:
  - Emergence of **zero-shot learning**, where the model generalized to unseen tasks using patterns learned during pretraining.
  - Simulated reasoning by identifying relationships and contextual patterns.
- **Significance**:
  - Highlighted the role of model size and diverse datasets in improving generalization.

### **3. GPT-3 (175B Parameters)**
- **Improvements**:
  - Parameters scaled significantly from 1.5B to 175B.
  - Training data expanded to over 600GB.
- **Capabilities**:
  - Emergence of **few-shot learning**: The ability to infer rules of a task from a small number of examples.
  - Demonstrated advanced reasoning and language capabilities across various domains.
- **Significance**:
  - GPT-3 set a new standard for general-purpose AI with remarkable abilities in text generation, summarization, translation, and more.

### **4. GPT-4 (1.76 Trillion Parameters)**
- **Improvements**:
  - Massive scale-up to 1.76 trillion parameters.
  - Inclusion of third-party datasets and more diverse training sources.
- **Capabilities**:
  - Emergence of **multi-modal reasoning**, allowing the model to process and reason across multiple types of data (e.g., text and images).
  - Enhanced performance in complex and creative tasks.
- **Significance**:
  - Demonstrated state-of-the-art abilities in reasoning, creativity, and adaptability for diverse applications.

### **Key Trends in LLM Evolution**
1. **Scale**: Larger models with more parameters and diverse datasets lead to better performance but come with increased computational demands.
2. **Data**: The quality and diversity of training datasets significantly affect model capabilities.
3. **Capabilities**: Each generation of models introduced new emergent abilities, enabling more complex reasoning and problem-solving.
4. **Applications**: From basic text completion to complex reasoning, LLMs have expanded into coding, multi-modal processing, and interactive systems.

## Learning Paradigms


### Zero-shot Learning
Definition: Enables a model to perform tasks it hasn’t explicitly trained for. It uses knowledge from pretraining and patterns from prompts to generalize to new tasks.<br>
Example:<br>
Prompt: "Is the text about sports or technology? Text: 'The game was exciting.'"<br>
Response: "The text is about sports."<br>
Despite no specific training, the model identifies the category based on contextual patterns.<br>

### One-shot Learning
Definition: The model learns to perform tasks from a single example provided in the prompt.<br>
Example:<br>
Prompt: "Who won the World Cup in 2014? Germany won. Who won in 2018?"<br>
Response: "France won the World Cup in 2018."<br>
The model uses the single example to understand the task format and generalizes.<br>

### Few-shot Learning
Definition: Improves task performance by providing multiple examples in the prompt. The model infers patterns and applies them to similar tasks.<br>
Example:<br>
Prompt:<br>
"Who won in 2010? Spain.<br>
Who won in 2014? Germany.<br>
Who won in 2028?"<br>
Response: "France won in 2022."<br>
Multiple examples help ensure consistent and accurate responses.<br>