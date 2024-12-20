# INFO7374_AdvTechniques_LLM

## Table of Contents

1. [Lecture 1](#lecture-1)
   - [Supervised Learning and Unsupervised Learning](#supervised-learning-and-unsupervised-learning)
   - [K Nearest Neighbor (KNN)](#k-nearest-neighbor-knn)
2. [Lecture 2](#lecture-2)
3. [Lecture 3](#lecture-3)
4. [Lecture 4](#lecture-4)
   - [Self-consistency](#self-consistency)
5. [Lecture 5](#lecture-5)
6. [Lecture 6](#lecture-6)
7. [Lecture 7](#lecture-7)
   - [Calibration](#calibration)
8. [Lecture 8](#lecture-8)
   - [Evolution of LLMs](#evolution-of-llms)
   - [Learning Techniques](#learning-techniques)
     - [Zero-shot Learning](#zero-shot-learning)
     - [One-shot Learning](#one-shot-learning)
     - [Few-shot Learning](#few-shot-learning)
   - [Prompt Engineering](#prompt-engineering)
     - [In-Context Learning](#in-context-learning)
     - [Chain of Thought Prompting](#chain-of-thought-prompting)
     - [Program-Aided Language Model (PAL)](#program-aided-language-model-pal)
     - [Self-Consistency](#self-consistency)
     - [Tree of Thought Prompting](#tree-of-thought-prompting)
9. [Lecture 9](#lecture-9)
   - [RAG (Vector Database)](#rag-vector-database)


# Lecture 1

## Supervised Learning and Unsupervised Learning

| **Type**               | **Training Data**  | **Purpose**                    | **Common Uses**                           |
|------------------------|--------------------|--------------------------------|-------------------------------------------|
| **Supervised**         | Labeled            | Predict known outputs          | Classification, regression                |
| **Unsupervised**       | Unlabeled          | Discover data patterns         | Clustering, dimensionality reduction      |

## K Nearest Neighbor (KNN)

- KNN is a supervised learning technique (Classifier).
- Classify a new data point by calculating the distance between existing training examples.
- Similar means similar feature values (nearest feature values)

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

- Distance to (50, 8) = sqrt((53-50)^2 + (8-8)^2) = 3
- Distance to (55, 9) = sqrt((53-55)^2 + (8-9)^2) = 2.24
- Distance to (60, 11) = sqrt((53-60)^2 + (8-11)^2) = 7.62
- Distance to (45, 5) = sqrt((53-45)^2 + (8-5)^2) = 8.54
- Distance to (48, 6) = sqrt((53-48)^2 + (8-6)^2) = 5.39
- Distance to (52, 7) = sqrt((53-52)^2 + (8-7)^2) = 1.41

#### 4. Find Nearest Neighbors
The 3 nearest points to (53, 8) are:
- (52, 7) labeled "Cat"
- (55, 9) labeled "Dog"
- (50, 8) labeled "Dog"

#### 5. Prediction
With K=3 , the most common class among the nearest neighbors is "Dog" (2 out of 3). Therefore, we classify the new data point as a **Dog**.

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

### Decoding Strategies

#### **1. Greedy Search**
- **Definition**: Chooses the most probable next word at each step.
- **Example**: For the prompt "I feel," it might always choose "good" as the next word because it's the most common continuation.

#### **2. Beam Search**
- **Definition**: Considers several top possibilities at each step, keeping the best sequences based on cumulative probabilities.
- **Example**: For "I feel," it might keep sequences like "I feel happy," "I feel good," and "I feel tired," and choose the best at the end.

#### **3. Top-k Sampling**
- **Definition**: Randomly selects the next word from the top k most likely candidates.
- **Example**: If "happy," "good," and "great" are the top-3 words, top-k sampling might randomly pick "great" even if it's not the highest probability.

#### **4. Nucleus Sampling (Top-p Sampling)**
- **Definition**: Chooses the next word from a dynamically sized set of top candidates that together exceed a probability threshold p.
- **Example**: It might consider words that cumulatively make up 90% probability, choosing among "happy," "good," "great," depending on which set crosses the threshold first.

#### **5. Temperature Setting**
- **Definition**: Adjusts the probability distribution of the next word, affecting the randomness of the choice.
- **Example**: With a high temperature, less likely words might be chosen, leading to "I feel ecstatic" instead of just "good."

Each strategy has its applications depending on the desired balance between randomness and accuracy in text generation.

# Lecture 7

## Calibration

### **Definition of LLM Calibration**
LLM Calibration is about making sure that when a language model says it's confident about an answer, that answer is likely to be correct. This process aligns the model's confidence with the true accuracy of its predictions.

### **Why Calibration is Important**
Calibration is crucial because it ensures that:
- Confident predictions are trustworthy.
- The model acknowledges when it's unsure about an answer.

### **How to Measure Calibration**
Calibration is typically measured using the **Expected Calibration Error (ECE)**, which looks at the difference between how confident the model is and how often it's actually right. This is done by:
1. Dividing predictions into groups based on confidence levels.
2. Comparing the real accuracy of each group to its confidence.
3. Averaging these differences to get the ECE.

### **Techniques for Improving Calibration**
To improve calibration, you can use:
- **Temperature Scaling**: Adjusts how the model calculates confidence, making it more or less sure about its answers.
- **Platt Scaling**: Uses a simple formula to adjust the predicted probabilities, making them match up better with real-world outcomes.
- **Ensemble Methods**: Combines predictions from several models to make overall predictions more reliable and well-calibrated.

### **Practical Example**
In a medical diagnosis tool, calibration makes sure that if a model is 90% confident in a diagnosis, it should be correct about 90% of the time. This is vital in healthcare, where being wrong could have serious consequences.

# Lecture 8

## Evolution of LLMs

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

## Learning Techniques

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

## Prompt Engineering

### In-Context Learning

In-context learning is when a model uses the context within a prompt to perform tasks it wasn't explicitly trained to do.<br>

Zero-shot Learning, One-shot Learning, and Few-shot Learning are types of in-context learning.<br>

- **Definition**: Employs the model's ability to understand and perform tasks based on context provided within the prompt, adapting based on provided examples.
- **Example**:
  - **Prompt**: "Solve the problem step-by-step. Start from the first left character of the word 'strawberry'. If the character is 'r', add one to the total number of 'r's. Repeat until the last character of the word. How many 'r's are in the word 'strawberry'?"
  - **Response**:
    1. "s: Not 'r,' so the count is 0."
    2. "t: Not 'r,' so the count remains 0."
    3. "r: This is 'r,' add 1. Count = 1."
    4. "a: Not 'r,' count remains 1."
    5. "w: Not 'r,' count remains 1."
    6. "b: Not 'r,' count remains 1."
    7. "e: Not 'r,' count remains 1."
    8. "r: This is 'r,' add 1. Count = 2."
    9. "r: This is 'r,' add 1. Count = 3."
    10. "y: Not 'r,' count remains 3."
    - **Final Answer**: "There are 3 'r's in the word 'strawberry'."

### Chain of Thought Prompting
- **Definition**: Breaks a task into intermediate steps, enhancing reasoning by solving problems step-by-step.
- **Example**:
  - **Prompt**: "Solve step-by-step: Calculate the area of a right triangle with a base of 3 meters and a height of 4 meters."
  - **Response**:
    1. "First, I recognize that the formula for the area of a right triangle is `Area = 0.5 * base * height`."
    2. "The base of the triangle is 3 meters, and the height is 4 meters."
    3. "Plugging in the values: `Area = 0.5 * 3 * 4`."
    4. "Perform the multiplication: `Area = 0.5 * 12 = 6`."
    5. "Therefore, the area of the triangle is 6 square meters."
    - **Answer**: "6 square meters."

### Program-Aided Language Model (PAL)
**Program-Aided Language Models (PAL)** combine LLM reasoning with programmatic tools to solve tasks requiring logic, calculations, or structured problem-solving.

#### Steps
1. **Combining Natural Language and Code**: LLMs generate code to solve sub-tasks, which is executed in an external programming environment.
2. **Enhanced Accuracy**: Results are programmatically verified, reducing errors in reasoning.
3. **Transparency**: Intermediate steps provide interpretable solutions.

#### Workflow
1. **Task Decomposition**: LLM breaks the problem into smaller steps.
2. **Code Generation**: LLM generates code for each step.
3. **Code Execution**: The code runs in a programming environment.
4. **Output Integration**: Results are combined into a coherent response.

#### Example
**Task**: How many 'r's are in "strawberry"?  
**Solution Using PAL**:
1. **Decomposition**: Identify the task (count occurrences of "r").
2. **Code Generation**:
   ```python
   def char_count(s, char):
       return s.count(char)
   result = char_count("strawberry", "r")
3. **Execution**: Run the function to get `3`.
4. **Response**: "The word 'strawberry' contains 3 'r's."

#### Advantages
- **Precision**: Ensures accurate results.
- **Explainability**: Provides interpretable intermediate steps.
- **Flexibility**: Solves complex tasks programmatically.

#### Limitations
- Requires external environments for execution.
- Errors in generated code can propagate.
- Not effective for tasks unsuited to programmatic solutions.

### Least-to-Most Prompting
- **Definition**: Decomposes a complex task into simpler sub-tasks, solving progressively harder problems.
- **Example**:
  - **Prompt**: "Let’s break down the problem of finding the area of a right triangle into simpler steps, starting with understanding the formula and then calculating progressively."
  - **Step 1**: "What is the formula for the area of a right triangle?"
    - **Response**: "The formula is `Area = 0.5 * base * height`."
  - **Step 2**: "Given a base of 3 meters, how would you find the half of this base?"
    - **Response**: "Half of the base is `0.5 * 3 = 1.5` meters."
  - **Step 3**: "Now multiply the result from Step 2 by the height of 4 meters to find the area."
    - **Response**: "The area is `1.5 * 4 = 6` square meters."
  - **Final Step**: "What is the total area of the triangle?"
    - **Final Answer**: "The area of the triangle is 6 square meters."

### Self-Consistency
- **Definition**: Generates multiple reasoning paths and selects the most consistent answer to ensure reliability.
- **Example**:
  - **Prompt**: "How many 'r's are in 'strawberry'?"
  - **Paths**:
    - Path 1: "2 'r's."
    - Path 2: "3 'r's."
    - Path 3: "3 'r's."
  - **Final Answer**: "3 'r's (majority vote)."

### Tree of Thought Prompting
- **Definition**: Organizes reasoning paths hierarchically in a tree structure, exploring multiple branches for solutions.
- **Example**:
  - **Task**: "How can we reduce carbon emissions?"
  - **Branches**:
    - Renewable energy (solar, wind).
    - Energy efficiency (buildings, behaviors).
    - Reforestation (planting trees).
  - **Response**: Evaluates each branch for feasibility and selects the best.

### CoT vs SC vs ToT
![This is an example image](Image/image01.jpg)

## Reasoning in LLMs

### Logical Reasoning Limitations
- LLMs do not "understand" in a human sense. They mimic reasoning patterns based on their training data but may fail when faced with problems requiring deep logical or step-by-step reasoning.
- Complex reasoning tasks can result in errors due to:
  - Ambiguity in prompts.
  - Lack of genuine logical inference abilities.
  - Limited context provided in the input.

### Static Knowledge Due to Pretraining Cutoffs
- LLMs are trained on static datasets up to a certain point in time, meaning they lack knowledge of events or updates that occurred after their training data cutoff.
- This results in inaccurate or outdated responses when the task involves recent information.

### Solutions

#### External Knowledge Integration
- Use external tools, databases, or APIs to provide up-to-date or domain-specific information.
- Example: Integrating Wikipedia, news APIs, or custom knowledge bases to augment responses.

#### Symbolic Reasoning
- Combine rule-based or symbolic reasoning approaches with LLM outputs to improve logical consistency.
- Symbolic reasoning enables the model to manipulate and reason with abstract concepts like mathematical symbols, logical operators, or structured rules.

#### Hybrid Approaches
- Combine LLM capabilities with programmatic reasoning (e.g., using Program-Aided Language Models, or PAL).
- Incorporate methods like Chain-of-Thought (CoT) or Least-to-Most (L2M) prompting for better reasoning transparency.

### Examples

#### Example 1: External Knowledge Integration
**Task**: Identify the current President of the United States.  
- **Issue**: A pre-trained model may provide outdated information (e.g., knowledge cutoff in 2021).  
- **Solution**:  
  - Query an external API (e.g., a news or government database) for real-time information.  
  - Combine the LLM's natural language generation with the API's data:  
    - **Prompt**: "Using external resources, who is the current President of the United States?"  
    - **Response**: "As of now, the President is Joe Biden. This was confirmed by querying the U.S. government database."

#### Example 2: Symbolic Reasoning
**Task**: Solve the equation `2x + 3 = 7`.  
- **Issue**: The model may return a single-step solution without clarity or make errors if reasoning chains are ambiguous.  
- **Solution**:  
  - Use symbolic reasoning or PAL to structure the solution programmatically:  
    - **Prompt**: "Solve step-by-step using symbolic reasoning. What is x if 2x + 3 = 7?"  
    - **Response**:  
      ```
      Step 1: Subtract 3 from both sides: 2x = 4.
      Step 2: Divide both sides by 2: x = 2.
      Answer: x = 2.
      ```

#### Example 3: Hybrid Reasoning with Symbolic and LLM Outputs
**Task**: Calculate the sale price of a $100 item with a 20% discount.  
- **Issue**: An LLM alone might make errors in interpreting the calculation.  
- **Solution**:  
  - Use an LLM to interpret the task and a program to perform the calculation:  
    - **Prompt**: "Describe the calculation step-by-step for clarity."  
    - **Response**:  
      ```
      Step 1: Identify the discount amount (20% of $100): 0.2 × 100 = $20.
      Step 2: Subtract the discount from the original price: $100 - $20 = $80.
      Answer: The sale price is $80.
      ```

# Lecture 9

## RAG (Vector Database)
