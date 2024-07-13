# K-Nearest Neighbors (KNN)

## Data Used
**psyc.csv**

## Source of Data
[Personality Classification Prediction Data on Kaggle](https://www.kaggle.com/code/brsdincer/personality-classification-prediction/data)

## Data Information

The dataset contains 8 features:
- **gender**: Categorical feature with values 'male' and 'female'.
- **age**: Numerical feature (1-10).
- **openness**: Numerical feature (1-10).
- **neuroticism**: Numerical feature (1-10).
- **conscientiousness**: Numerical feature (1-10).
- **agreeableness**: Numerical feature (1-10).
- **extraversion**: Numerical feature (1-10).

The target variable is **personality**, which is a categorical feature with five unique values:
- serious
- extraverted
- responsible
- lively
- dependable

The dataset consists of 315 instances.

## Problem Statement

For this activity, we will implement two additional distance calculation methods for KNN: Manhattan and Minkowski distances. The challenge is to implement these methods in Python.

## Method

We will use mathematical formulas and Python modules to implement Manhattan and Minkowski distance calculations.

## Algorithm

1. **Manhattan Distance Calculation**: Calculate all Manhattan distances of an instance in the test set from each instance in the train set using the formula: 
   \[
   \text{Manhattan Distance} = \sum_{i=0}^{n} |x_i - y_i|
   \]

2. **Minkowski Distance Calculation**: Calculate all Minkowski distances of an instance in the test set from each instance in the train set using the formula:
   \[
   \text{Minkowski Distance} = \left( \sum_{i=0}^{n} |x_i - y_i|^p \right)^{1/p}
   \]
   where \( p = 1 \).

3. **Sort Distances**: Sort all distances from Manhattan and Minkowski methods from least to greatest.

4. **Get Top k Distances**: Select the top k distances.

5. **Select Most Common Class**: Determine the most common class among the top k distances.

6. **Classify All Test Instances**: Apply the above steps to classify all instances in the test set.

7. **Evaluate Model**: Evaluate the model's accuracy and plot its predictions.

## Solution

### Steps 1 and 2: Distance Calculation

These steps were implemented using NumPy functions:

```python
import numpy as np

# Manhattan distance
manhattan_distance = np.sum(np.abs(train_data - test_data), axis=1)

# Minkowski distance with p=1 (equivalent to Manhattan distance)
minkowski_distance = np.sum(np.abs(train_data - test_data)**1, axis=1)**(1/1)
```

### Steps 3 and 4: Sorting and Selecting Top k Distances

This was implemented using sort and slice functions, then the most common class was taken using the `Counter` function from the `collections` module.

### Step 5: Classification

Implemented using the `apply` function in pandas.

### Step 6: Evaluation

To evaluate the model, we calculated its accuracy using the `accuracy_score` function from the `sklearn.metrics` module and plotted the results using `matplotlib.pyplot`.

## Data Preparation

The dataset was downloaded from Kaggle. We then:
1. Loaded and read the dataset.
2. Cleaned the data and replaced values.
3. Divided the dataset into training and testing sets (70-30 split) using the `train_test_split` function from `sklearn.model_selection`.

## Results & Discussion

1. **Data Preparation**: The dataset was successfully loaded and cleaned, with no null or empty values found. However, a bug was found where all values in the "gender" column became NaN. This was fixed by changing "male" and "female" to "Male" and "Female".

2. **Algorithm Implementation**: The code correctly calculated distances, sorted them, picked the top k distances, and predicted the personality based on the most common value.

3. **Classification Issue**: An issue was encountered when classifying all test instances using the `apply` function. The problem was caused by passing an extra parameter "knn". This was fixed by removing the "knn" parameter.

4. **Model Accuracy**: With k=5, the Euclidean method achieved an accuracy of 0.57, while both Manhattan and Minkowski methods achieved an accuracy of 0.61. These models did not undergo parameter optimization. The performance plot showed no signs of overfitting or underfitting.