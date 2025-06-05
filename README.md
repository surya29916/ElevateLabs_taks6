# ElevateLabs_Task6
# Iris Flower Classification using K-Nearest Neighbors (KNN)

## ✅ Steps Performed

### 1. Load and Preprocess the Dataset
- Loaded the dataset using `pandas`.
- Dropped the unnecessary `Id` column.
- Encoded the `Species` target column into numerical values using **Label Encoding**.
- Normalized all the feature values using `StandardScaler`.

### 2. Train-Test Split
- Split the dataset into training and testing sets using `train_test_split()`.
- Used an 80-20 split for training and testing respectively.

### 3. Train KNN Classifier with Different Values of K
- Used `KNeighborsClassifier` from `sklearn` to train models with different values of **K** (1 to 10).
- Recorded and printed the accuracy of the model for each value of K.
- Plotted **K vs Accuracy** to visualize model performance.

### 4. Evaluate Best K Model
- Chose the best value of K based on accuracy.
- Evaluated the final model using:
  - **Accuracy Score**
  - **Confusion Matrix**

### 5. Visualize Decision Boundaries
- Visualized the decision boundaries using the first two features only (Sepal Length & Sepal Width).
- Plotted the classifier’s regions and data points using `matplotlib` and `contourf`.

---

## 📈 Observations

- The accuracy varied with the value of K; selecting an appropriate K is important for model performance.
- Visualization of decision boundaries gives insight into how the classifier separates different classes based on feature values.
- KNN works well when features are scaled/normalized, as it relies on distance metrics.

---

## 🛠️ Tools & Libraries Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

---
