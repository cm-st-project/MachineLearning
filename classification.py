import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn import svm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Assuming your dataset is in a CSV file called "iris_dataset.csv"
# Replace the file name with your actual file name if needed
df = pd.read_csv("IRIS.csv")

# Select the features (sepal_length, sepal_width, petal_length, and petal_width) and species as X
X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]

# Select the target variable  and encode the 'species' column to numeric values.
le = LabelEncoder()
y = le.fit_transform(df['species'])

# Split the dataset into a training set and a test set (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a SVM model
model = svm.SVC()

# Fit the model to the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("Classification Report:\n", classification_report(y_test, y_pred))
# Compute the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Display the confusion matrix using a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


# Do prediction with new data
def prediction(data):
    predicted_value = model.predict([data])
    print('predicted numerical value: ', predicted_value)
    # convert the numerical value back to its original categorical label
    label = le.inverse_transform(predicted_value)

    return label[0]


# predict the Iris species with new data
x = [5.4, 3, 1.7, 0.4]
pred = prediction(x)
print(pred)
