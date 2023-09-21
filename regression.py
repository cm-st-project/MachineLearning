import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Read Dataset
df = pd.read_csv("IRIS.csv")
print(df.head())

# Encode the 'species' column to numeric values
le = LabelEncoder()
species_encoded = le.fit_transform(df['species'])

# Add the encoded species values as a new column 'species_encoded'
df['species_encoded'] = species_encoded

# Select the features (sepal_length, sepal_width, petal_length, and species_encoded) as X
X = df[['sepal_length', 'sepal_width', 'petal_length', 'species_encoded']]

# Select the target variable (petal_width) as y
y = df['petal_width']

# Split the dataset into a training set and a test set (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Fit the model to the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Calculate the mean squared error
mse = mean_squared_error(y_test, y_pred)

# Calculate the coefficient of determination (R-squared)
r2 = r2_score(y_test, y_pred)

# Print the results
print("Mean Squared Error:", mse)
print("R-squared:", r2)

for test, pred in zip(y_test, y_pred):
    print(test, pred)


# Do prediction with new data
def prediction(data):
    species = data[-1]
    encoded_species = le.transform([species])
    print('encoded_species:', encoded_species)
    data[-1] = encoded_species[0]
    predicted_value = model.predict([data])
    return predicted_value[0]


# predict the width of Iris petal
x = [5.4, 3.2, 1.5, 'Iris-setosa']
pred = prediction(x)
print('prediction:', pred)
