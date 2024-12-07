# affiliate-marketing-
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Example data: User behavior data (User IDs, Product IDs, Rating/Clicks)
data = {
    'user_id': [1, 1, 1, 2, 2, 3, 3, 3, 4, 4],
    'product_id': [101, 102, 103, 101, 104, 102, 104, 105, 103, 106],
    'clicks': [1, 1, 0, 0, 1, 1, 0, 1, 1, 0]  # 1 means clicked, 0 means not clicked
}

# Convert data into a pandas DataFrame
df = pd.DataFrame(data)

# Convert categorical data (user_id, product_id) into numerical representations
df['user_id'] = df['user_id'].astype('category').cat.codes
df['product_id'] = df['product_id'].astype('category').cat.codes

# Define features (user_id, product_id) and target (clicks)
X = df[['user_id', 'product_id']].values
y = df['clicks'].values

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build a simple neural network model
model = Sequential()
model.add(Dense(128, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  # Output layer for binary classification (clicked or not clicked)

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=1, verbose=1)

# Evaluate the model
accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {accuracy[1]*100:.2f}%")

# Example prediction (user 1, product 101)
user_input = np.array([[0, 0]])  # user_id=0 (mapped) and product_id=0 (mapped)
user_input = scaler.transform(user_input)  # Standardize the input
pred = model.predict(user_input)

# Output the prediction (probability of click)
print(f"Prediction (click probability): {pred[0][0]:.4f}")
