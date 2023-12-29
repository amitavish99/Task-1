# Task-1
MOVIE RATING PREDICTION WITH PYTHON
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
2. Load Data
python
Copy code
# Load your movie dataset
# Assuming you have a CSV file named 'movie_data.csv'
data = pd.read_csv('movie_data.csv')

# Display the first few rows of the dataset
print(data.head())
3. Data Preprocessing
python
Copy code
# Handle missing values if any
data = data.dropna()

# Extract relevant features (genre, director, actors) and target variable (rating)
features = data[['genre', 'director', 'actors']]
target = data['rating']
4. Feature Engineering
python
Copy code
# One-hot encode categorical features (genre, director, actors)
preprocessor = ColumnTransformer(
    transformers=[
        ('genre_director_actors', OneHotEncoder(), ['genre', 'director', 'actors'])
    ],
    remainder='passthrough'
)

# Define the model pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])
5. Train-Test Split
python
Copy code
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
6. Model Training
python
Copy code
# Train the model
model.fit(X_train, y_train)
7. Model Evaluation
python
Copy code
# Make predictions on the test set
predictions = model.predict(X_test)

# Evaluate the model performance
mse = mean_squared_error(y_test, predictions)
print(f'Mean Squared Error: {mse}')
This is a basic example using linear regression. Depending on the characteristics of your dataset, you might want to explore more advanced regression techniques and fine-tune your model. Additionally, feature scaling, hyperparameter tuning, and cross-validation can be used to improve the model's performance.
