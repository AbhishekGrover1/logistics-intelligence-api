import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib

def train_model():
    # 1. Load processed data (Path assumes you run the script from the root project folder)
    print("Loading dataset...")
    df = pd.read_csv("data/03_processed/logistics_training_data.csv")
    
    # 2. Isolate features and target
    X = df.drop(columns=['freight_value'])
    y = df['freight_value']

    # 3. Define column types for preprocessing
    num_cols = ['product_weight_g', 'product_length_cm', 'product_height_cm', 'product_width_cm']
    cat_cols = ['seller_state', 'customer_state']

    # 4. Build preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
        ])

    # 5. Create pipeline with Random Forest
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1))
    ])

    # 6. Split data and train
    print("Training model...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline.fit(X_train, y_train)

    # 7. Evaluate
    y_pred = pipeline.predict(X_test)
    print(f"Mean Absolute Error (Freight Cost R$): {mean_absolute_error(y_test, y_pred):.2f}")

    # 8. Export the serialized model
    joblib.dump(pipeline, 'models/freight_model.joblib')
    print("Model serialized successfully to models/freight_model.joblib")

if __name__ == "__main__":
    train_model()