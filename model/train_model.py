import numpy as np
import pandas as pd
import joblib
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

class HousePriceTrainer:
    def __init__(self, data_path, target_col='median_house_value'):
        self.data_path = data_path
        self.target_col = target_col
        self.model = None
        self.preprocessor = None
        self.feature_names = None
        
    def load_and_prepare_data(self):
        """Load and prepare the dataset"""
        df = pd.read_csv(self.data_path)
        
        # Separate features and target
        X = df.drop(columns=[self.target_col])
        y = df[self.target_col]
        
        return X, y, df
    
    def create_preprocessor(self, X):
        """Create preprocessing pipeline"""
        numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()
        
        # Numerical pipeline
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        # Categorical pipeline
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        # Combine preprocessors
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features)
            ])
        
        # Store feature names for later use
        self.feature_names = {
            'numerical': numerical_features,
            'categorical': categorical_features
        }
        
        return self.preprocessor
    
    def train_best_model(self):
        """Train the best model with hyperparameter tuning"""
        X, y, df = self.load_and_prepare_data()
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Create preprocessor
        preprocessor = self.create_preprocessor(X)
        
        # Create pipeline with best parameters from your notebook
        best_params = {
            'l2_regularization': 0.1,
            'learning_rate': 0.1,
            'max_depth': None,
            'max_leaf_nodes': 63,
            'min_samples_leaf': 20
        }
        
        model = HistGradientBoostingRegressor(
            random_state=42,
            **best_params
        )
        
        # Create full pipeline
        self.model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', model)
        ])
        
        # Train the model
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)
        
        metrics = {
            'train': {
                'rmse': root_mean_squared_error(y_train, y_pred_train),
                'mae': mean_absolute_error(y_train, y_pred_train),
                'r2': r2_score(y_train, y_pred_train)
            },
            'test': {
                'rmse': root_mean_squared_error(y_test, y_pred_test),
                'mae': mean_absolute_error(y_test, y_pred_test),
                'r2': r2_score(y_test, y_pred_test)
            }
        }
        
        return self.model, metrics, X_train, X_test, y_train, y_test, df
    
    def save_model(self, path='model/best_model.pkl'):
        """Save the trained model"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({
            'model': self.model,
            'feature_names': self.feature_names,
            'preprocessor': self.preprocessor
        }, path)
        print(f"Model saved to {path}")

if __name__ == "__main__":
    trainer = HousePriceTrainer('data/housing.csv')
    model, metrics, _, _, _, _, _ = trainer.train_best_model()
    trainer.save_model()
    print("Training completed!")
    print(f"Test RMSE: {metrics['test']['rmse']:.2f}")
    print(f"Test R²: {metrics['test']['r2']:.3f}")