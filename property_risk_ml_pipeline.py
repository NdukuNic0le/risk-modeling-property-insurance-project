import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import KFold
import xgboost as xgb
from datetime import datetime
from sklearn.impute import SimpleImputer

class PropertyRiskModel:
    def __init__(self, property_data, weather_data):
        self.property_data = property_data
        self.weather_data = weather_data
        self.combined_data = None   
        self.X = None
        self.y = None
        self.models = {}
        self.best_model = None
        
    def engineer_features(self):
        """Feature engineering process with proper NaN handling"""
        # Aggregate weather events by property
        weather_features = self.weather_data.groupby('property_id').agg({
            'damage_value': ['count', 'sum', 'mean', 'max'],
            'event_type': lambda x: x.value_counts().to_dict(),
            'severity': lambda x: x.value_counts().to_dict(),
            'date': ['min', 'max']
        }).reset_index()
        
        # Flatten the multi-level columns
        weather_features.columns = ['property_id', 
                                'event_count', 'total_damage', 'avg_damage', 'max_damage',
                                'event_type_dist', 'severity_dist',
                                'first_event_date', 'last_event_date']
        
        # Create time-based features
        weather_features['days_between_events'] = (
            weather_features['last_event_date'] - weather_features['first_event_date']
        ).dt.days
        
        # Extract event type counts
        event_types = pd.json_normalize(weather_features['event_type_dist']).fillna(0)
        weather_features = pd.concat([weather_features, event_types], axis=1)
        
        # Extract severity counts
        severity_types = pd.json_normalize(weather_features['severity_dist']).fillna(0)
        weather_features = pd.concat([weather_features, severity_types], axis=1)
        
        # Merge with property data
        self.combined_data = self.property_data.merge(
            weather_features, on='property_id', how='left'
        )
        
        # Fill NaN values for properties with no events
        self.combined_data['event_count'] = self.combined_data['event_count'].fillna(0)
        self.combined_data['total_damage'] = self.combined_data['total_damage'].fillna(0)
        self.combined_data['avg_damage'] = self.combined_data['avg_damage'].fillna(0)
        self.combined_data['max_damage'] = self.combined_data['max_damage'].fillna(0)
        self.combined_data['days_between_events'] = self.combined_data['days_between_events'].fillna(0)
        
        # Calculate risk-related features
        self.combined_data['age'] = datetime.now().year - self.combined_data['year_built']
        self.combined_data['damage_per_event'] = np.where(
            self.combined_data['event_count'] > 0,
            self.combined_data['total_damage'] / self.combined_data['event_count'],
            0
        )
        self.combined_data['damage_ratio'] = np.where(
            self.combined_data['property_value'] > 0,
            self.combined_data['total_damage'] / self.combined_data['property_value'],
            0
        )
        
        # Create final feature matrix
        numeric_features = [
            'property_value', 'year_built', 'elevation', 'distance_to_water',
            'event_count', 'age', 'damage_per_event', 'damage_ratio',
            'days_between_events'
        ]
        
        categorical_features = ['city', 'zone_type', 'property_type', 'flood_zone']
        
        # Prepare feature matrix and ensure no NaN values
        self.X = self.combined_data[numeric_features + categorical_features]
        self.y = self.combined_data['total_damage']
        
        # Fill any remaining NaN values
        self.X = self.X.fillna(method='ffill').fillna(method='bfill')
        self.y = self.y.fillna(0)  # Assuming no damage for missing values
        
        return self.X, self.y
    
    
    def train_models(self, test_size=0.2, random_state=42):
        """Train and evaluate multiple models with proper preprocessing"""
        # Verify no NaN values before splitting
        assert not self.X.isna().any().any(), "Features contain NaN values"
        assert not self.y.isna().any(), "Target variable contains NaN values"
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state
        )
        
        # Create preprocessing pipeline
        numeric_features = self.X.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = self.X.select_dtypes(include=['object']).columns
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
            ])
        
        # Define models with hyperparameter grids
        models = {
            'random_forest': {
                'model': RandomForestRegressor(random_state=random_state),
                'params': {
                    'n_estimators': [100, 200],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2]
                }
            },
            'gradient_boosting': {
                'model': GradientBoostingRegressor(random_state=random_state),
                'params': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.01, 0.1],
                    'max_depth': [3, 5],
                    'min_samples_split': [2, 5]
                }
            },
            'xgboost': {
                'model': xgb.XGBRegressor(random_state=random_state),
                'params': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.01, 0.1],
                    'max_depth': [3, 5],
                    'min_child_weight': [1, 3]
                }
            }
        }
        
        # Train and evaluate each model
        results = {}
        for name, model_info in models.items():
            print(f"\nTraining {name}...")
            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('regressor', model_info['model'])
            ])
            
            grid_search = GridSearchCV(
                pipeline,
                {f'regressor__{k}': v for k, v in model_info['params'].items()},
                cv=5,
                scoring='neg_mean_squared_error',
                n_jobs=-1,
                error_score='raise'  # Raise errors for debugging
            )
            
            grid_search.fit(X_train, y_train)
            
            results[name] = {
                'best_params': grid_search.best_params_,
                'best_score': -grid_search.best_score_,
                'test_score': mean_squared_error(y_test, grid_search.predict(X_test)),
                'r2_score': r2_score(y_test, grid_search.predict(X_test)),
                'model': grid_search.best_estimator_
            }
            
            print(f"Completed training {name}")
            print(f"Best parameters: {results[name]['best_params']}")
            print(f"Test MSE: {results[name]['test_score']:.2f}")
            print(f"R² Score: {results[name]['r2_score']:.2f}")
            
            self.models[name] = results[name]['model']
        
        best_model_name = min(results, key=lambda x: results[x]['test_score'])
        self.best_model = self.models[best_model_name]
        
        return results
    
    def cross_validate_best_model(self, n_splits=5):
        """Perform detailed cross-validation on the best model"""
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        metrics = {
            'mse': [],
            'mae': [],
            'r2': []
        }
        
        for train_idx, val_idx in kf.split(self.X):
            X_train, X_val = self.X.iloc[train_idx], self.X.iloc[val_idx]
            y_train, y_val = self.y.iloc[train_idx], self.y.iloc[val_idx]
            
            # Fit and predict
            self.best_model.fit(X_train, y_train)
            y_pred = self.best_model.predict(X_val)
            
            # Calculate metrics
            metrics['mse'].append(mean_squared_error(y_val, y_pred))
            metrics['mae'].append(mean_absolute_error(y_val, y_pred))
            metrics['r2'].append(r2_score(y_val, y_pred))
        
        # Calculate mean and std for each metric
        cv_results = {
            metric: {
                'mean': np.mean(values),
                'std': np.std(values)
            }
            for metric, values in metrics.items()
        }
        
        return cv_results
    
    # If you uncomment this, comment the cross validation method above or rename the objects in this one
    # def cross_val_score_evaluation(self, cv=5, scoring=['neg_mean_squared_error', 'neg_mean_absolute_error', 'r2']):
    #     """Perform cross-validation using cross_val_score for multiple metrics"""
    #     if self.best_model is None:
    #         raise ValueError("Model has not been trained yet!")
        
    #     cv_results = {}
    #     for score in scoring:
    #         scores = cross_val_score(
    #             self.best_model,
    #             self.X,
    #             self.y,
    #             cv=cv,
    #             scoring=score,
    #             n_jobs=-1
    #         )
            
    #         # Convert negative scores back to positive for error metrics
    #         if score.startswith('neg_'):
    #             scores = -scores
    #             score = score.replace('neg_', '')
                
    #         cv_results[score] = {
    #             'mean': scores.mean(),
    #             'std': scores.std(),
    #             'all_scores': scores
    #         }
        
    #     return cv_results
    
    def predict_risk(self, new_data):
        """Make predictions on new data"""
        if self.best_model is None:
            raise ValueError("Model has not been trained yet!")
        
        return self.best_model.predict(new_data)

# Example usage:
if __name__ == "__main__":
    from property_risk_data_generator import KenyaPropertyDataGenerator
    
    # Generate sample data
    generator = KenyaPropertyDataGenerator()
    property_data = generator.generate_property_data(1000)
    weather_data = generator.generate_weather_events(property_data)
    
    # Create and train model
    risk_model = PropertyRiskModel(property_data, weather_data)
    X, y = risk_model.engineer_features()
    
    # Train models and get results
    results = risk_model.train_models()
    
    # Perform cross-validation
    cv_results = risk_model.cross_validate_best_model()
    
    # Print results
    print("\nModel Training Results:")
    for model_name, model_results in results.items():
        print(f"\n{model_name.upper()}:")
        print(f"Best Parameters: {model_results['best_params']}")
        print(f"Best CV Score (MSE): {model_results['best_score']:.2f}")
        print(f"Test Score (MSE): {model_results['test_score']:.2f}")
        print(f"R² Score: {model_results['r2_score']:.2f}")
    
    print("\nCross-Validation Results for Best Model:")
    for metric, values in cv_results.items():
        print(f"{metric.upper()}: {values['mean']:.2f} (±{values['std']:.2f})")