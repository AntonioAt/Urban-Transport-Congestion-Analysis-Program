"""
Regression modeling module for urban transport analysis.
Builds and evaluates predictive models for traffic congestion.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class CongestionPredictor:
    """
    A class to build and evaluate regression models for predicting traffic congestion.
    """
    
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_names = []
        self.best_model = None
        self.best_model_name = None
        
    def prepare_features(
        self,
        df: pd.DataFrame,
        target: str = 'Traffic_Index',
        features: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare feature matrix and target vector.
        """
        if features is None:
            # Default features for the model
            potential_features = [
                'Vehicle_Count', 'Population', 'Public_Transport_Capacity',
                'Public_Transport_Fleet', 'Vehicles_Per_Capita',
                'Transport_Capacity_Per_Capita', 'Vehicle_Transport_Ratio'
            ]
            features = [f for f in potential_features if f in df.columns]
        
        self.feature_names = features
        
        # Remove rows with missing values in features or target
        clean_df = df[features + [target]].dropna()
        
        X = clean_df[features].values
        y = clean_df[target].values
        
        return X, y
    
    def train_models(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Dict[str, Dict]:
        """
        Train multiple regression models and evaluate their performance.
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Define models
        model_configs = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=0.1),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=random_state),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=random_state)
        }
        
        results = {}
        best_r2 = -np.inf
        
        for name, model in model_configs.items():
            # Train
            if 'Regression' in name:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                y_train_pred = model.predict(X_train_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_train_pred = model.predict(X_train)
            
            # Evaluate
            train_r2 = r2_score(y_train, y_train_pred)
            test_r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            
            # Cross-validation (on scaled data for linear models)
            if 'Regression' in name:
                cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=min(5, len(X_train)), scoring='r2')
            else:
                cv_scores = cross_val_score(model, X_train, y_train, cv=min(5, len(X_train)), scoring='r2')
            
            results[name] = {
                'model': model,
                'train_r2': round(train_r2, 4),
                'test_r2': round(test_r2, 4),
                'rmse': round(rmse, 4),
                'mae': round(mae, 4),
                'cv_mean': round(cv_scores.mean(), 4),
                'cv_std': round(cv_scores.std(), 4)
            }
            
            self.models[name] = model
            
            if test_r2 > best_r2:
                best_r2 = test_r2
                self.best_model = model
                self.best_model_name = name
        
        return results
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance from the best model.
        """
        importances = []
        
        for name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                # Tree-based models
                imp = pd.DataFrame({
                    'Feature': self.feature_names,
                    'Importance': model.feature_importances_,
                    'Model': name
                })
                importances.append(imp)
            elif hasattr(model, 'coef_'):
                # Linear models
                imp = pd.DataFrame({
                    'Feature': self.feature_names,
                    'Importance': np.abs(model.coef_),
                    'Model': name
                })
                importances.append(imp)
        
        if importances:
            return pd.concat(importances, ignore_index=True)
        return pd.DataFrame()
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the best model.
        """
        if self.best_model is None:
            raise ValueError("No model trained yet. Call train_models first.")
        
        if 'Regression' in self.best_model_name:
            X_scaled = self.scaler.transform(X)
            return self.best_model.predict(X_scaled)
        return self.best_model.predict(X)
    
    def get_linear_equation(self) -> str:
        """
        Get the equation for the linear regression model.
        """
        if 'Linear Regression' not in self.models:
            return "Linear regression model not trained."
        
        model = self.models['Linear Regression']
        
        terms = [f"{model.intercept_:.2f}"]
        for feature, coef in zip(self.feature_names, model.coef_):
            sign = "+" if coef >= 0 else "-"
            terms.append(f"{sign} {abs(coef):.6f} × {feature}")
        
        equation = "Traffic_Index = " + " ".join(terms)
        return equation


def build_congestion_model(
    df: pd.DataFrame,
    target: str = 'Traffic_Index',
    features: Optional[List[str]] = None,
    verbose: bool = True
) -> Tuple[CongestionPredictor, Dict, pd.DataFrame]:
    """
    Main function to build and evaluate congestion prediction models.
    
    Returns:
        Tuple of (predictor, results, feature_importance)
    """
    predictor = CongestionPredictor()
    
    # Prepare data
    X, y = predictor.prepare_features(df, target, features)
    
    if verbose:
        print(f"\n=== Regression Modeling ===")
        print(f"Features: {predictor.feature_names}")
        print(f"Samples: {len(X)}")
    
    # Train models
    results = predictor.train_models(X, y)
    
    if verbose:
        print(f"\n--- Model Performance ---")
        print(f"{'Model':<25} {'Train R²':<12} {'Test R²':<12} {'RMSE':<12} {'CV Mean':<12}")
        print("-" * 73)
        for name, metrics in results.items():
            print(f"{name:<25} {metrics['train_r2']:<12} {metrics['test_r2']:<12} "
                  f"{metrics['rmse']:<12} {metrics['cv_mean']:<12}")
        
        print(f"\nBest Model: {predictor.best_model_name}")
        print(f"\nLinear Regression Equation:")
        print(predictor.get_linear_equation())
    
    # Get feature importance
    feature_importance = predictor.get_feature_importance()
    
    return predictor, results, feature_importance


def interpret_coefficients(predictor: CongestionPredictor) -> List[str]:
    """
    Generate interpretations of the linear regression coefficients.
    """
    if 'Linear Regression' not in predictor.models:
        return ["Linear regression model not available."]
    
    model = predictor.models['Linear Regression']
    interpretations = []
    
    for feature, coef in zip(predictor.feature_names, model.coef_):
        if 'Vehicle' in feature and coef > 0:
            interpretations.append(
                f"Increasing {feature} is associated with higher congestion "
                f"(+{coef:.6f} Traffic Index points per unit increase)."
            )
        elif 'Transport' in feature and coef < 0:
            interpretations.append(
                f"Increasing {feature} is associated with lower congestion "
                f"({coef:.6f} Traffic Index points per unit increase)."
            )
        elif 'Capacity' in feature:
            direction = "reduces" if coef < 0 else "increases"
            interpretations.append(
                f"Each unit increase in {feature} {direction} the Traffic Index by {abs(coef):.6f} points."
            )
    
    return interpretations
