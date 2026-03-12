"""
Regression modeling module for urban transport analysis.
Builds predictive models for traffic congestion while actively 
diagnosing and mitigating multicollinearity.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')


def calculate_vif(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    """
    Calculate Variance Inflation Factor (VIF) to detect multicollinearity.
    VIF values > 5 indicate problematic collinearity.
    """
    X = df[features].dropna()
    
    # Add constant for the statsmodels VIF calculation
    X_with_const = sm.add_constant(X)
    
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X_with_const.columns
    vif_data["VIF"] = [
        variance_inflation_factor(X_with_const.values, i) 
        for i in range(X_with_const.shape[1])
    ]
    
    # Exclude the constant row from the final output
    return vif_data[vif_data["Feature"] != "const"].round(2)


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
        Defaults to ratio-based features to mitigate multicollinearity.
        """
        if features is None:
            # CRITICAL FIX: Replaced absolute values (Vehicle_Count, Population) 
            # with per-capita ratios to resolve severe multicollinearity.
            potential_features = [
                'Vehicles_Per_Capita',
                'Transport_Capacity_Per_Capita',
                'Vehicle_Transport_Ratio'
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
            if 'Regression' in name:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                y_train_pred = model.predict(X_train_scaled)
                cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=min(5, len(X_train)), scoring='r2')
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_train_pred = model.predict(X_train)
                cv_scores = cross_val_score(model, X_train, y_train, cv=min(5, len(X_train)), scoring='r2')
            
            # Evaluate
            train_r2 = r2_score(y_train, y_train_pred)
            test_r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            
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
        Extract feature importance from the trained models.
        """
        importances = []
        
        for name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                imp = pd.DataFrame({
                    'Feature': self.feature_names,
                    'Importance': model.feature_importances_,
                    'Model': name
                })
                importances.append(imp)
            elif hasattr(model, 'coef_'):
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
        if self.best_model is None:
            raise ValueError("No model trained yet. Call train_models first.")
        
        if 'Regression' in self.best_model_name:
            X_scaled = self.scaler.transform(X)
            return self.best_model.predict(X_scaled)
        return self.best_model.predict(X)
    
    def get_linear_equation(self) -> str:
        if 'Linear Regression' not in self.models:
            return "Linear regression model not trained."
        
        model = self.models['Linear Regression']
        
        terms = [f"{model.intercept_:.2f}"]
        for feature, coef in zip(self.feature_names, model.coef_):
            sign = "+" if coef >= 0 else "-"
            terms.append(f"{sign} {abs(coef):.6f} × {feature}")
        
        return "Traffic_Index = " + " ".join(terms)


def build_congestion_model(
    df: pd.DataFrame,
    target: str = 'Traffic_Index',
    features: Optional[List[str]] = None,
    verbose: bool = True
) -> Tuple[CongestionPredictor, Dict, pd.DataFrame]:
    """
    Main orchestration function to build, evaluate, and diagnose models.
    """
    predictor = CongestionPredictor()
    
    X, y = predictor.prepare_features(df, target, features)
    
    if verbose:
        print("\n=== Regression Modeling & Diagnostics ===")
        print(f"Engineered Features: {predictor.feature_names}")
        print(f"Samples: {len(X)}")
        
        # Run VIF Diagnostic
        print("\n--- Multicollinearity Diagnostic (VIF) ---")
        vif_df = calculate_vif(df, predictor.feature_names)
        print(vif_df.to_string(index=False))
        if any(vif_df['VIF'] > 5):
            print("WARNING: High multicollinearity detected (VIF > 5). Consider further feature engineering.")
        else:
            print("STATUS: VIF values are within acceptable limits. Features are well-isolated.")
    
    results = predictor.train_models(X, y)
    
    if verbose:
        print(f"\n--- Model Performance ---")
        print(f"{'Model':<25} {'Train R²':<12} {'Test R²':<12} {'RMSE':<12} {'CV Mean':<12}")
        print("-" * 73)
        for name, metrics in results.items():
            print(f"{name:<25} {metrics['train_r2']:<12} {metrics['test_r2']:<12} "
                  f"{metrics['rmse']:<12} {metrics['cv_mean']:<12}")
        
        print(f"\nBest Model: {predictor.best_model_name}")
        print(f"\nLinear Equation:\n{predictor.get_linear_equation()}")
    
    feature_importance = predictor.get_feature_importance()
    
    return predictor, results, feature_importance


def interpret_coefficients(predictor: CongestionPredictor) -> List[str]:
    """
    Interpret the standardized coefficients of the Linear Regression model.
    """
    if 'Linear Regression' not in predictor.models:
        return ["Linear regression model not available."]
    
    model = predictor.models['Linear Regression']
    interpretations = []
    
    for feature, coef in zip(predictor.feature_names, model.coef_):
        impact = "increases" if coef > 0 else "decreases"
        interpretations.append(
            f"A one standard deviation increase in {feature} {impact} the "
            f"Traffic Index by {abs(coef):.2f} points."
        )
    
    return interpretations
