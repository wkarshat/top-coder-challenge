"""
Ensemble Models

Combines multiple models to create more robust predictions through
ensemble methods like voting, bagging, and stacking.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any  # noqa: F401
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor,
    VotingRegressor, BaggingRegressor
)
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_score

from core.interfaces import IModel


class EnsembleModel(IModel):
    """Ensemble model combining multiple algorithms."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize ensemble model."""
        self.config = config
        self.ensemble_method = config.get('ensemble_method', 'voting')
        self.base_models = config.get('base_models', ['linear', 'tree', 'forest'])
        self.n_estimators = config.get('n_estimators', 100)
        self.random_state = config.get('random_state', 42)
        
        self.models = {}
        self.ensemble_model = None
        self.is_fitted = False
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'EnsembleModel':
        """Fit the ensemble model (sklearn-style interface)."""
        # Create a combined DataFrame for the train method
        data = X.copy()
        data[y.name or 'target'] = y
        
        # Call the train method
        self.train(data, y.name or 'target')
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions (sklearn-style interface)."""
        if not self.is_fitted:
            raise ValueError('Model not trained yet')
        
        # Get ensemble predictions
        return self.ensemble_model.predict(X)
    
    def get_metrics(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate model metrics."""
        return {
            'r2_score': float(r2_score(y_true, y_pred)),
            'mse': float(mean_squared_error(y_true, y_pred)),
            'mae': float(mean_absolute_error(y_true, y_pred))
        }
    
    def train(self, data: pd.DataFrame, target_column: str) -> Dict[str, Any]:
        """Train the ensemble model."""
        results = {}
        
        # Prepare features and target
        feature_cols = [col for col in data.columns if col != target_column]
        X = data[feature_cols]
        y = data[target_column]
        
        if len(X) < 5:
            return {'error': 'Need at least 5 samples for training'}
        
        # Initialize base models
        self._initialize_base_models()
        
        # Train individual models
        individual_results = {}
        for name, model in self.models.items():
            try:
                model.fit(X, y)
                
                # Evaluate individual model
                y_pred = model.predict(X)
                individual_results[name] = {
                    'r2_score': float(r2_score(y, y_pred)),
                    'mse': float(mean_squared_error(y, y_pred)),
                    'mae': float(mean_absolute_error(y, y_pred))
                }
                
                # Cross-validation score
                cv_scores = cross_val_score(model, X, y, cv=min(5, len(X)), 
                                          scoring='r2')
                individual_results[name]['cv_r2_mean'] = float(cv_scores.mean())
                individual_results[name]['cv_r2_std'] = float(cv_scores.std())
                
            except Exception as e:
                individual_results[name] = {'error': str(e)}
        
        # Create ensemble
        ensemble_result = self._create_ensemble(X, y)
        
        results.update({
            'individual_models': individual_results,
            'ensemble': ensemble_result,
            'feature_columns': feature_cols,
            'target_column': target_column,
            'training_samples': len(X)
        })
        
        self.is_fitted = True
        return results
    
    def predict_detailed(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Make detailed predictions with individual model results."""
        if not self.is_fitted:
            return {'error': 'Model not trained yet'}
        
        try:
            # Get predictions from ensemble
            ensemble_pred = self.ensemble_model.predict(data)
            
            # Get predictions from individual models
            individual_predictions = {}
            for name, model in self.models.items():
                try:
                    pred = model.predict(data)
                    individual_predictions[name] = pred.tolist()
                except Exception as e:
                    individual_predictions[name] = {'error': str(e)}
            
            return {
                'ensemble_predictions': ensemble_pred.tolist(),
                'individual_predictions': individual_predictions,
                'prediction_count': len(ensemble_pred)
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _initialize_base_models(self):
        """Initialize base models for the ensemble."""
        self.models = {}
        
        if 'linear' in self.base_models:
            self.models['linear_regression'] = LinearRegression()
            self.models['ridge'] = Ridge(alpha=1.0, random_state=self.random_state)
            self.models['lasso'] = Lasso(alpha=1.0, random_state=self.random_state)
        
        if 'tree' in self.base_models:
            self.models['decision_tree'] = DecisionTreeRegressor(
                random_state=self.random_state,
                max_depth=10
            )
        
        if 'forest' in self.base_models:
            self.models['random_forest'] = RandomForestRegressor(
                n_estimators=self.n_estimators,
                random_state=self.random_state,
                max_depth=10
            )
        
        if 'gradient_boosting' in self.base_models:
            self.models['gradient_boosting'] = GradientBoostingRegressor(
                n_estimators=self.n_estimators,
                random_state=self.random_state,
                max_depth=6
            )
    
    def _create_ensemble(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Create and train the ensemble model."""
        ensemble_results = {}
        
        try:
            if self.ensemble_method == 'voting':
                # Voting regressor
                estimators = [(name, model) for name, model in self.models.items()]
                self.ensemble_model = VotingRegressor(estimators=estimators)
                
            elif self.ensemble_method == 'bagging':
                # Use the best performing individual model as base
                best_model = self._find_best_model(X, y)
                self.ensemble_model = BaggingRegressor(
                    base_estimator=best_model,
                    n_estimators=self.n_estimators,
                    random_state=self.random_state
                )
                
            else:
                # Default to voting
                estimators = [(name, model) for name, model in self.models.items()]
                self.ensemble_model = VotingRegressor(estimators=estimators)
            
            # Train ensemble
            self.ensemble_model.fit(X, y)
            
            # Evaluate ensemble
            y_pred = self.ensemble_model.predict(X)
            
            ensemble_results.update({
                'method': self.ensemble_method,
                'r2_score': float(r2_score(y, y_pred)),
                'mse': float(mean_squared_error(y, y_pred)),
                'mae': float(mean_absolute_error(y, y_pred))
            })
            
            # Cross-validation for ensemble
            cv_scores = cross_val_score(self.ensemble_model, X, y, 
                                      cv=min(5, len(X)), scoring='r2')
            ensemble_results.update({
                'cv_r2_mean': float(cv_scores.mean()),
                'cv_r2_std': float(cv_scores.std())
            })
            
            # Feature importance (if available)
            if hasattr(self.ensemble_model, 'feature_importances_'):
                feature_names = X.columns.tolist()
                importances = self.ensemble_model.feature_importances_
                ensemble_results['feature_importances'] = dict(
                    zip(feature_names, importances.tolist())
                )
            
        except Exception as e:
            ensemble_results['error'] = str(e)
        
        return ensemble_results
    
    def _find_best_model(self, X: pd.DataFrame, y: pd.Series):
        """Find the best performing individual model."""
        best_score = -float('inf')
        best_model = None
        
        for model in self.models.values():
            try:
                cv_scores = cross_val_score(model, X, y, cv=min(3, len(X)), 
                                          scoring='r2')
                avg_score = cv_scores.mean()
                
                if avg_score > best_score:
                    best_score = avg_score
                    best_model = model
                    
            except Exception:
                continue
        
        return best_model if best_model is not None else LinearRegression()


class AdvancedEnsembleModel(IModel):
    """Advanced ensemble with stacking and blending."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize advanced ensemble model."""
        self.config = config
        self.meta_learner = config.get('meta_learner', 'linear')
        self.base_models = config.get('base_models', ['linear', 'tree', 'forest'])
        self.n_estimators = config.get('n_estimators', 50)
        self.random_state = config.get('random_state', 42)
        
        self.level_0_models = {}
        self.meta_model = None
        self.is_fitted = False
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'AdvancedEnsembleModel':
        """Fit the advanced ensemble model (sklearn-style interface)."""
        # Create a combined DataFrame for the train method
        data = X.copy()
        data[y.name or 'target'] = y
        
        # Call the train method
        self.train(data, y.name or 'target')
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions (sklearn-style interface)."""
        if not self.is_fitted:
            raise ValueError('Model not trained yet')
        
        # Generate level-0 predictions
        meta_features = []
        for model in self.level_0_models.values():
            pred = model.predict(X)
            meta_features.append(pred)
        
        # Stack meta-features and predict
        meta_X = np.column_stack(meta_features)
        return self.meta_model.predict(meta_X)
    
    def get_metrics(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate model metrics."""
        return {
            'r2_score': float(r2_score(y_true, y_pred)),
            'mse': float(mean_squared_error(y_true, y_pred)),
            'mae': float(mean_absolute_error(y_true, y_pred))
        }
    
    def train(self, data: pd.DataFrame, target_column: str) -> Dict[str, Any]:
        """Train the stacked ensemble model."""
        results = {}
        
        # Prepare features and target
        feature_cols = [col for col in data.columns if col != target_column]
        X = data[feature_cols]
        y = data[target_column]
        
        if len(X) < 10:
            return {'error': 'Need at least 10 samples for stacking'}
        
        # Initialize base models
        self._initialize_base_models()
        
        # Train level-0 models and generate meta-features
        meta_features, level_0_results = self._train_level_0_models(X, y)
        
        # Train meta-learner
        meta_result = self._train_meta_learner(meta_features, y)
        
        results.update({
            'level_0_models': level_0_results,
            'meta_learner': meta_result,
            'feature_columns': feature_cols,
            'target_column': target_column,
            'training_samples': len(X)
        })
        
        self.is_fitted = True
        return results
    
    def predict(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Make predictions using the stacked ensemble."""
        if not self.is_fitted:
            return {'error': 'Model not trained yet'}
        
        try:
            # Generate level-0 predictions
            meta_features = []
            level_0_predictions = {}
            
            for name, model in self.level_0_models.items():
                pred = model.predict(data)
                meta_features.append(pred)
                level_0_predictions[name] = pred.tolist()
            
            # Stack meta-features
            meta_X = np.column_stack(meta_features)
            
            # Meta-learner prediction
            final_predictions = self.meta_model.predict(meta_X)
            
            return {
                'final_predictions': final_predictions.tolist(),
                'level_0_predictions': level_0_predictions,
                'prediction_count': len(final_predictions)
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _initialize_base_models(self):
        """Initialize base models for stacking."""
        self.level_0_models = {}
        
        if 'linear' in self.base_models:
            self.level_0_models['ridge'] = Ridge(alpha=1.0, 
                                               random_state=self.random_state)
            self.level_0_models['lasso'] = Lasso(alpha=0.1, 
                                               random_state=self.random_state)
        
        if 'tree' in self.base_models:
            self.level_0_models['tree'] = DecisionTreeRegressor(
                max_depth=8, random_state=self.random_state
            )
        
        if 'forest' in self.base_models:
            self.level_0_models['forest'] = RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=8,
                random_state=self.random_state
            )
        
        # Initialize meta-learner
        if self.meta_learner == 'linear':
            self.meta_model = LinearRegression()
        elif self.meta_learner == 'ridge':
            self.meta_model = Ridge(alpha=1.0, random_state=self.random_state)
        else:
            self.meta_model = LinearRegression()
    
    def _train_level_0_models(self, X: pd.DataFrame, 
                            y: pd.Series) -> tuple:
        """Train level-0 models and generate meta-features."""
        meta_features = []
        level_0_results = {}
        
        for name, model in self.level_0_models.items():
            try:
                # Train model
                model.fit(X, y)
                
                # Generate out-of-fold predictions for meta-features
                predictions = model.predict(X)
                meta_features.append(predictions)
                
                # Evaluate model
                level_0_results[name] = {
                    'r2_score': float(r2_score(y, predictions)),
                    'mse': float(mean_squared_error(y, predictions)),
                    'mae': float(mean_absolute_error(y, predictions))
                }
                
            except Exception as e:
                level_0_results[name] = {'error': str(e)}
        
        # Stack meta-features
        if meta_features:
            meta_X = np.column_stack(meta_features)
        else:
            meta_X = np.zeros((len(X), 1))
        
        return meta_X, level_0_results
    
    def _train_meta_learner(self, meta_features: np.ndarray, 
                          y: pd.Series) -> Dict[str, Any]:
        """Train the meta-learner."""
        meta_results = {}
        
        try:
            # Train meta-model
            self.meta_model.fit(meta_features, y)
            
            # Evaluate meta-model
            meta_pred = self.meta_model.predict(meta_features)
            
            meta_results.update({
                'r2_score': float(r2_score(y, meta_pred)),
                'mse': float(mean_squared_error(y, meta_pred)),
                'mae': float(mean_absolute_error(y, meta_pred)),
                'meta_learner_type': self.meta_learner
            })
            
        except Exception as e:
            meta_results['error'] = str(e)
        
        return meta_results 