import numpy as np
import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Concatenate, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import os
from typing import Tuple, Dict, Optional, List
import logging
from config import Config

# Import individual models
from .lstm_model import LSTMModel
from .cnn_model import CNNModel
from .rnn_model import RNNModel

class HybridModel:
    """Hybrid model combining LSTM, CNN, and RNN for ensemble predictions"""
    
    def __init__(self, config: Dict = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or Config.HYBRID_CONFIG
        
        # Individual models
        self.lstm_model = LSTMModel({
            'sequence_length': self.config['sequence_length'],
            'units': self.config['lstm_units'],
            'dropout': self.config['dropout'],
            'epochs': self.config['epochs'],
            'batch_size': self.config['batch_size'],
            'validation_split': self.config['validation_split']
        })
        
        self.cnn_model = CNNModel({
            'sequence_length': self.config['sequence_length'],
            'filters': self.config['cnn_filters'],
            'kernel_size': 3,
            'pool_size': 2,
            'dropout': self.config['dropout'],
            'epochs': self.config['epochs'],
            'batch_size': self.config['batch_size'],
            'validation_split': self.config['validation_split']
        })
        
        self.rnn_model = RNNModel({
            'sequence_length': self.config['sequence_length'],
            'units': self.config['rnn_units'],
            'dropout': self.config['dropout'],
            'epochs': self.config['epochs'],
            'batch_size': self.config['batch_size'],
            'validation_split': self.config['validation_split']
        })
        
        # Ensemble model
        self.ensemble_model = None
        self.ensemble_weights = self.config['ensemble_weights']
        self.is_trained = False
        self.training_history = {}
        
    def build_ensemble_model(self, input_shape: Tuple, output_shape: int) -> Model:
        """Build ensemble model that combines predictions from individual models"""
        try:
            # Input layer
            input_layer = Input(shape=input_shape)
            
            # Build individual models
            lstm_base = self.lstm_model.build_model(input_shape, output_shape)
            cnn_base = self.cnn_model.build_model(input_shape, output_shape)
            rnn_base = self.rnn_model.build_model(input_shape, output_shape)
            
            # Get predictions from each model
            lstm_pred = lstm_base(input_layer)
            cnn_pred = cnn_base(input_layer)
            rnn_pred = rnn_base(input_layer)
            
            # Concatenate predictions
            combined = Concatenate()([lstm_pred, cnn_pred, rnn_pred])
            
            # Meta-learner layers
            x = Dense(64, activation='relu')(combined)
            x = BatchNormalization()(x)
            x = Dropout(self.config['dropout'])(x)
            
            x = Dense(32, activation='relu')(x)
            x = BatchNormalization()(x)
            x = Dropout(self.config['dropout'])(x)
            
            # Final output
            output = Dense(output_shape, activation='linear')(x)
            
            # Create ensemble model
            ensemble_model = Model(inputs=input_layer, outputs=output)
            
            # Compile model
            ensemble_model.compile(
                optimizer=Adam(learning_rate=0.0005),  # Lower learning rate for ensemble
                loss='mse',
                metrics=['mae', 'mape']
            )
            
            self.ensemble_model = ensemble_model
            
            self.logger.info(f"Hybrid ensemble model built with input shape: {input_shape}, output shape: {output_shape}")
            self.logger.info(f"Total parameters: {ensemble_model.count_params()}")
            
            return ensemble_model
            
        except Exception as e:
            self.logger.error(f"Error building ensemble model: {str(e)}")
            return None
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: np.ndarray = None, y_val: np.ndarray = None, 
              method: str = 'ensemble') -> Dict:
        """Train the hybrid model using different strategies"""
        try:
            if method == 'ensemble':
                return self._train_ensemble(X_train, y_train, X_val, y_val)
            elif method == 'sequential':
                return self._train_sequential(X_train, y_train, X_val, y_val)
            elif method == 'weighted':
                return self._train_weighted(X_train, y_train, X_val, y_val)
            else:
                raise ValueError(f"Unknown training method: {method}")
                
        except Exception as e:
            self.logger.error(f"Error training hybrid model: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _train_ensemble(self, X_train: np.ndarray, y_train: np.ndarray, 
                       X_val: np.ndarray = None, y_val: np.ndarray = None) -> Dict:
        """Train using ensemble approach with meta-learner"""
        try:
            # Build ensemble model if not exists
            if self.ensemble_model is None:
                input_shape = (X_train.shape[1], X_train.shape[2])
                output_shape = y_train.shape[1] if len(y_train.shape) > 1 else 1
                self.build_ensemble_model(input_shape, output_shape)
            
            # Prepare validation data
            validation_data = None
            if X_val is not None and y_val is not None:
                validation_data = (X_val, y_val)
            
            # Callbacks
            callbacks = [
                EarlyStopping(
                    monitor='val_loss' if validation_data else 'loss',
                    patience=20,
                    restore_best_weights=True,
                    verbose=1
                ),
                ReduceLROnPlateau(
                    monitor='val_loss' if validation_data else 'loss',
                    factor=0.5,
                    patience=10,
                    min_lr=1e-8,
                    verbose=1
                )
            ]
            
            # Train ensemble model
            history = self.ensemble_model.fit(
                X_train, y_train,
                epochs=self.config['epochs'],
                batch_size=self.config['batch_size'],
                validation_data=validation_data,
                validation_split=self.config['validation_split'] if validation_data is None else None,
                callbacks=callbacks,
                verbose=1,
                shuffle=True
            )
            
            self.is_trained = True
            self.training_history['ensemble'] = history.history
            
            # Calculate training metrics
            train_predictions = self.ensemble_model.predict(X_train)
            train_mse = mean_squared_error(y_train, train_predictions)
            train_mae = mean_absolute_error(y_train, train_predictions)
            
            metrics = {
                'train_mse': float(train_mse),
                'train_mae': float(train_mae),
                'train_rmse': float(np.sqrt(train_mse)),
                'epochs_trained': len(history.history['loss']),
                'method': 'ensemble'
            }
            
            # Validation metrics if available
            if validation_data is not None:
                val_predictions = self.ensemble_model.predict(X_val)
                val_mse = mean_squared_error(y_val, val_predictions)
                val_mae = mean_absolute_error(y_val, val_predictions)
                
                metrics.update({
                    'val_mse': float(val_mse),
                    'val_mae': float(val_mae),
                    'val_rmse': float(np.sqrt(val_mse))
                })
            
            self.logger.info(f"Hybrid ensemble model training completed. Final metrics: {metrics}")
            
            return {
                'success': True,
                'metrics': metrics,
                'history': history.history
            }
            
        except Exception as e:
            self.logger.error(f"Error training ensemble model: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _train_sequential(self, X_train: np.ndarray, y_train: np.ndarray, 
                         X_val: np.ndarray = None, y_val: np.ndarray = None) -> Dict:
        """Train individual models sequentially"""
        try:
            all_metrics = {}
            all_histories = {}
            
            # Train LSTM
            self.logger.info("Training LSTM model...")
            lstm_result = self.lstm_model.train(X_train, y_train, X_val, y_val)
            if lstm_result['success']:
                all_metrics['lstm'] = lstm_result['metrics']
                all_histories['lstm'] = lstm_result['history']
            
            # Train CNN
            self.logger.info("Training CNN model...")
            cnn_result = self.cnn_model.train(X_train, y_train, X_val, y_val)
            if cnn_result['success']:
                all_metrics['cnn'] = cnn_result['metrics']
                all_histories['cnn'] = cnn_result['history']
            
            # Train RNN
            self.logger.info("Training RNN model...")
            rnn_result = self.rnn_model.train(X_train, y_train, X_val, y_val)
            if rnn_result['success']:
                all_metrics['rnn'] = rnn_result['metrics']
                all_histories['rnn'] = rnn_result['history']
            
            self.is_trained = True
            self.training_history = all_histories
            
            # Calculate ensemble metrics based on validation performance
            if X_val is not None and y_val is not None:
                ensemble_metrics = self._calculate_ensemble_metrics(X_val, y_val)
                all_metrics['ensemble'] = ensemble_metrics
            
            self.logger.info("Sequential training completed for all models")
            
            return {
                'success': True,
                'metrics': all_metrics,
                'history': all_histories,
                'method': 'sequential'
            }
            
        except Exception as e:
            self.logger.error(f"Error in sequential training: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _train_weighted(self, X_train: np.ndarray, y_train: np.ndarray, 
                       X_val: np.ndarray = None, y_val: np.ndarray = None) -> Dict:
        """Train with weighted ensemble approach"""
        try:
            # First train individual models
            sequential_result = self._train_sequential(X_train, y_train, X_val, y_val)
            
            if not sequential_result['success']:
                return sequential_result
            
            # Optimize ensemble weights based on validation performance
            if X_val is not None and y_val is not None:
                optimal_weights = self._optimize_ensemble_weights(X_val, y_val)
                self.ensemble_weights = optimal_weights
                
                self.logger.info(f"Optimized ensemble weights: {optimal_weights}")
            
            return sequential_result
            
        except Exception as e:
            self.logger.error(f"Error in weighted training: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _optimize_ensemble_weights(self, X_val: np.ndarray, y_val: np.ndarray) -> List[float]:
        """Optimize ensemble weights using validation data"""
        try:
            from scipy.optimize import minimize
            
            # Get predictions from individual models
            lstm_pred = self.lstm_model.predict(X_val)
            cnn_pred = self.cnn_model.predict(X_val)
            rnn_pred = self.rnn_model.predict(X_val)
            
            def objective(weights):
                # Ensure weights sum to 1
                weights = weights / np.sum(weights)
                
                # Calculate weighted ensemble prediction
                ensemble_pred = (weights[0] * lstm_pred + 
                               weights[1] * cnn_pred + 
                               weights[2] * rnn_pred)
                
                # Return MSE as objective to minimize
                return mean_squared_error(y_val, ensemble_pred)
            
            # Initial weights
            initial_weights = np.array(self.ensemble_weights)
            
            # Constraints: weights must be positive and sum to 1
            constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
            bounds = [(0, 1) for _ in range(3)]
            
            # Optimize
            result = minimize(objective, initial_weights, 
                            method='SLSQP', bounds=bounds, constraints=constraints)
            
            if result.success:
                return result.x.tolist()
            else:
                self.logger.warning("Weight optimization failed, using default weights")
                return self.ensemble_weights
                
        except Exception as e:
            self.logger.error(f"Error optimizing ensemble weights: {str(e)}")
            return self.ensemble_weights
    
    def predict(self, X: np.ndarray, method: str = 'weighted') -> np.ndarray:
        """Make predictions using the hybrid model"""
        try:
            if not self.is_trained:
                raise ValueError("Model must be trained before making predictions")
            
            if method == 'ensemble' and self.ensemble_model is not None:
                return self.ensemble_model.predict(X)
            elif method == 'weighted':
                return self._predict_weighted(X)
            elif method == 'voting':
                return self._predict_voting(X)
            else:
                return self._predict_weighted(X)  # Default to weighted
                
        except Exception as e:
            self.logger.error(f"Error making predictions: {str(e)}")
            return None
    
    def _predict_weighted(self, X: np.ndarray) -> np.ndarray:
        """Make weighted ensemble predictions"""
        try:
            # Get predictions from individual models
            lstm_pred = self.lstm_model.predict(X)
            cnn_pred = self.cnn_model.predict(X)
            rnn_pred = self.rnn_model.predict(X)
            
            # Weighted combination
            weights = np.array(self.ensemble_weights)
            ensemble_pred = (weights[0] * lstm_pred + 
                           weights[1] * cnn_pred + 
                           weights[2] * rnn_pred)
            
            self.logger.info(f"Generated {len(ensemble_pred)} weighted ensemble predictions")
            
            return ensemble_pred
            
        except Exception as e:
            self.logger.error(f"Error in weighted prediction: {str(e)}")
            return None
    
    def _predict_voting(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using majority voting"""
        try:
            # Get predictions from individual models
            lstm_pred = self.lstm_model.predict(X)
            cnn_pred = self.cnn_model.predict(X)
            rnn_pred = self.rnn_model.predict(X)
            
            # Simple average (equal voting)
            ensemble_pred = (lstm_pred + cnn_pred + rnn_pred) / 3
            
            self.logger.info(f"Generated {len(ensemble_pred)} voting ensemble predictions")
            
            return ensemble_pred
            
        except Exception as e:
            self.logger.error(f"Error in voting prediction: {str(e)}")
            return None
    
    def predict_next_candles(self, recent_data: np.ndarray, n_candles: int, method: str = 'weighted') -> np.ndarray:
        """Predict next n candlesticks using ensemble approach"""
        try:
            if not self.is_trained:
                raise ValueError("Model must be trained before making predictions")
            
            predictions = []
            current_sequence = recent_data.copy()
            
            for i in range(n_candles):
                # Get next prediction from ensemble
                if method == 'ensemble' and self.ensemble_model is not None:
                    next_pred = self.ensemble_model.predict(current_sequence[-1:], verbose=0)
                else:
                    next_pred = self._predict_weighted(current_sequence[-1:])
                
                predictions.append(next_pred[0])
                
                # Update sequence for next prediction
                new_timestep = np.zeros((1, current_sequence.shape[2]))
                
                # Set OHLC values from prediction
                if len(next_pred[0]) >= 4:  # Assuming OHLC prediction
                    new_timestep[0, :4] = next_pred[0][:4]
                
                # Update sequence
                current_sequence = np.concatenate([
                    current_sequence[:, 1:, :],  # Remove first timestep
                    new_timestep.reshape(1, 1, -1)  # Add new timestep
                ], axis=1)
            
            return np.array(predictions)
            
        except Exception as e:
            self.logger.error(f"Error predicting next candles: {str(e)}")
            return None
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray, method: str = 'weighted') -> Dict:
        """Evaluate hybrid model performance"""
        try:
            if not self.is_trained:
                raise ValueError("Model must be trained before evaluation")
            
            # Get ensemble predictions
            ensemble_pred = self.predict(X_test, method=method)
            
            if ensemble_pred is None:
                return {}
            
            # Calculate ensemble metrics
            ensemble_metrics = self._calculate_metrics(y_test, ensemble_pred)
            ensemble_metrics['method'] = method
            
            # Get individual model metrics for comparison
            individual_metrics = {}
            
            if self.lstm_model.is_trained:
                lstm_pred = self.lstm_model.predict(X_test)
                individual_metrics['lstm'] = self._calculate_metrics(y_test, lstm_pred)
            
            if self.cnn_model.is_trained:
                cnn_pred = self.cnn_model.predict(X_test)
                individual_metrics['cnn'] = self._calculate_metrics(y_test, cnn_pred)
            
            if self.rnn_model.is_trained:
                rnn_pred = self.rnn_model.predict(X_test)
                individual_metrics['rnn'] = self._calculate_metrics(y_test, rnn_pred)
            
            result = {
                'ensemble': ensemble_metrics,
                'individual': individual_metrics,
                'ensemble_weights': self.ensemble_weights
            }
            
            self.logger.info(f"Hybrid model evaluation completed: {ensemble_metrics}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error evaluating hybrid model: {str(e)}")
            return {}
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Calculate evaluation metrics"""
        try:
            mse = mean_squared_error(y_true, y_pred)
            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            
            # Calculate MAPE
            mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
            
            # Calculate directional accuracy
            if y_true.shape[1] >= 4:  # Assuming OHLC format
                actual_direction = np.sign(y_true[:, 3] - y_true[:, 0])  # Close vs Open
                pred_direction = np.sign(y_pred[:, 3] - y_pred[:, 0])
                directional_accuracy = np.mean(actual_direction == pred_direction) * 100
            else:
                directional_accuracy = 0
            
            # Calculate R-squared
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            r2_score = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            return {
                'mse': float(mse),
                'mae': float(mae),
                'rmse': float(rmse),
                'mape': float(mape),
                'directional_accuracy': float(directional_accuracy),
                'r2_score': float(r2_score),
                'total_predictions': len(y_pred)
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating metrics: {str(e)}")
            return {}
    
    def _calculate_ensemble_metrics(self, X_val: np.ndarray, y_val: np.ndarray) -> Dict:
        """Calculate ensemble metrics on validation data"""
        try:
            ensemble_pred = self._predict_weighted(X_val)
            return self._calculate_metrics(y_val, ensemble_pred)
        except Exception as e:
            self.logger.error(f"Error calculating ensemble metrics: {str(e)}")
            return {}
    
    def save_model(self, filepath: str) -> bool:
        """Save the hybrid model"""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Save individual models
            base_path = filepath.replace('.h5', '')
            
            lstm_saved = self.lstm_model.save_model(f"{base_path}_lstm.h5")
            cnn_saved = self.cnn_model.save_model(f"{base_path}_cnn.h5")
            rnn_saved = self.rnn_model.save_model(f"{base_path}_rnn.h5")
            
            # Save ensemble model if exists
            ensemble_saved = True
            if self.ensemble_model is not None:
                self.ensemble_model.save(f"{base_path}_ensemble.h5")
            
            # Save hybrid model metadata
            metadata = {
                'config': self.config,
                'ensemble_weights': self.ensemble_weights,
                'is_trained': self.is_trained,
                'training_history': self.training_history,
                'model_type': 'Hybrid'
            }
            
            metadata_path = f"{base_path}_hybrid_metadata.pkl"
            joblib.dump(metadata, metadata_path)
            
            success = lstm_saved and cnn_saved and rnn_saved and ensemble_saved
            
            if success:
                self.logger.info(f"Hybrid model saved to {filepath}")
            else:
                self.logger.error("Failed to save some components of hybrid model")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error saving hybrid model: {str(e)}")
            return False
    
    def load_model(self, filepath: str) -> bool:
        """Load the hybrid model"""
        try:
            from tensorflow.keras.models import load_model
            
            base_path = filepath.replace('.h5', '')
            
            # Load individual models
            lstm_loaded = self.lstm_model.load_model(f"{base_path}_lstm.h5")
            cnn_loaded = self.cnn_model.load_model(f"{base_path}_cnn.h5")
            rnn_loaded = self.rnn_model.load_model(f"{base_path}_rnn.h5")
            
            # Load ensemble model if exists
            ensemble_path = f"{base_path}_ensemble.h5"
            if os.path.exists(ensemble_path):
                self.ensemble_model = load_model(ensemble_path)
            
            # Load metadata
            metadata_path = f"{base_path}_hybrid_metadata.pkl"
            if os.path.exists(metadata_path):
                metadata = joblib.load(metadata_path)
                self.config = metadata.get('config', self.config)
                self.ensemble_weights = metadata.get('ensemble_weights', self.ensemble_weights)
                self.is_trained = metadata.get('is_trained', True)
                self.training_history = metadata.get('training_history', {})
            
            success = lstm_loaded and cnn_loaded and rnn_loaded
            
            if success:
                self.logger.info(f"Hybrid model loaded from {filepath}")
            else:
                self.logger.error("Failed to load some components of hybrid model")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error loading hybrid model: {str(e)}")
            return False
    
    def get_model_summary(self) -> Dict:
        """Get summary of all models in the hybrid ensemble"""
        summary = {
            'hybrid_config': self.config,
            'ensemble_weights': self.ensemble_weights,
            'is_trained': self.is_trained
        }
        
        if self.lstm_model.model is not None:
            summary['lstm_summary'] = self.lstm_model.get_model_summary()
        
        if self.cnn_model.model is not None:
            summary['cnn_summary'] = self.cnn_model.get_model_summary()
        
        if self.rnn_model.model is not None:
            summary['rnn_summary'] = self.rnn_model.get_model_summary()
        
        if self.ensemble_model is not None:
            import io
            import sys
            
            old_stdout = sys.stdout
            sys.stdout = buffer = io.StringIO()
            self.ensemble_model.summary()
            sys.stdout = old_stdout
            
            summary['ensemble_summary'] = buffer.getvalue()
        
        return summary
    
    def plot_model_comparison(self, X_test: np.ndarray, y_test: np.ndarray, save_path: str = None) -> bool:
        """Plot comparison of all models"""
        try:
            import matplotlib.pyplot as plt
            
            if not self.is_trained:
                self.logger.warning("Models not trained yet")
                return False
            
            # Get predictions from all models
            predictions = {}
            
            if self.lstm_model.is_trained:
                predictions['LSTM'] = self.lstm_model.predict(X_test)
            
            if self.cnn_model.is_trained:
                predictions['CNN'] = self.cnn_model.predict(X_test)
            
            if self.rnn_model.is_trained:
                predictions['RNN'] = self.rnn_model.predict(X_test)
            
            predictions['Hybrid'] = self._predict_weighted(X_test)
            
            # Calculate metrics for each model
            metrics = {}
            for name, pred in predictions.items():
                metrics[name] = self._calculate_metrics(y_test, pred)
            
            # Create comparison plots
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Hybrid Model Comparison', fontsize=16)
            
            # Plot 1: MSE comparison
            models = list(metrics.keys())
            mse_values = [metrics[model]['mse'] for model in models]
            
            axes[0, 0].bar(models, mse_values)
            axes[0, 0].set_title('Mean Squared Error Comparison')
            axes[0, 0].set_ylabel('MSE')
            axes[0, 0].tick_params(axis='x', rotation=45)
            
            # Plot 2: Directional accuracy comparison
            acc_values = [metrics[model]['directional_accuracy'] for model in models]
            
            axes[0, 1].bar(models, acc_values)
            axes[0, 1].set_title('Directional Accuracy Comparison')
            axes[0, 1].set_ylabel('Accuracy (%)')
            axes[0, 1].tick_params(axis='x', rotation=45)
            
            # Plot 3: Prediction vs Actual (sample)
            sample_size = min(100, len(y_test))
            sample_indices = np.random.choice(len(y_test), sample_size, replace=False)
            
            axes[1, 0].scatter(y_test[sample_indices, 0], predictions['Hybrid'][sample_indices, 0], alpha=0.6)
            axes[1, 0].plot([y_test[:, 0].min(), y_test[:, 0].max()], 
                           [y_test[:, 0].min(), y_test[:, 0].max()], 'r--')
            axes[1, 0].set_title('Hybrid Model: Predicted vs Actual')
            axes[1, 0].set_xlabel('Actual')
            axes[1, 0].set_ylabel('Predicted')
            
            # Plot 4: R² comparison
            r2_values = [metrics[model]['r2_score'] for model in models]
            
            axes[1, 1].bar(models, r2_values)
            axes[1, 1].set_title('R² Score Comparison')
            axes[1, 1].set_ylabel('R² Score')
            axes[1, 1].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"Model comparison plot saved to {save_path}")
            
            plt.show()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error plotting model comparison: {str(e)}")
            return False
    
    def get_feature_importance_ensemble(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Get feature importance from ensemble perspective"""
        try:
            importance = {}
            
            # Get individual model importances
            if self.lstm_model.is_trained:
                importance['lstm'] = self.lstm_model.get_feature_importance(X, y)
            
            if self.cnn_model.is_trained:
                importance['cnn'] = self.cnn_model.get_pattern_importance(X, y)
            
            if self.rnn_model.is_trained:
                importance['rnn'] = self.rnn_model.get_sequence_importance(X, y)
            
            # Calculate weighted ensemble importance
            ensemble_importance = {}
            weights = np.array(self.ensemble_weights)
            
            # Combine importances using ensemble weights
            all_features = set()
            for model_imp in importance.values():
                all_features.update(model_imp.keys())
            
            for feature in all_features:
                weighted_imp = 0
                total_weight = 0
                
                if 'lstm' in importance and feature in importance['lstm']:
                    weighted_imp += weights[0] * importance['lstm'][feature]
                    total_weight += weights[0]
                
                if 'cnn' in importance and feature in importance['cnn']:
                    weighted_imp += weights[1] * importance['cnn'][feature]
                    total_weight += weights[1]
                
                if 'rnn' in importance and feature in importance['rnn']:
                    weighted_imp += weights[2] * importance['rnn'][feature]
                    total_weight += weights[2]
                
                if total_weight > 0:
                    ensemble_importance[feature] = weighted_imp / total_weight
            
            return {
                'individual': importance,
                'ensemble': ensemble_importance,
                'weights': self.ensemble_weights
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating ensemble feature importance: {str(e)}")
            return {}