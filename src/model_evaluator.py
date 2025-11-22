import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

class ModelEvaluator:
    """
    Evaluates and compares trained regression models.
    
    Inputs:
    - models_dict: dictionary from TrainModel.train_models()
    - X_train, X_test, y_train, y_test: datasets used for evaluation
    - cv: number of folds for cross-validation
    """

    def __init__(self, models_dict, X_train, X_test, y_train, y_test, cv=5):
        self.models_dict = models_dict
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.cv = cv
        self.results_df = None

    # -----------------------------
    # Evaluate all models
    # -----------------------------
    def evaluate_models(self):
        results = []

        for model_name, info in self.models_dict.items():
            model = info['model']

            # TRAINING PREDICTIONS AND METRICS
            y_train_pred = model.predict(self.X_train)
            train_r2 = r2_score(self.y_train, y_train_pred)
            train_mae = mean_absolute_error(self.y_train, y_train_pred)
            train_mse = mean_squared_error(self.y_train, y_train_pred)
            train_rmse = np.sqrt(train_mse)

            # TESTING PREDICTIONS AND METRICS
            y_test_pred = model.predict(self.X_test)
            test_r2 = r2_score(self.y_test, y_test_pred)
            test_mae = mean_absolute_error(self.y_test, y_test_pred)
            test_mse = mean_squared_error(self.y_test, y_test_pred)
            test_rmse = np.sqrt(test_mse)

            # CROSS-VALIDATION METRICS (using training data)
            cv_preds = cross_val_predict(model, self.X_train, self.y_train, cv=self.cv)
            cv_r2 = r2_score(self.y_train, cv_preds)
            cv_mae = mean_absolute_error(self.y_train, cv_preds)
            cv_mse = mean_squared_error(self.y_train, cv_preds)
            cv_rmse = np.sqrt(cv_mse)

            # Collect results
            results.append({
                'Model': model_name,
                'Train_R2': train_r2,
                'Test_R2': test_r2,
                'CV_R2': cv_r2,
                'Train_MAE': train_mae,
                'Test_MAE': test_mae,
                'CV_MAE': cv_mae,
                'Train_MSE': train_mse,
                'Test_MSE': test_mse,
                'CV_MSE': cv_mse,
                'Train_RMSE': train_rmse,
                'Test_RMSE': test_rmse,
                'CV_RMSE': cv_rmse,
                'Best_Params': info['best_params']
            })

        self.results_df = pd.DataFrame(results).sort_values(by='Test_R2', ascending=False).reset_index(drop=True)
        return self.results_df

    # -----------------------------
    # Compare models with plots
    # -----------------------------
    def plot_comparison(self, figsize=(12,6)):
        if self.results_df is None:
            raise ValueError("Run evaluate_models() before plotting.")
        
        self.results_df.set_index('Model')[['Train_R2', 'Test_R2', 'CV_R2']].plot(
            kind='bar', figsize=figsize
        )
        plt.title("Model Performance Comparison (R²)")
        plt.ylabel("R² Score")
        plt.xticks(rotation=45)
        plt.grid(axis='y')
        plt.show()

    # -----------------------------
    # Identify best model & check overfitting/underfitting
    # -----------------------------
    def get_best_model(self):
        if self.results_df is None:
            raise ValueError("Run evaluate_models() first.")
        
        best_row = self.results_df.iloc[0]
        model_name = best_row['Model']
        train_r2 = best_row['Train_R2']
        test_r2 = best_row['Test_R2']

        overfitting = False
        underfitting = False

        if train_r2 - test_r2 > 0.1:
            overfitting = True
        if train_r2 < 0.3 and test_r2 < 0.3:
            underfitting = True

        summary = f"Best Model: {model_name}\n"
        summary += f"Train R²: {train_r2:.3f}, Test R²: {test_r2:.3f}, CV R²: {best_row['CV_R2']:.3f}\n"
        summary += f"Train RMSE: {best_row['Train_RMSE']:.3f}, Test RMSE: {best_row['Test_RMSE']:.3f}\n"
        summary += f"Train MAE: {best_row['Train_MAE']:.3f}, Test MAE: {best_row['Test_MAE']:.3f}\n"
        summary += f"Best Hyperparameters: {best_row['Best_Params']}\n"

        if overfitting:
            summary += "Warning: Model may be overfitting (train >> test).\n"
        elif underfitting:
            summary += "Warning: Model may be underfitting (low train/test scores).\n"
        else:
            summary += "Model shows good generalization.\n"

        return summary