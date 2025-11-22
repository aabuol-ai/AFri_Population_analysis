import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor

class TrainModel:
    """
    Handles:
    - Train-test split (with identifier alignment)
    - Numeric feature scaling
    - Training multiple regression models
    - Hyperparameter tuning
    - Returning all fitted models and training info for evaluation
    """

    def __init__(self, df, target):
        self.df = df
        self.target = target

        # Data splits
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        self.train_indices = None
        self.test_indices = None

        self.scaler = None

        # Storage for all trained models and their info
        self.models_dict = {}

    # ---------------------------------------------------------------------
    # TRAIN-TEST SPLIT
    # ---------------------------------------------------------------------
    def train_test_split(self, test_size=0.2, random_state=42):
        df_clean = self.df.dropna(subset=[self.target]).reset_index(drop=True)

        # Store identifiers if needed
        self.identifiers = df_clean[["region_country_area", "year", "code"]].copy() if "region_country_area" in df_clean.columns else None

        # Separate predictors and target
        X = df_clean.drop(columns=[self.target])
        y = df_clean[self.target]

        # Keep only numeric predictors
        X = X.select_dtypes(include=["float64", "int64"])

        # Split
        self.X_train, self.X_test, self.y_train, self.y_test, idx_train, idx_test = train_test_split(
            X, y, df_clean.index, test_size=test_size, random_state=random_state
        )

        # Save indices for mapping if needed
        self.train_indices = idx_train
        self.test_indices = idx_test

        return self.X_train, self.X_test, self.y_train, self.y_test
    #----------------------------------------------------------------------
    # REMOVE BLANK ROWS
    #----------------------------------------------------------------------
        
    def clean_missing_rows(self):
        before = len(self.df)# Drop rows where target or any numeric feature is missing
        numeric_cols = self.df.select_dtypes(include=["float64", "int64"]).columns
        self.df = self.df.dropna(subset=[self.target] + list(numeric_cols))
        after = len(self.df)
        print(f"Removed {before - after} rows with missing values.")
        return self.df
    
    # ---------------------------------------------------------------------
    # SCALING
    # ---------------------------------------------------------------------
    def scale_numeric(self):
        numeric_cols = self.X_train.columns
        self.scaler = StandardScaler()
        self.X_train[numeric_cols] = self.scaler.fit_transform(self.X_train[numeric_cols])
        self.X_test[numeric_cols] = self.scaler.transform(self.X_test[numeric_cols])
        return self.X_train, self.X_test

    # ---------------------------------------------------------------------
    # MODEL CONFIGS
    # ---------------------------------------------------------------------
    def get_model_configs(self):
        """
        Returns a dictionary with model_name: (estimator, hyperparameters)
        Must include at least 3 regression models
        """
        return {
            "LinearRegression": (LinearRegression(), {}),
            "RandomForestRegressor": (
                RandomForestRegressor(random_state=42),
                {"n_estimators": [200, 400], "max_depth": [None, 10, 20]}
            ),
            "Ridge": (
                Ridge(),
                {"alpha": [0.1, 1.0, 10.0]}
            )
        }

    # ---------------------------------------------------------------------
    # TRAIN ALL MODELS WITH HYPERPARAMETER TUNING
    # ---------------------------------------------------------------------
    def train_models(self, cv=5):
        configs = self.get_model_configs()
        self.models_dict = {}

        for name, (model, params) in configs.items():
            print(f"\nTraining: {name}")

            try:
                if len(params) == 0:
                    # Simple fit
                    model.fit(self.X_train, self.y_train)
                    self.models_dict[name] = {
                        "model": model,
                        "best_params": "default",
                        "cv_results": None
                    }
                else:
                    # Hyperparameter tuning
                    grid = GridSearchCV(
                        estimator=model,
                        param_grid=params,
                        cv=cv,
                        scoring="r2",
                        n_jobs=-1
                    )
                    grid.fit(self.X_train, self.y_train)
                    self.models_dict[name] = {
                        "model": grid.best_estimator_,
                        "best_params": grid.best_params_,
                        "cv_results": grid.cv_results_
                    }

            except Exception as e:
                print(f"Model {name} FAILED. Reason: {e}")

        return self.models_dict