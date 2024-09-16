from dataclasses import dataclass, field
import numpy as np
import pandas as pd
import time
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.utils.validation import check_is_fitted
from joblib import Parallel, delayed
from scipy.stats import mode

@dataclass
class MyModel:
    # Wrapper class for an individual model in the ensemble
    model: any
    index: int
    random_state: int = None
    selected_features: list = field(default_factory=list)
    validation_accuracy: float = 0.0
    weight: float = None
    feature_importance_dict: dict = field(default_factory=dict)
    training_time: float = 0.0

class RandomBoosting(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        model_classification=GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            max_features=1,
        ),
        n_models=60,
        model_features='sqrt',  # Number of features for training each boosting model
        random_state=42,
        n_jobs=-1,
        bootstrap=1.0,
        level_of_trust=0.0,  # If > 0, models are evaluated on X_val, and those with lower accuracy are excluded from voting
        voting_weights=0.0,
        warm_start=False,  # Flag to enable ensemble incremental learning
        **kwargs
    ):
        # Model parameters
        self.model_classification = model_classification
        self.n_models = n_models
        self.model_features = model_features
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.bootstrap = bootstrap
        self.level_of_trust = level_of_trust
        self.voting_weights = voting_weights
        self.warm_start = warm_start
        self.MyTrainedModels = []
        self.models = ()
        # Save initial parameter values
        self.initial_n_estimators = self.model_classification.get_params().get('n_estimators', 100)
        self.initial_n_models = self.n_models

    def fit(self, X_train, y_train, X_val=None, y_val=None, increment_n_models=0, increment_n_estimators=0):
        # Initialize necessary attributes if they don't exist yet
        self.n_features_in_ = getattr(self, 'n_features_in_', X_train.shape[1])
        self.feature_names_in_ = getattr(self, 'feature_names_in_', np.array(X_train.columns))
        self.classes_ = getattr(self, 'classes_', np.unique(y_train))
        # Validation data for weighted voting and level of trust
        self.X_train, self.y_train = X_train, y_train
        self.X_val = X_val if X_val is not None else X_train if self.level_of_trust > 0 or self.voting_weights > 0 else None
        self.y_val = y_val if y_val is not None else y_train if self.level_of_trust > 0 or self.voting_weights > 0 else None
        # Determine the number of features for each model
        self._determine_model_features()
        # Check if models need to be trained from scratch or incrementally
        if not self.warm_start or not getattr(self, 'is_fitted_', False):
            # Train models from scratch
            self._initialize_models()
            self._select_features_for_models()
            self._train_models()
            self.is_fitted_ = True  # Set flag indicating the ensemble is trained
        else:
            # Incrementally train existing models or add new ones
            if increment_n_estimators > 0:
                # Update existing models by increasing n_estimators
                self._update_existing_models(increment_n_estimators)
            if increment_n_models > 0:
                # Add new models to the ensemble
                self._add_new_models(increment_n_models)
            # Train new models and incrementally train existing ones
            self._train_models()
        # Update the list of trusted models based on level_of_trust
        self._update_models_trust()
        return self

    def _determine_model_features(self):
        # Determine the number of features for training each model in the ensemble.
        if isinstance(self.model_features, int):
            self.my_model_features = self.model_features
        elif isinstance(self.model_features, float):
            self.my_model_features = int(self.model_features * self.n_features_in_)
        elif self.model_features == 'sqrt':
            self.my_model_features = int(np.sqrt(self.n_features_in_))
        else:
            raise ValueError("Invalid value for model_features")

    def _initialize_models(self):
        # Initialize models in the ensemble during first training or when warm_start=False.
        self.models = tuple(
            MyModel(
                model=clone(self.model_classification).set_params(
                    random_state=self.random_state + i,
                    n_estimators=self.initial_n_estimators
                ),
                index=i,
                random_state=self.random_state + i,
            )
            for i in range(self.n_models)
        )

    def _select_features_for_models(self):
        # Select random features for each model in the ensemble.
        columns = self.feature_names_in_
        for my_model in self.models:
            rng = np.random.default_rng(my_model.random_state)
            my_model.selected_features = rng.choice(
                columns, size=self.my_model_features, replace=False
            )

    def _train_models(self):
        # Train all models in the ensemble.
        # Record the start time of ensemble training
        ensemble_start_time = time.time()
        # Separate models into new and existing
        new_models = [model for model in self.models if not hasattr(model.model, 'estimators_')]
        existing_models = [model for model in self.models if hasattr(model.model, 'estimators_')]

        # Train new models in parallel
        if new_models:
            results_new = Parallel(n_jobs=self.n_jobs)(
                delayed(self.fit_model)(my_model)
                for my_model in new_models
            )
        else:
            results_new = []
        # Incrementally train existing models sequentially
        for my_model in existing_models:
            self._fit_model_sequential(my_model)
        # Combine results
        self.MyTrainedModels = results_new + existing_models
        # Total ensemble training time
        self.max_time = time.time() - ensemble_start_time

    def _update_existing_models(self, increment_n_estimators):
        # Update existing models by increasing the number of trees (n_estimators).
        for my_model in self.models:
            if hasattr(my_model.model, 'warm_start') and isinstance(my_model.model, GradientBoostingClassifier):
                # Set warm_start=True for incremental training
                n_estimators = my_model.model.get_params().get('n_estimators', 100) + increment_n_estimators
                my_model.model.set_params(warm_start=True, n_estimators=n_estimators)
            else:
                # If warm_start is not supported, create a new model with increased n_estimators
                new_n_estimators = my_model.model.get_params().get('n_estimators', 100) + increment_n_estimators
                my_model.model = clone(self.model_classification).set_params(
                    random_state=my_model.random_state,
                    n_estimators=new_n_estimators
                )

    def _add_new_models(self, increment_n_models):
        # Add new models to the ensemble, increasing its size.
        # Generate new indices for added models
        start_index = max((model.index for model in self.models), default=-1) + 1
        new_models = tuple(
            MyModel(
                model=clone(self.model_classification).set_params(
                    random_state=self.random_state + i,
                    n_estimators=self.initial_n_estimators
                ),
                index=i,
                random_state=self.random_state + i,
            )
            for i in range(start_index, start_index + increment_n_models)
        )
        # Add new models to the self.models tuple
        self.models += new_models
        # Select random features for new models
        self._select_features_for_models()

    def _update_models_trust(self):
        # Update the list of trusted models based on the level_of_trust parameter.
        self.modelsTrast = [model for model in self.MyTrainedModels if model.validation_accuracy >= self.level_of_trust]

    def fit_model(self, my_model):
        # Train a single model in the ensemble (used for new models).
        # If using bootstrap, select a subset of data
        X_train, y_train = self.X_train, self.y_train
        if self.bootstrap < 1:
            X_subset = X_train.sample(frac=self.bootstrap, replace=True, random_state=my_model.random_state)
            y_subset = y_train.sample(frac=self.bootstrap, replace=True, random_state=my_model.random_state)
        else:
            X_subset, y_subset = X_train, y_train
        # Train the model and record training time
        start_time = time.time()
        try:
            my_model.model.fit(X_subset[my_model.selected_features], y_subset)
            my_model.training_time = time.time() - start_time  # Model training time
            # Feature importance evaluation
            feature_importances = my_model.model.feature_importances_
            my_model.feature_importance_dict = dict(zip(my_model.selected_features, feature_importances))
            # Evaluate model accuracy on the validation set if required
            if (self.level_of_trust > 0 or self.voting_weights > 0) and self.X_val is not None:
                accuracy = accuracy_score(
                    my_model.model.predict(self.X_val[my_model.selected_features]), self.y_val
                )
                my_model.validation_accuracy = accuracy
            return my_model
        except Exception as e:
            raise ValueError(f"Error in model {my_model.index}: {e}")

    def _fit_model_sequential(self, my_model):
        # Incrementally train an existing model sequentially (used for models with warm_start).
        # If using bootstrap, select a subset of data
        X_train, y_train = self.X_train, self.y_train
        if self.bootstrap < 1:
            X_subset = X_train.sample(frac=self.bootstrap, replace=True, random_state=my_model.random_state)
            y_subset = y_train.sample(frac=self.bootstrap, replace=True, random_state=my_model.random_state)
        else:
            X_subset, y_subset = X_train, y_train

        # Train the model and record training time
        start_time = time.time()
        try:
            my_model.model.fit(X_subset[my_model.selected_features], y_subset)
            my_model.training_time += time.time() - start_time  # Update model training time

            # Update feature importance
            feature_importances = my_model.model.feature_importances_
            my_model.feature_importance_dict = dict(zip(my_model.selected_features, feature_importances))

            # Compute model accuracy on the validation set if required
            if (self.level_of_trust > 0 or self.voting_weights > 0) and self.X_val is not None:
                accuracy = accuracy_score(
                    my_model.model.predict(self.X_val[my_model.selected_features]), self.y_val
                )
                my_model.validation_accuracy = accuracy
        except Exception as e:
            raise ValueError(f"Error in model {my_model.index}: {e}")

    def predict(self, X):
        check_is_fitted(self, 'models')
        if not hasattr(self, 'classes_') or len(self.classes_) == 0:
            raise ValueError("The model is not trained or classes are not initialized.")

        # Get predictions from each model for all objects in X
        predictions = np.array([
            my_model.model.predict(X[my_model.selected_features])
            for my_model in self.modelsTrast
        ])
        # Apply majority voting for each object
        return mode(predictions, axis=0, keepdims=False).mode.flatten()

    def predict_proba(self, X):
        check_is_fitted(self, 'models')
        if not hasattr(self, 'classes_') or len(self.classes_) == 0:
            raise ValueError("The model is not trained or classes are not initialized.")
        proba_sum = sum(
            my_model.model.predict_proba(X[my_model.selected_features])
            for my_model in self.modelsTrast
        )
        avg_proba = proba_sum / len(self.modelsTrast)
        return avg_proba

    @property
    def feature_importances_(self):
        # Compute the average feature importance across all models in the ensemble.
        importances_list = [
            [model.feature_importance_dict.get(feature, np.nan) for feature in self.feature_names_in_]
            for model in self.modelsTrast
        ]
        means = np.nanmean(importances_list, axis=0)
        self.mean_variance = np.nanmean(means)
        return means

    @property
    def feature_importances_var_(self):
        # Compute the variance of feature importance across all models in the ensemble.
        df_importance = pd.DataFrame(
            [model.feature_importance_dict for model in self.modelsTrast],
            columns=self.feature_names_in_
        )
        variances = df_importance.var(skipna=True)
        return variances.values