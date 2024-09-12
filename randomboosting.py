import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeClassifier, plot_tree 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix 
import matplotlib.pyplot as plt 
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier 
from sklearn.utils import shuffle 
from sklearn.base import BaseEstimator, ClassifierMixin 
import copy 
import time 
from joblib import Parallel, delayed  
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.utils.validation import check_is_fitted 
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer 
from sklearn.preprocessing import normalize 
from sklearn.metrics import average_precision_score 
import plotly.graph_objects as go 
import tempfile  
from scipy.stats import mode 
from itertools import product 

class MyModel:
    def __init__(self, model, index, random_state=None):
        self.model = model
        self.index = index
        self.selected_features = None
        self.validation_accuracy = 0
        self.weight = None
        self.feature_importance_dict = {}
        self.education_time = 0 
        self.random_state = random_state

class RandomBoosting(BaseEstimator, ClassifierMixin):
    def __init__( 
        self,  
        model_classification = GradientBoostingClassifier( 
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            max_features=1,     
        ),
        n_models = 60, 
        model_features='sqrt',  # number of features each boosting model is trained on
        random_state=42, 
        n_jobs=-1, 
        bootstrap=1.0, 
        level_of_trust=0.0,  # if level_of_trust > 0, models are evaluated on X_val, and those with lower values are excluded from voting
        voting_weights=0.0, 
        **kwargs): 
                self.model_classification =  model_classification
                self.n_models = n_models
                self.model_features = model_features
                self.random_state = random_state
                self.n_jobs = n_jobs
                self.bootstrap = bootstrap
                self.level_of_trust = level_of_trust
                self.voting_weights = voting_weights 
                self.MyTrainedModels = []
        
                # Initialize an immutable array of MyModel objects
                self.models = tuple(
                    MyModel(
                        model=self.model_classification.set_params(random_state=self.random_state + i),
                        index=i, 
                        random_state=self.random_state + i,
                    )
                    for i in range(self.n_models)
                )

    def fit(self, X_train, y_train, X_val=None, y_val=None): 
        self.n_features_in_ = len(X_train.columns) 
        self.feature_names_in_ = np.array(X_train.columns) 
        self.classes_ = np.unique(y_train)
        # for weighted voting
        if self.level_of_trust > 0 or self.voting_weights > 0:
            if X_val is None:
                X_val = X_train
            if y_val is None:
                y_val = y_train  

        if type(self.model_features) == int:   self.my_model_features = self.model_features 
        if type(self.model_features) == float: self.my_model_features = int(self.model_features * len(X_train.columns)) 
        if self.model_features == 'sqrt': self.my_model_features = int(np.sqrt(len(X_train.columns)))   
            
        for my_model in self.models:
            my_model.selected_features = np.random.default_rng(my_model.random_state).choice(X_train.columns, size=self.my_model_features, replace=False)

        # Record the start time for the entire ensemble training
        ensemble_start_time = time.time()

        # Train models in parallel
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(self.fit_model)(my_model, X_train, y_train, X_val, y_val)
            for my_model in self.models
        )

        # Record the end time for the entire ensemble training
        ensemble_end_time = time.time()
        self.max_time = ensemble_end_time - ensemble_start_time  # Time spent training the entire ensemble

        self.MyTrainedModels.extend(results)
        if len(self.models) == 0:
            raise RuntimeError("len(self.models) == 0") 

        self.modelsTrast = [] 
        for model in self.MyTrainedModels: 
            if model.validation_accuracy >= self.level_of_trust: 
                self.modelsTrast += [model]
        return self

    def fit_model(self, my_model, X_train, y_train, X_val=None, y_val=None):
        if self.bootstrap < 1:
            X_subset, y_subset = X_train.sample(frac=self.bootstrap, replace=True, random_state=my_model.random_state), \
                                 y_train.sample(frac=self.bootstrap, replace=True, random_state=my_model.random_state) 
        else:
            X_subset, y_subset = X_train, y_train
        # Train the model
        start_time = time.time()
        try:
            my_model.model.fit(X_subset[my_model.selected_features], y_subset)
            end_time = time.time()
            my_model.education_time = end_time - start_time  # Record the time spent training the model

            # Feature importance assessment
            feature_importances = my_model.model.feature_importances_
            my_model.feature_importance_dict = {feature: importance for feature, importance in zip(my_model.selected_features, feature_importances)}

            # Add nan for unused features
            for feature in X_train.columns:
                if feature not in my_model.feature_importance_dict:
                    my_model.feature_importance_dict[feature] = np.nan

            if self.level_of_trust > 0 or self.voting_weights > 0: 
                if my_model.selected_features is None or len(X_val) == 0:
                    raise ValueError(f"{my_model.index} are not features or X_val")
                accuracy = accuracy_score(my_model.model.predict(X_val[my_model.selected_features]), y_val)
                my_model.validation_accuracy = accuracy 
            return my_model 
        except: 
            raise ValueError(f"{my_model.index} ERROR")
        
    def predict(self, X):
        # Check if models have been trained
        check_is_fitted(self, 'models')
    
        # Check if classes have been initialized
        if not hasattr(self, 'classes_') or len(self.classes_) == 0:
            raise ValueError("The model has not been fitted yet or classes_ is not initialized.")

        # Get predictions from each model for all objects in X
        predictions = np.asarray([
            my_model.model.predict(X[my_model.selected_features])
            for my_model in self.modelsTrast])
    
        # Apply majority voting for each object
        # mode() returns the most frequently occurring value for each column (i.e., for each object)
        most_common_votes = mode(predictions, axis=0, keepdims=False)
        return most_common_votes.mode.flatten() 

    def predict_proba(self, X):  
        
        check_is_fitted(self, 'models')
        # Check if classes have been initialized
        if not hasattr(self, 'classes_') or len(self.classes_) == 0:
            raise ValueError("The model has not been fitted yet or classes_ is not initialized.") 
                
        proba_sum = np.zeros((X.shape[0], len(self.classes_)))
        for my_model in self.modelsTrast:
            model_proba = my_model.model.predict_proba(X[my_model.selected_features])
            proba_sum += model_proba
    
        avg_proba = proba_sum / len(self.modelsTrast)    
        return avg_proba


    @property
    def feature_importances_(self):
        feature_names = self.feature_names_in_
        importances_list = []
        for model in self.modelsTrast:
            importances = [model.feature_importance_dict.get(feature, np.nan) for feature in feature_names]
            importances_list.append(importances)
        importances_array = np.array(importances_list)
        means = np.nanmean(importances_array, axis=0)
        self.mean_variance = np.nanmean(means)  
        return means 

    @property
    def feature_importances_var_(self):
        df_importance = pd.DataFrame(model.feature_importance_dict for model in self.modelsTrast)  
        df_importance = df_importance[self.feature_names_in_]
        variances = df_importance.var(skipna=True)
        return np.array(variances)



