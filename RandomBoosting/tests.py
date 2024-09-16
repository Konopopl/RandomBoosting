import unittest
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_text
import warnings

from randomboosting import RandomBoosting  
from randomboosting import MyModel  
# Предполагается, что класс RandomBoosting уже импортирован или определён

class TestRandomBoosting(unittest.TestCase):
    def setUp(self):
        # Игнорируем предупреждения, чтобы сохранить вывод чистым
        warnings.filterwarnings("ignore")
        
        # Загрузка датасета и разделение на обучающие и тестовые наборы
        iris = load_iris()
        X = pd.DataFrame(iris.data, columns=iris.feature_names)
        y = pd.Series(iris.target)  # Конвертируем в Series для корректного использования .sample()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Инициализация модели RandomBoosting
        self.ensemble = RandomBoosting(
            model_classification=GradientBoostingClassifier(),
            n_models=10,
            model_features='sqrt',
            random_state=42,
            n_jobs=1,
            bootstrap=0.8,
            level_of_trust=0.0,
            warm_start=True  # Устанавливаем warm_start=True для тестирования инкрементального обучения
        )
    
    def test_initial_fit(self):
        # Тест начального обучения модели
        self.ensemble.fit(self.X_train, self.y_train)
        y_pred = self.ensemble.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        self.assertGreaterEqual(accuracy, 0.8, "Initial accuracy should be at least 0.8")
        # Проверка, что модели обучены
        self.assertTrue(hasattr(self.ensemble, 'modelsTrast'))
        self.assertEqual(len(self.ensemble.modelsTrast), self.ensemble.n_models)
    
    def test_warm_start_increment_n_estimators(self):
        # Тест инкрементального увеличения n_estimators
        self.ensemble.fit(self.X_train, self.y_train)
        # Получаем n_estimators перед обновлением
        n_estimators_before = self.ensemble.modelsTrast[0].model.n_estimators
        # Обучение модели с увеличением n_estimators
        self.ensemble.fit(self.X_train, self.y_train, increment_n_estimators=50)
        n_estimators_after = self.ensemble.modelsTrast[0].model.n_estimators
        self.assertEqual(n_estimators_after, n_estimators_before + 50, "n_estimators should be incremented by 50")
        # Проверка, что модели были переобучены
        training_time = self.ensemble.modelsTrast[0].training_time
        self.assertGreater(training_time, 0, "Training time should be greater than 0 after warm start")
    
    def test_warm_start_increment_n_models(self):
        # Тест добавления новых моделей в ансамбль
        self.ensemble.fit(self.X_train, self.y_train)
        n_models_before = len(self.ensemble.modelsTrast)
        # Добавляем 5 новых моделей
        self.ensemble.fit(self.X_train, self.y_train, increment_n_models=5)
        n_models_after = len(self.ensemble.modelsTrast)
        self.assertEqual(n_models_after, n_models_before + 5, "Number of models should be incremented by 5")
        # Проверка, что новые модели были обучены
        new_model_indices = [model.index for model in self.ensemble.modelsTrast][-5:]
        expected_indices = list(range(n_models_before, n_models_after))
        self.assertEqual(new_model_indices, expected_indices, "Indices of new models should match expected values")
    
    def test_models_access(self):
        # Проверка доступа к полям моделей из self.modelsTrast
        self.ensemble.fit(self.X_train, self.y_train)
        for model in self.ensemble.modelsTrast:
            # Убедитесь, что каждая модель имеет необходимые атрибуты
            self.assertTrue(hasattr(model, 'model'))
            self.assertTrue(hasattr(model, 'selected_features'))
            self.assertTrue(hasattr(model, 'validation_accuracy'))
            # Проверка, что selected_features не пустой
            self.assertGreater(len(model.selected_features), 0, "Selected features should not be empty")
            # Проверка, что validation_accuracy имеет значение
            self.assertIsInstance(model.validation_accuracy, float)
    
    def test_predict_proba(self):
        # Тест метода predict_proba
        self.ensemble.fit(self.X_train, self.y_train)
        y_proba = self.ensemble.predict_proba(self.X_test)
        self.assertEqual(y_proba.shape, (len(self.X_test), len(np.unique(self.y_train))))
        # Проверка, что сумма вероятностей равна 1 для каждого образца
        self.assertTrue(np.allclose(y_proba.sum(axis=1), 1), "Sum of probabilities should be 1 for each sample")
    
    def test_feature_importances(self):
        # Проверка вычисления средней важности признаков
        self.ensemble.fit(self.X_train, self.y_train)
        importances = self.ensemble.feature_importances_
        self.assertEqual(len(importances), self.X_train.shape[1])
        # Убедитесь, что importances не содержит NaN
        self.assertFalse(np.isnan(importances).any(), "Feature importances should not contain NaN")
        # Убедитесь, что importances неотрицательные
        self.assertTrue((importances >= 0).all(), "Feature importances should be non-negative")
    
    def test_feature_importances_variance(self):
        # Проверка вычисления дисперсии важности признаков
        self.ensemble.fit(self.X_train, self.y_train)
        variances = self.ensemble.feature_importances_var_
        self.assertEqual(len(variances), self.X_train.shape[1])
        # Убедитесь, что дисперсии неотрицательные
        self.assertTrue((variances >= 0).all(), "Feature importance variances should be non-negative")
    
    def test_level_of_trust(self):
        # Тест параметра level_of_trust
        self.ensemble.level_of_trust = 0.9  # Устанавливаем высокий уровень доверия
        self.ensemble.fit(self.X_train, self.y_train, X_val=self.X_test, y_val=self.y_test)
        # Проверка, что некоторые модели были исключены из голосования
        n_trusted_models = len(self.ensemble.modelsTrast)
        self.assertLessEqual(n_trusted_models, self.ensemble.n_models, "Number of trusted models should be less than or equal to n_models")
        # Проверка, что validation_accuracy моделей в modelsTrast не ниже level_of_trust
        for model in self.ensemble.modelsTrast:
            self.assertGreaterEqual(model.validation_accuracy, self.ensemble.level_of_trust, "Model's validation accuracy should be >= level_of_trust")
    
    def test_bootstrap(self):
        # Тест работы с bootstrap < 1
        self.ensemble.bootstrap = 0.5  # Используем только половину данных для каждой модели
        self.ensemble.fit(self.X_train, self.y_train)
        # Проверка, что модели были обучены
        for model in self.ensemble.modelsTrast:
            self.assertIsNotNone(model.training_time)
            self.assertGreater(model.training_time, 0)
    
    def test_invalid_model_features(self):
        # Проверка обработки некорректного значения model_features
        self.ensemble.model_features = 'invalid_value'
        with self.assertRaises(ValueError):
            self.ensemble.fit(self.X_train, self.y_train)
    
    def test_predict_before_fit(self):
        # Проверка вызова predict до fit
        ensemble_unfitted = RandomBoosting()
        with self.assertRaises(ValueError):
            ensemble_unfitted.predict(self.X_test)
    
    def test_predict_proba_before_fit(self):
        # Проверка вызова predict_proba до fit
        ensemble_unfitted = RandomBoosting()
        with self.assertRaises(ValueError):
            ensemble_unfitted.predict_proba(self.X_test)
    
    def test_increment_n_models_and_n_estimators(self):
        # Тест одновременного увеличения n_models и n_estimators
        self.ensemble.fit(self.X_train, self.y_train)
        n_models_before = len(self.ensemble.modelsTrast)
        n_estimators_before = self.ensemble.modelsTrast[0].model.n_estimators
        # Увеличиваем оба
        self.ensemble.fit(self.X_train, self.y_train, increment_n_models=5, increment_n_estimators=50)
        n_models_after = len(self.ensemble.modelsTrast)
        n_estimators_after = self.ensemble.modelsTrast[0].model.n_estimators
        self.assertEqual(n_models_after, n_models_before + 5, "Number of models should be incremented by 5")
        self.assertEqual(n_estimators_after, n_estimators_before + 50, "n_estimators should be incremented by 50")
    
    def test_model_random_state_uniqueness(self):
        # Проверка уникальности random_state у моделей
        self.ensemble.fit(self.X_train, self.y_train)
        random_states = [model.random_state for model in self.ensemble.modelsTrast]
        self.assertEqual(len(random_states), len(set(random_states)), "Random states should be unique for each model")
    
    def test_selected_features_diversity(self):
        # Проверка разнообразия выбранных признаков для моделей
        self.ensemble.fit(self.X_train, self.y_train)
        selected_features = [tuple(model.selected_features) for model in self.ensemble.modelsTrast]
        self.assertGreater(len(set(selected_features)), 1, "There should be diversity in selected features among models")
    
    def test_parallel_training(self):
        # Тест параллельного обучения
        self.ensemble.n_jobs = -1  # Используем все доступные ядра
        self.ensemble.fit(self.X_train, self.y_train)
        # Проверка, что обучение завершилось и модели были обучены
        for model in self.ensemble.modelsTrast:
            self.assertIsNotNone(model.training_time)
            self.assertGreater(model.training_time, 0)
    
    def test_exception_handling_in_fit_model(self):
        # Тест обработки исключений в fit_model
        # Передаем некорректные данные, чтобы вызвать исключение
        self.X_train.iloc[:, 0] = 'invalid_data'
        with self.assertRaises(ValueError):
            self.ensemble.fit(self.X_train, self.y_train)

    def test_n_estimators_equals_one(self):
        # Тест соответствия с RandomForestClassifier при n_estimators=1
        rf = RandomForestClassifier(n_estimators=1, max_depth=3, random_state=42)
        rf.fit(self.X_train, self.y_train)
        y_pred_rf = rf.predict(self.X_test)

        # Настройка модели RandomBoosting
        self.ensemble = RandomBoosting(
            model_classification=GradientBoostingClassifier(n_estimators=1, max_depth=3),
            n_models=1,  # Только одна модель
            random_state=42,
            bootstrap=1.0  # Без подвыборки
        )
        self.ensemble.fit(self.X_train, self.y_train)
        y_pred_rb = self.ensemble.predict(self.X_test)

        # Сравнение предсказаний
        np.testing.assert_array_equal(y_pred_rf, y_pred_rb, "Predictions should be the same for n_estimators=1")

    def test_tree_structure(self):
        # Тест соответствия структуры дерева
        dt = DecisionTreeClassifier(max_depth=2, random_state=42)
        dt.fit(self.X_train, self.y_train)
        tree_rules = export_text(dt, feature_names=list(self.X_train.columns))

        # Настройка модели RandomBoosting с одним деревом
        self.ensemble = RandomBoosting(
            model_classification=GradientBoostingClassifier(n_estimators=1, max_depth=2),
            n_models=1,  # Только одна модель
            random_state=42,
            bootstrap=1.0  # Без подвыборки
        )
        self.ensemble.fit(self.X_train, self.y_train)
        est = self.ensemble.modelsTrast[0].model.estimators_[0, 0]
        est_rules = export_text(est, feature_names=list(self.X_train.columns))

        # Сравнение структур дерева
        self.assertEqual(tree_rules, est_rules, "Tree structures should be the same")

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
