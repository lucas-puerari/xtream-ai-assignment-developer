import optuna
import xgboost
from sklearn.metrics import mean_absolute_error


SEARCH_BUDGET = 100


def optmize_xgboost_model(x_train, x_test, y_train, y_test):
    def objective(trial: optuna.trial.Trial):
        param = {
            'lambda': trial.suggest_float('lambda', 1e-8, 1.0, log=True),
            'alpha': trial.suggest_float('alpha', 1e-8, 1.0, log=True),
            'colsample_bytree': trial.suggest_categorical('colsample_bytree', [0.3, 0.4, 0.5, 0.7]),
            'subsample': trial.suggest_categorical('subsample', [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
            'learning_rate': trial.suggest_float('learning_rate', 1e-8, 1.0, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'max_depth': trial.suggest_int('max_depth', 3, 9),
            'random_state': 42,
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'enable_categorical': True
        }

        model = xgboost.XGBRegressor(**param)

        model.fit(x_train, y_train)

        preds = model.predict(x_test)

        mae = mean_absolute_error(y_test, preds)

        return mae

    study = optuna.create_study(direction='minimize', study_name='Diamonds XGBoost')
    study.optimize(objective, n_trials=SEARCH_BUDGET)

    return study.best_params
