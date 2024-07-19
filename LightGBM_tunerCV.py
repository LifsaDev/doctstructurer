import json
import optuna.integration.lightgbm as LightGBM

from lightgbm import early_stopping
from lightgbm import log_evaluation
import sklearn.datasets
from sklearn.model_selection import KFold


def run_hpo_tune():
    feauture, target = sklearn.datasets.load_wine(return_X_y=True)
    data = LightGBM.Dataset(feauture, label=target)

    params = {
        "objective": "binary",
        "metric": "binary_error", 
        "verbosity": -1,
        "boosting_type": "gbdt",
    }

    tuner = LightGBM.LightGBMTunerCV(
        params,
        data,
        folds=KFold(n_splits=5, shuffle=True),
        callbacks=[early_stopping(100), log_evaluation(100)],
    )

    tuner.run()
    
    best_params = tuner.best_params
    best_score =  1-tuner.best_score 
    n_trials = len(tuner.study.trials)
    tune_result = {"n_trials": n_trials,"best_score": best_score,"payload": {"algorithm": "STEP WISE","best_params": best_params}}
    
    with open("tune_LightGBM_result.json", 'w') as file:
        json.dump(tune_result, file)




if __name__ == "__main__":
    run_hpo_tune()

