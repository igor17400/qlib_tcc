
import qlib
import optuna
from qlib.constant import REG_US 
from qlib.utils import init_instance_by_config
from qlib.tests.data import GetData
from qlib.tests.config import get_dataset_config, BR_MARKET, BR_BENCH, DATASET_ALPHA360_CLASS
from qlib.log import get_module_logger

logger = get_module_logger("Hyperparameter")

market = "ibov"
benchmark = "^bvsp"

data_handler_config = {
    "start_time": "2008-01-01",
    "end_time": "2022-02-24",
    "fit_start_time": "2008-01-01",
    "fit_end_time": "2018-12-31",
    "instruments": market,
    "infer_processors": [
      {
        "class": "RobustZScoreNorm",
        "kwargs": {
          "fields_group": "feature",
          "clip_outlier": true
        }
      },
      {
        "class": "Fillna",
        "kwargs": {
          "fields_group": "feature"
        }
      }
    ],
    "learn_processors": [
      {
        "class": "DropnaLabel"
      },
      {
        "class": "CSRankNorm",
        "kwargs": {
          "fields_group": "label"
        }
      }
    ],
    "label": [
      "Ref($close, -2) / Ref($close, -1) - 1"
    ]
}

dataset_config = {
    "class": "DatasetH",
    "module_path": "qlib.data.dataset",
    "kwargs": {
        "handler": {
            "class": "Alpha360",
            "module_path": "qlib.contrib.data.handler",
            "kwargs": data_handler_config,
        },
        "segments": {
            "train": ("2008-01-01", "2018-12-31"),
            "valid": ("2019-01-01", "2020-12-31"),
            "test": ("2021-01-01", "2022-02-24"),
        },
    },
}

def objective(trial):
    task = {
        "model": {
            "class": "DNNModelPytorch",
            "module_path": "qlib.contrib.model.pytorch_nn",
            "kwargs": {
                "loss": "RMSE",
                "learning_rate": trial.suggest_uniform("eta", 1e-8, 1.0),
                "subsample": trial.suggest_uniform("subsample", 0, 1),
                "depth": trial.suggest_int("depth", 1, 12),
                "num_leaves": trial.suggest_int("num_leaves", 1, 1024),
                "thread_count": 20,
                "grow_policy": "Lossguide",
                "bootstrap_type": "Poisson",
            },
        },
    }

    evals_result = dict()
    model = init_instance_by_config(task["model"])
    evals_result = model.fit(dataset, evals_result=evals_result)
    return min(evals_result["valid"])


if __name__ == "__main__":
    logger.info("Qlib intialization")
    provider_uri = "~/igorlima/igor_tcc/qlib_data/br_data"
    GetData().qlib_data(target_dir=provider_uri, region=REG_US, exists_skip=True)
    qlib.init(provider_uri=provider_uri, region=REG_US)

    logger.info("Dataset intialization")
    dataset = init_instance_by_config(dataset_config)

    logger.info("Start parameter tuning")
    study = optuna.Study(study_name="MLP_360_br", storage="sqlite:///db_1_0.sqlite3")
    study.optimize(objective)
    
    trial = study.best_trial
    logger.info("--- Trial Results ---")
    logger.info('Accuracy: {}'.format(trial.value))
    logger.info("Best hyperparameters: {}".format(trial.params))
