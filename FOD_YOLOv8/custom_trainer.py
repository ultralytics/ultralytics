from ultralytics import YOLO
import yaml
import math
import os
import shutil
from FOD_YOLOv8.hyperparameter import Hyperparameters
import logging

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


class CustomTrainer:
    def __init__(self, data_path: str, hyps_path: str, model_path: str, evolve: bool, imgsz: int, batch: int, optimizer: str, epochs: int, device: str, trials: int, save: bool, study_name: str, resume: bool, resume_model: str) -> None:
        self._evolve = evolve
        self._data_path = data_path
        self._hyps_path = hyps_path
        self._trial_count = 0
        self._img = imgsz
        self._optim = optimizer
        self._epochs = epochs
        self._device = device
        self._trials = trials
        self._batch = batch
        self._save = save
        self._study_name = study_name
        self._resume = resume
        self._model = YOLO(model_path)
        self._resume_model = YOLO(model_path) if not self._resume else YOLO(resume_model)
        if self._resume:
            if not self._evolve:
                self._train_with_resume()
            else:
                self._optuna_train()
        elif self._evolve:
            self._optuna_train()
        else:
            self._train()

    def _optuna_train(self):
        import optuna
        self._optuna = optuna
        self._train_with_evolve()

    def _create_hyps(self, trial):
        # logger.info("Trial")
        hypsObject = Hyperparameters(self._hyps_path)
        hyps = hypsObject.get_hyps()
        vals = hypsObject.hyp_ranges_dict
        if self._trial_count>0:
            for l in vals:
                vals3 = vals[l]
                if l=="lr0":
                    hyps[l] = 0.01 if self._optim == 'SGD' else 0.001
                else:
                    if vals3[0]:
                        hyps[l] = trial.suggest_float(l, vals3[2], vals3[3])
                    else:
                        hyps[l] = trial.suggest_int(l, vals3[2], vals3[3])
            hypsObject.saveHyps(hyps)
        else:
            for l in vals:
                vals3 = vals[l]
                if l=="lr0":
                    hyps[l] = 0.01 if self._optim == 'SGD' else 0.001
                else:
                    if vals3[0]:
                        hyps[l] = trial.suggest_float(l, vals3[1], vals3[1])
                    else:
                        hyps[l] = trial.suggest_int(l, vals3[1], vals3[1])
            hypsObject.saveHyps(hyps)
        return hypsObject.hyp_dict

    def objective(self, trial):
        if self._resume:
            self._first_trial_in_resume = False
            self._train_with_resume()
            self._resume=False
        else:
            hyp_dict = self._create_hyps(trial)
            self._model.train(epochs=self._epochs, imgsz=self._img, device=self._device, data=self._data_path, optimizer=self._optim, batch=self._batch, **hyp_dict, save=self._save)
        metrics = self._model.val()
        # 0.5*mAP50-95 + 0.35*mAP75 + 0.15*mAP50
        ws = [0.5, 0.35, 0.15]
        maps = [metrics.box.map, metrics.box.map75, metrics.box.map50]
        self._trial_count+=1
        aggregate_map = sum([i[0]*i[1] for i in zip(maps, ws)])
        return aggregate_map
    
    def _train_with_evolve(self) -> None:
        storage=f"sqlite:///{self._study_name}.db"
        study = self._optuna.create_study(study_name=self._study_name, storage=storage, load_if_exists=True, direction='maximize')
        study.optimize(self.objective, n_trials=self._trials)
        self._plot(study)
        Hyperparameters(self._optuna_path.replace('.jpeg', '.yaml'), default=study.best_params)
    
    def _train(self) -> None:
        hypsObject = Hyperparameters(self._hyps_path)
        self._model.train(epochs=self._epochs, imgsz=self._img, device=self._device, data=self._data_path, optimizer=self._optim, batch=self._batch, **hypsObject.hyp_dict, save=self._save)

    def _plot(self, study):
        fig = self._optuna.visualization.plot_optimization_history(study)
        if not os.path.isdir('runs/detect/optuna'):
            os.makedirs('runs/detect/optuna', exist_ok=True)
        path = os.path.join('runs/detect/optuna')
        if os.path.isdir(os.path.join(path, self._study_name)):
            shutil.rmtree(os.path.join(path, self._study_name), ignore_errors=True)
        os.makedirs(os.path.join(path, self._study_name), exist_ok=True)
        self._optuna_path = os.path.join(path, self._study_name, 'Optimizer_Output.jpeg')
        fig.write_image(self._optuna_path)
        print(f"Optimization history plot stored in {self._optuna_path}")

    def _train_with_resume(self):
        self._resume_model.train(save=self._save, resume=self._resume)
