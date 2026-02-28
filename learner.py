import logging
from typing import Any, Dict, List, Tuple, Union

import torch
import torch.nn as nn
import torchmetrics
import pytorch_lightning as pl
import argparse

from src.utils import denormalise, LinearScheduler, WeightDecay

logging = logging.getLogger('pytorch_lightning')

class OnlineLearner(pl.LightningModule):
    def __init__(self,
                 controller: nn.Module,
                 regression_loss: nn.Module,
                 dataset: Any,
                 learning_rate: float = 1e-3,
                 weight_decay_start_weight: float = 1e-4,
                 weight_decay_end_weight: float = 1e-4,
                 weight_decay_start_time_step: int = 0,
                 weight_decay_end_time_step: int = 1e8,
                 swa_gamma: float = 0.99,
                 swa_replace_frequency: int = 5,
                 swa_start_time_step: int = 0,
                 swa_end_time_step: int = 1e8,
                 mse_start_weight: float = 1.0,
                 mse_exp_decay: float = 0.8,
                 mse_end_weight: float = 1.0,
                 mse_start_time_step: int = 0,
                 mse_end_time_step: int = 1e8):
        super(OnlineLearner, self).__init__()
        self.save_hyperparameters()

        self._controller = controller
        self._regression_loss = regression_loss

        self._input_features = dataset.input_features
        self._indices_input_features = dataset.indices_input_features
        self._output_features = dataset.output_features
        self._indices_output_features = dataset.indices_output_features

        self._scaling_factors_min = dataset.scaling_factors_min
        self._scaling_factors_max = dataset.scaling_factors_max
        self._prediction_horizon = dataset.prediction_horizon

        self._create_metrics()
        self._create_prediction_containers()

        self.time_step = 0

        self._mse_weight_scheduler = LinearScheduler(
            start_value=self.hparams.mse_start_weight,
            end_value=self.hparams.mse_end_weight,
            start_time_step=self.hparams.mse_start_time_step,
            end_time_step=self.hparams.mse_end_time_step,
        )

        self._weight_decay_scheduler = LinearScheduler(
            start_value=self.hparams.weight_decay_start_weight,
            end_value=self.hparams.weight_decay_end_weight,
            start_time_step=self.hparams.weight_decay_start_time_step,
            end_time_step=self.hparams.weight_decay_end_time_step,
        )
        self._weight_decay = WeightDecay()

        self._swa_model_copy = [p.clone().detach().to(p.device)
                        for p in self._controller.parameters()]

        self.training_losses=[]

    def _create_metrics(self) -> None:
        self.metrics = {
            "train": nn.ModuleDict({
                "loss_mse": torchmetrics.MeanMetric(),
                "loss_weight_decay": torchmetrics.MeanMetric(),
                "loss": torchmetrics.MeanMetric(),
            }),
            "val": nn.ModuleDict({
                "loss_mse": torchmetrics.MeanMetric(),
            }),
            "test": nn.ModuleDict({
                "loss_mse": torchmetrics.MeanMetric(),
            }),
        }

    def _create_prediction_containers(self) -> None:
        self.predictions = {}
        self.targets = {}
        for data_split in ["train", "val", "test"]:
            self.predictions[data_split] = torch.empty(
                (0, self._prediction_horizon, len(self._output_features)))
            self.targets[data_split] = torch.empty(
                (0, self._prediction_horizon, len(self._output_features)))

    def _append_prediction_containers(self, data_split: str,
                                      predictions: torch.Tensor,
                                      targets: torch.Tensor) -> None:
        """Appends the predictions and targets to their respective containers."""
        self.predictions[data_split] = torch.cat(
            (self.predictions[data_split], predictions.cpu()), dim=0)
        self.targets[data_split] = torch.cat(
            (self.targets[data_split], targets.cpu()), dim=0)
        

    def _update_swa_model_copy(self) -> None:
        """
        Update the stochastic weight averaging (SWA) copy of the model parameters.
        This function maintains an exponential moving average of the model weights
        using the gamma factor defined in the hyperparameters.

        It ensures that both parameter sets (model and SWA copy) reside on the same device
        without performing unnecessary device transfers at each step.
        """
        # Ensure all SWA parameters are on the same device as the model
        device = next(self._controller.parameters()).device
        self._swa_model_copy = [p.to(device) for p in self._swa_model_copy]

        # Exponential moving average update
        for swa_param, param in zip(self._swa_model_copy, self._controller.parameters()):
            swa_param.data = (
                self.hparams.swa_gamma * swa_param.data +
                (1.0 - self.hparams.swa_gamma) * param.data)
        

    #def _update_swa_model_copy(self) -> None:
       # for swa_param, param in zip(self._swa_model_copy, self._controller.parameters()):
            # assicurati che la copia sia sullo stesso device del parametro attuale
           # if swa_param.device != param.device:
               # swa_param.data = swa_param.data.to(param.device)
           # swa_param.data = (self.hparams.swa_gamma * swa_param.data + (1.0 - self.hparams.swa_gamma) * param.data)        


    #def _update_swa_model_copy(self) -> None:
        #for swa_param, param in zip(self._swa_model_copy, self._controller.parameters()):
           # swa_param.data = self.hparams.swa_gamma * swa_param.data + \
              #  (1.0 - self.hparams.swa_gamma) * param.data

    def _replace_current_model_with_swa_model_copy(self) -> None:
        for swa_param, param in zip(self._swa_model_copy, self._controller.parameters()):
            param.data = swa_param.data.clone()

    def _split_and_squeeze(self, pair: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        inputs, targets = pair
        inputs, targets = inputs.squeeze(0), targets.squeeze(0)
        return inputs, targets

    def _features_selection(self, batch: Tuple[Tuple[torch.Tensor, torch.Tensor],
                                               Tuple[torch.Tensor, torch.Tensor],
                                               Tuple[torch.Tensor, torch.Tensor]]
                            ) -> Tuple[Tuple[torch.Tensor, torch.Tensor],
                                       Tuple[torch.Tensor, torch.Tensor],
                                       Tuple[torch.Tensor, torch.Tensor]]:
        inputs_training, targets_training = self._split_and_squeeze(batch[0])
        inputs_validation, targets_validation = self._split_and_squeeze(batch[1])
        inputs_test, targets_test = self._split_and_squeeze(batch[2])

        batch_out = []
        for inputs, targets in zip(
                [inputs_training, inputs_validation, inputs_test],
                [targets_training, targets_validation, targets_test]
            ):
            inputs = inputs[:, :, self._indices_input_features]
            targets = targets[:, :, self._indices_output_features]
            batch_out.append((inputs, targets))

        return batch_out

    def training_step(self, batch: Tuple[Tuple[torch.Tensor, torch.Tensor],
                                         Tuple[torch.Tensor, torch.Tensor],
                                         Tuple[torch.Tensor, torch.Tensor]],
                       batch_idx: int) -> torch.Tensor:
        """Perform a single training, validation, and testing step."""
        super(OnlineLearner, self).training_step(batch, batch_idx)
        self._controller.train()

        (inputs_training, targets_training), \
            (inputs_validation, targets_validation), \
            (inputs_test, targets_test) = self._features_selection(batch)

        # forward pass: training
        training_predictions = self._controller(inputs=inputs_training)
        regression_loss_training = self._regression_loss(
            training_predictions, targets_training
        )

        gradient_loss = regression_loss_training * self._mse_weight_scheduler.get_value()

        # weight decay
        weight_decay = self._weight_decay(self._controller)
        gradient_loss += weight_decay * self._weight_decay_scheduler.get_value()

        # metrics
        self.metrics["train"]["loss_mse"].update(regression_loss_training.item())
        self.metrics["train"]["loss_weight_decay"].update(weight_decay.item())
        self.metrics["train"]["loss"].update(gradient_loss.item())

        # store predictions
        self._append_prediction_containers("train", training_predictions, targets_training)

        # validation/test step without gradient
        with torch.no_grad():
            self._controller.eval()

            valid_predictions = self._controller(inputs=inputs_validation)
            regression_loss_validation = self._regression_loss(
                valid_predictions, targets_validation
            )
            self.metrics["val"]["loss_mse"].update(regression_loss_validation.item())
            self._append_prediction_containers("val", valid_predictions, targets_validation)

            test_predictions = self._controller(inputs=inputs_test)
            regression_loss_test = self._regression_loss(
                test_predictions, targets_test
            )
            self.metrics["test"]["loss_mse"].update(regression_loss_test.item())
            self._append_prediction_containers("test", test_predictions, targets_test)

        self.time_step += 1

        # Log metrics
        self._log_metrics("train", self.metrics["train"])
        self._log_metrics("val", self.metrics["val"])
        self._log_metrics("test", self.metrics["test"])

        # keep track of training loss
        self.training_losses.append(regression_loss_training.item())

        return gradient_loss

    def on_train_batch_end(self, outputs: Dict[str, Any], batch: Any, batch_idx: int) -> None:
        # SWA updates
        self._update_swa_model_copy()
        if (self.time_step % self.hparams.swa_replace_frequency == 0
                and self.time_step > 0
                and self.hparams.swa_replace_frequency > 0
                and self.time_step >= self.hparams.swa_start_time_step
                and self.time_step <= self.hparams.swa_end_time_step):
            self._replace_current_model_with_swa_model_copy()

        self.log("mse_weight", self._mse_weight_scheduler.get_value(), on_step=True, on_epoch=False)
        self.log("weight_decay", self._weight_decay_scheduler.get_value(), on_step=True, on_epoch=False)

        self._mse_weight_scheduler.step()
        self._weight_decay_scheduler.step()

        super().on_train_batch_end(outputs, batch, batch_idx)

    def _log_metrics(self, prefix: str, metrics: nn.ModuleDict) -> None:
        for metric_name, metric in metrics.items():
            self.log(f"{prefix}/{metric_name}", metric.compute(), on_step=True, on_epoch=False)

    def _reset_metrics(self, prefix: str) -> None:
        if self.current_epoch < self.trainer.max_epochs - 1:
            for metric in self.metrics[prefix].values():
                metric.reset()

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(
            self._controller.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=0.0
        )

    @staticmethod
    def add_specific_args(parent_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate.")
        parser.add_argument("--weight_decay_start_weight", type=float, default=0.0001, help="Initial weight decay.")
        parser.add_argument("--weight_decay_end_weight", type=float, default=0.0001, help="Final weight decay.")
        parser.add_argument("--weight_decay_start_time_step", type=int, default=0, help="Weight decay start time step.")
        parser.add_argument("--weight_decay_end_time_step", type=int, default=int(0), help="Weight decay end time step.")
        parser.add_argument("--swa_gamma", type=float, default=0.5,
                            help="SWA gamma factor for moving average.")
        parser.add_argument("--swa_start_time_step", type=int, default=0,
                            help="SWA start time step.")
        parser.add_argument("--swa_end_time_step", type=int, default=int(0),
                            help="SWA end time step.")
        parser.add_argument("--swa_replace_frequency", type=int, default=5,
                            help="How often to replace params with SWA copy.")
        parser.add_argument("--mse_start_weight", type=float, default=1.0,
                            help="Starting weight for MSE loss.")
        parser.add_argument("--mse_exp_decay", type=float, default=0.8,
                            help="Exponential decay rate for MSE weight.")
        parser.add_argument("--mse_end_weight", type=float, default=1.0,
                            help="Final weight for MSE loss.")
        parser.add_argument("--mse_start_time_step", type=int, default=0,
                            help="Time step at which MSE weighting starts.")
        parser.add_argument("--mse_end_time_step", type=int, default=int(0),
                            help="Time step at which MSE weighting ends.")
        return parser
