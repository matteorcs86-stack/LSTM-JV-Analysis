import torch
import torch.nn as nn
import argparse
#from cvxpy.expressions.variable import Variable
#from cvxpy.expressions.constants.parameter import Parameter
#from cvxpy.problems.problem import Problem
#from cvxpy.problems.objective import Minimize
#from cvxpylayers.torch import CvxpyLayer

from src.dataloader.csv_dataset import CSVDataset
from src.utils import denormalise

import logging
logging = logging.getLogger('pytorch_lightning')


class Controller(nn.Module):
    def __init__(self, dataset, hidden_dim: int, num_layers: int) -> None:
        super(Controller, self).__init__()
        self._hidden_dim = hidden_dim
        self._num_layers = num_layers

        self._lookback_window = dataset.lookback_window
        self._prediction_horizon = dataset.prediction_horizon
        self._input_features = dataset.input_features
        self._output_features = dataset.output_features

        # Find the index of "BloodGlucose" in input features
        if 'BloodGlucose' not in self._input_features:
            raise ValueError("BloodGlucose must be in input_features.")
        self._bg_idx = self._input_features.index('BloodGlucose')

        self._input_dim = len(self._input_features)
        self._output_dim = len(self._output_features)

        self._lstm = nn.LSTM(
            input_size=self._input_dim,
            hidden_size=self._hidden_dim,
            num_layers=self._num_layers,
            batch_first=True,
            bias=True
        )

        # The linear layer is only used for single-step prediction: hidden_dim -> output_dim
        self._fc = nn.Linear(self._hidden_dim, self._output_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for recursive prediction.

        Args:
            inputs (torch.Tensor): The input tensor of shape [batch_size, lookback_window + prediction_horizon, input_dim].

        Returns:
            torch.Tensor: The predicted tensor of shape [batch_size, prediction_horizon, output_dim].
        """
        
        # ============ 0) Basic validation ============
        batch_size, total_seq_len, in_dim = inputs.size()
        expected_len = self._lookback_window + self._prediction_horizon
        if total_seq_len != expected_len:
            raise ValueError(f"Expected sequence length {expected_len}, got {total_seq_len}.")
        if in_dim != self._input_dim:
            raise ValueError(f"Expected input_dim {self._input_dim}, got {in_dim}.")

        # ============ 1) Initialize LSTM hidden states using lookback window ============
        lookback_seq = inputs[:, :self._lookback_window, :]  # shape: [batch, lookback_window, input_dim]
        lstm_out, (hidden_h, hidden_c) = self._lstm(lookback_seq)
        
        # Prepare a list to collect predictions for each step
        all_preds = []

        # ============ 2) Process future time steps one by one ============
        # `last_bg` keeps the predicted BG value from the previous step
        last_bg = None  

        for step in range(self._prediction_horizon):
            # Extract the frame to predict for this step: [batch, 1, input_dim]
            # Use clone() to avoid modifying the original `inputs` tensor in-place
            one_step_input = inputs[:, self._lookback_window + step : self._lookback_window + step + 1, :].clone()
            
            if step > 0:
                # Replace the BG value in the current frame with the last predicted BG
                # Modify only the cloned `one_step_input`, keeping the original safe
                one_step_input[:, 0, self._bg_idx] = last_bg  

            # Forward pass through the LSTM
            lstm_out, (hidden_h, hidden_c) = self._lstm(one_step_input, (hidden_h, hidden_c))

            # Linear layer to map the hidden state to output dimension (batch_size, output_dim)
            step_prediction = self._fc(lstm_out[:, -1, :])

            # Collect the prediction for this step
            all_preds.append(step_prediction)

            # Extract the predicted BG value (assuming output_features[0] corresponds to BG)
            # Use it for the next step's prediction
            last_bg = step_prediction[:, 0]

        # Stack all step predictions into a single tensor of shape [batch_size, prediction_horizon, output_dim]
        all_preds = torch.stack(all_preds, dim=1)
        return all_preds

    @staticmethod
    def add_specific_args(parent_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=False)
        parser.add_argument("--hidden_dim", type=int, default=16,
                            help="Hidden dimension of the LSTM layers.")
        parser.add_argument("--num_layers", type=int, default=1,
                            help="Number of LSTM layers.")
        return parser
