from src.utils import normalise
from typing import Tuple, List, Any, Dict
from torch.utils.data import Dataset
import pandas as pd
import torch
import argparse
import logging
import os

logging = logging.getLogger('pytorch_lightning')


class CSVDataset(Dataset):
    """
    Dataset per la previsione del glucosio basato su file CSV.
    Compatibile con dataset disposti in sottocartelle tipo 'site_1/dataset.csv'.
    """

    def __init__(self,
                 site_id: int,
                 lookback_window: int,
                 prediction_horizon: int,
                 train_batchsize: int,
                 valid_batchsize: int,
                 input_features: List = ['BloodGlucose', 'basal', 'bolus', 'CHO'],
                 output_features: List = ['BloodGlucose'],
                 data_dir: str = "data"
                 ) -> None:
        
        self.site_id = site_id   # <<--- per inizializzare il site_id e usarlo per salvare i file .mat con identificativo (in plots.py)

        # --- Gestione flessibile percorso file ---
        candidate_path = os.path.join(data_dir, f"site_{site_id}", "dataset.csv")
        alt_path = os.path.join(data_dir, "dataset.csv")

        if os.path.exists(candidate_path):
            file_path = candidate_path
        elif os.path.exists(alt_path):
            file_path = alt_path
        else:
            raise FileNotFoundError(
                f"Dataset non trovato per site_id={site_id}.\n"
                f"Percorsi tentati:\n - {candidate_path}\n - {alt_path}"
            )

        print(f"[DEBUG] Caricamento dataset per site_id={site_id}")
        print(f"[DEBUG] data_dir = {data_dir}")
        print(f"[DEBUG] file_path selezionato = {file_path}")
        print(f"[DEBUG] Esiste il file? {os.path.exists(file_path)}")

        # --- Caricamento CSV ---
        self._data = pd.read_csv(file_path)
        self.available_features = list(self._data.columns.values)

        # Verifica feature
        for f in input_features:
            if f not in self.available_features:
                raise ValueError(f"Input feature {f} non presente nel dataset.")
        for f in output_features:
            if f not in self.available_features:
                raise ValueError(f"Output feature {f} non presente nel dataset.")

        self.lookback_window = lookback_window
        self.prediction_horizon = prediction_horizon
        self.train_batchsize = train_batchsize
        self.valid_batchsize = valid_batchsize
        self.input_features = input_features
        self.output_features = output_features
        self.bypass_features = [f for f in self.available_features if f not in self.output_features]

        # Indici
        self.indices_input_features = [i for i, f in enumerate(self.available_features) if f in self.input_features]
        self.indices_output_features = [i for i, f in enumerate(self.available_features) if f in self.output_features]
        self.indices_bypass_features = [i for i in range(len(self.available_features))
                                        if i not in self.indices_output_features]

        self.scaling_factors_min: torch.Tensor = None
        self.scaling_factors_max: torch.Tensor = None
        self._preprocess_and_split_data()

    def state_dict(self) -> Dict[str, Any]:
        return {
            'scaling_factors_min': self.scaling_factors_min,
            'scaling_factors_max': self.scaling_factors_max,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.scaling_factors_min = state_dict['scaling_factors_min']
        self.scaling_factors_max = state_dict['scaling_factors_max']
        self._preprocess_and_split_data()

    # ===============================================================
    # >>> SEZIONE CORRETTA PER IL PREPROCESSING E SPLITTING <<<
    # ===============================================================
    def _preprocess_and_split_data(self) -> None:
        data = self._data[self.available_features].values
        data = torch.from_numpy(data).type(torch.float32)

        # Normalizzazione
        self.scaling_factors_min = data.min(dim=0).values if self.scaling_factors_min is None else self.scaling_factors_min
        self.scaling_factors_max = data.max(dim=0).values if self.scaling_factors_max is None else self.scaling_factors_max
        data_normalised = normalise(data, self.scaling_factors_min, self.scaling_factors_max)

        L = self.lookback_window + self.prediction_horizon  # finestra totale

        # Container
        self._inputs_train, self._targets_train = [], []
        self._inputs_validation, self._targets_validation = [], []
        self._inputs_test, self._targets_test = [], []

        # Indici globali
        self.t_i = self.train_batchsize + self.valid_batchsize + self.lookback_window + self.prediction_horizon - 1
        self.t_total = len(data) - self.prediction_horizon + 1

        def make_extended_input(t_eff: int) -> torch.Tensor:
            x = torch.zeros(L, len(self.input_features))
            for j, feat in enumerate(self.input_features):
                col = self.indices_input_features[j]
                if feat in ['basal', 'bolus', 'CHO']:
                    start = t_eff - self.lookback_window
                    end = start + L
                    if end > len(data_normalised):
                        end = len(data_normalised)
                        start = end - L
                    x[:, j] = data_normalised[start:end, col]
                elif feat == 'BloodGlucose':
                    start_past = t_eff - self.lookback_window
                    end_past = t_eff
                    bg_extended = torch.zeros(L)
                    bg_extended[:self.lookback_window] = data_normalised[start_past:end_past, col]
                    last_val = data_normalised[t_eff - 1, col]
                    bg_extended[self.lookback_window:] = last_val
                    x[:, j] = bg_extended
            return x

        def make_target(t_eff: int) -> torch.Tensor:
            y = torch.zeros(self.prediction_horizon, len(self.available_features))
            for j in range(len(self.available_features)):
                y[:, j] = data_normalised[t_eff:t_eff + self.prediction_horizon, j]
            return y

        for t in range(self.t_i, self.t_total):
            # TEST
            t_eff = t
            x_test = make_extended_input(t_eff).unsqueeze(0)
            y_test = make_target(t_eff).unsqueeze(0)
            self._inputs_test.append(x_test)
            self._targets_test.append(y_test)

            # VALIDATION
            x_val = torch.zeros(self.valid_batchsize, L, len(self.input_features))
            y_val = torch.zeros(self.valid_batchsize, self.prediction_horizon, len(self.available_features))
            for k in range(self.valid_batchsize):
                t_eff_val = t - (self.prediction_horizon + k)
                x_val[k] = make_extended_input(t_eff_val)
                y_val[k] = make_target(t_eff_val)
            self._inputs_validation.append(x_val)
            self._targets_validation.append(y_val)

            # TRAIN
            x_tr = torch.zeros(self.train_batchsize, L, len(self.input_features))
            y_tr = torch.zeros(self.train_batchsize, self.prediction_horizon, len(self.available_features))
            for k in range(self.train_batchsize):
                t_eff_tr = t - (self.valid_batchsize + self.prediction_horizon + k)
                x_tr[k] = make_extended_input(t_eff_tr)
                y_tr[k] = make_target(t_eff_tr)
            self._inputs_train.append(x_tr)
            self._targets_train.append(y_tr)

        # Stack finale
        self._inputs_train = torch.stack(self._inputs_train, dim=0)
        self._targets_train = torch.stack(self._targets_train, dim=0)
        self._inputs_validation = torch.stack(self._inputs_validation, dim=0)
        self._targets_validation = torch.stack(self._targets_validation, dim=0)
        self._inputs_test = torch.stack(self._inputs_test, dim=0)
        self._targets_test = torch.stack(self._targets_test, dim=0)

        # Verifica di sicurezza
        L_check = self.lookback_window + self.prediction_horizon
        assert self._inputs_train.shape[-2] == L_check
        assert self._inputs_validation.shape[-2] == L_check
        assert self._inputs_test.shape[-2] == L_check

    # ===============================================================

    def get_profile(self) -> torch.Tensor:
        data = self._data[self.available_features].values
        data = torch.from_numpy(data).type(torch.float32)
        profiles = data[self.t_i:self.t_total]
        return profiles

    def __len__(self) -> int:
        return len(self._inputs_train)

    def __getitem__(self, time: int) -> Tuple[
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor]]:
        inputs_train = self._inputs_train[time]
        targets_train = self._targets_train[time]
        inputs_val = self._inputs_validation[time]
        targets_val = self._targets_validation[time]
        inputs_test = self._inputs_test[time]
        targets_test = self._targets_test[time]
        return (inputs_train, targets_train), (inputs_val, targets_val), (inputs_test, targets_test)

    @staticmethod
    def add_specific_args(parent_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--site_id", type=int, default=1, help="ID del dataset (paziente).")
        parser.add_argument('--train_batchsize', type=int, default=1, help='Numero campioni per step di training.')
        parser.add_argument('--valid_batchsize', type=int, default=1, help='Numero campioni per step di validation/test.')
        parser.add_argument('--input_features', nargs='+', default=['BloodGlucose', 'basal', 'bolus', 'CHO'],
                            help='Feature di input.')
        parser.add_argument('--output_features', nargs='+', default=['BloodGlucose'], help='Feature target.')
        parser.add_argument('--lookback_window', type=int, default=60, help='Numero di passi passati.')
        parser.add_argument('--prediction_horizon', type=int, default=12, help='Numero di passi futuri da predire.')
        # Non riaggiungiamo --data_dir per evitare conflitti duplicati
        return parser