from argparse import ArgumentParser
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from src.utils import denormalise
import os
import logging

import src.utils as utils
import src.plots as plots
from src.training_and_evaluation.loss import MSELoss
from src.models.controller import Controller
from src.dataloader.csv_dataset import CSVDataset
from src.training_and_evaluation.learner import OnlineLearner
from syne_tune import Reporter

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import torch
torch.set_float32_matmul_precision("high")  # Usa Tensor Cores in modo ottimale

# ✅ Logger corretto
logger = logging.getLogger('pytorch_lightning')


def main(args: ArgumentParser) -> None:
    # Set seed
    pl.seed_everything(args.seed)
    
    # Create experiment structure
    if args.experiment_name is None:
        experiment_name = utils.get_current_time()
    else:
        experiment_name = args.experiment_name
    experiment_path = os.path.join(args.save_dir, experiment_name) 
    experiment_path = utils.create_experiment_folder(experiment_path, "./src")

    # Set the logger
    utils.config_logger(experiment_path)
    logger.info("Beginning experiment: %s", experiment_name)
    logger.info("Arguments: %s", args)

    # Save the arguments
    utils.save_pickle(args, utils.args_file(experiment_path))

    # Load data
    print(f"Using site_id: {args.site_id}")
    dataset = CSVDataset(site_id=args.site_id,
                         lookback_window=args.lookback_window,
                         prediction_horizon=args.prediction_horizon,
                         train_batchsize=args.train_batchsize,
                         valid_batchsize=args.valid_batchsize,
                         input_features=args.input_features,
                         output_features=args.output_features,
                         data_dir=args.data_dir)
    if args.load_dir is not None:
        logger.info("Loading dataset scaling constants from %s", args.load_dir)
        utils.load_dataset(dataset, utils.dataset_file(args.load_dir))
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
    
    # Create the controller
    controller = Controller(dataset=dataset,
                            hidden_dim=args.hidden_dim,
                            num_layers=args.num_layers)
    if args.load_dir is not None:
        logger.info("Loading model from %s", args.load_dir)
        controller = utils.load_model(controller, utils.model_file(args.load_dir))

    # Create loss
    regression_loss = MSELoss()

    # Create trainer
    tb_logger = pl.loggers.TensorBoardLogger(
        save_dir=experiment_path, name="logs") if args.st_checkpoint_dir is None else None

    trainer = pl.Trainer(accelerator="gpu", devices=[args.gpu], max_epochs=1, enable_checkpointing=False,
                         logger=tb_logger, enable_progress_bar=True if args.st_checkpoint_dir is None else False)

    # Create Learner
    learner = OnlineLearner(controller=controller,
                            regression_loss=regression_loss,
                            dataset=dataset,
                            learning_rate=args.learning_rate,
                            weight_decay_start_weight=args.weight_decay_start_weight,
                            weight_decay_end_weight=args.weight_decay_start_weight,
                            weight_decay_start_time_step=args.weight_decay_start_time_step, 
                            weight_decay_end_time_step=args.weight_decay_end_time_step,
                            swa_gamma=args.swa_gamma,
                            swa_start_time_step=args.swa_start_time_step, 
                            swa_end_time_step=args.swa_end_time_step, 
                            swa_replace_frequency=args.swa_replace_frequency,
                            mse_start_weight=args.mse_start_weight,
                            mse_exp_decay=args.mse_exp_decay,
                            mse_end_weight=args.mse_end_weight,
                            mse_start_time_step=args.mse_start_time_step, 
                            mse_end_time_step=args.mse_end_time_step,
                            )

    # Fit the model
    trainer.fit(learner, train_dataloaders=dataloader)

    # Plot results
    results = {}
    if args.plot:
        data_splits = ['train', 'val', 'test']
    else:
        data_splits = ['test']

    for data_split in data_splits:
        results[data_split] = {}
        plots.plot_predictions(experiment_path=experiment_path,
                               dataset=dataset,
                               predictions=learner.predictions[data_split],
                               targets=learner.targets[data_split],
                               output_features=args.output_features,
                               t_initial=0, t_final=learner.predictions[data_split].shape[0],
                               data_split=data_split)
    
    if args.plot:
        plots.plot_training_loss(experiment_path=experiment_path, training_losses=learner.training_losses)
        logger.info(f"Training loss plot saved at: {os.path.join(experiment_path, 'training_loss', 'training_loss_plot.png')}")

    # Salvataggio CSV predizioni (formato semplificato)
    import pandas as pd
    import numpy as np
    
    try:
        # Tensori completi (T, horizon, 1)
        preds_full = learner.predictions["test"].detach().cpu()
        trues_full = learner.targets["test"].detach().cpu()

        # Denormalizzazione come in plots.py
        scaling_factors_min = dataset.scaling_factors_min[dataset.indices_output_features]
        scaling_factors_max = dataset.scaling_factors_max[dataset.indices_output_features]

        preds_full = denormalise(preds_full, scaling_factors_min, scaling_factors_max).numpy()
        trues_full = denormalise(trues_full, scaling_factors_min, scaling_factors_max).numpy()

        T, H, _ = preds_full.shape

        rows = []

        scenario = os.path.basename(args.data_dir).replace("csv-", "")

        for t in range(T):
            for h in range(H):
                rows.append({
                    "site_id": args.site_id,
                    "scenario": scenario,
                    "prediction_horizon": args.prediction_horizon * 5,  # opzionale, se vuoi minuti reali
                    "timestep": t,
                    "horizon_step": h,
                    "True_BloodGlucose": float(trues_full[t, h, 0]),
                    "Predicted_BloodGlucose": float(preds_full[t, h, 0]),
                })

        df = pd.DataFrame(rows)

        pred_filename = f"predictions_full_site_{args.site_id:03d}.csv"
        pred_path = os.path.join(experiment_path, pred_filename)
        df.to_csv(pred_path, index=False)

        logging.info(f"✅ CSV completo salvato in: {pred_path}")

    except Exception as e:
        logging.warning(f"⚠️ Errore durante il salvataggio delle predizioni complete: {e}")

    # Store results
    utils.store_metrics(results, metrics=learner.metrics["train"], prefix="train")
    utils.store_metrics(results, metrics=learner.metrics["val"], prefix="val")
    utils.store_metrics(results, metrics=learner.metrics["test"], prefix="test")
    logger.info("Results: %s", results)

    # Save model, results, learner, and dataset
    utils.save_model(controller, utils.model_file(experiment_path))
    utils.save_pickle(results, utils.results_file(experiment_path))
    utils.save_learner(learner, utils.learner_file(experiment_path))
    utils.save_learner_as_csv(learner, experiment_path)
    utils.save_dataset(dataset, utils.dataset_file(experiment_path))


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument_group("Data")
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='The directory where the data is stored.')
    parser.add_argument('--num_workers', type=int, default=80,
                        help='The number of workers to be used for loading the data.')
    parser.add_argument('--st_checkpoint_dir', type=str, default=None,
                        help='Directory where syne-tune checkpoints are stored.')

    parser = CSVDataset.add_specific_args(parser)
    
    parser.add_argument_group("Network")
    parser = Controller.add_specific_args(parser)

    parser.add_argument_group("Experiment")
    parser.add_argument('--seed', type=int, default=42,
                        help='The seed to be used for training.')
    parser.add_argument('--gpu', type=int, default=0,
                        help='The gpu to be used for training.')
    parser.add_argument('--save_dir', type=str, default='experiments',
                        help='The directory where the experiment results are stored.')
    parser.add_argument('--load_dir', type=str, default=None,
                        help='The directory where the model is loaded from.')
    parser.add_argument('--plot', type=int, choices=[0, 1], default=1,
                        help='Enable the plot of the train and validation results.')
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Name of the experiment directory.')

    parser = OnlineLearner.add_specific_args(parser)
    args = parser.parse_args()

    main(args)