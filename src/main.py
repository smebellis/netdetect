import argparse
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Optional, Union

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


from config import config
from DataCleaner import DataCleaner
from DataLoader import DataLoader
from DataPreprocess import DataPreprocessor
from helper import split_data, parse_args, parse_kwargs, generate_plot
from LoggerSingleton import get_logger
from ModelTrainer import ModelTrainer
from ClassBalancer import ClassBalancer

# Configure logger
logger = get_logger(__name__)

random_state = config.DEFAULT_RANDOM_STATE


# Load the data from file
def main():

    try:
        args = parse_args()
    except Exception as e:
        logger.error(f"Error parsing arguments: {e}")
        return

    # Ensure the directories exist for saving
    logger.info("Creating directories for saving...")
    directories = config.DIRECTORIES
    for directory in directories.values():
        os.makedirs(directory, exist_ok=True)
    logger.info("Directories created or already exist.")

    # Initialize DataCleaner
    data_cleaner = DataCleaner(
        replace_inf=True,
        drop_na=True,
        drop_duplicates=True,
        subset_duplicates=None,  # or specify if needed
        inplace=False,
    )

    # Initialize DataLoader
    data_loader = DataLoader(
        pickle_path=args.data_path,
        csv_dir=args.csv_dir,
        data_cleaner=data_cleaner,
    )

    # Load data
    try:
        logger.info("Loading data...")
        df = data_loader.load_data()
        logger.info("Data loaded successfully.")
    except FileNotFoundError as fnf_error:
        logger.error(f"File not found: {fnf_error}")
        sys.exit(1)
    except pd.errors.EmptyDataError as ede_error:
        logger.error(f"No data: {ede_error}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading data: {e}")
        sys.exit(1)

    # Initialize DataPreprocessor
    data_preprocessor = DataPreprocessor(
        label_column=args.label_column,
        scale=args.scale,
        feature_selection=args.feature_selection,
        k_features=args.k_features,
    )

    # Preprocess the data
    try:
        logger.info("Preprocessing data...")
        df, le, scaler = data_preprocessor.preprocess(df)
        logger.info("Data preprocessing completed.")
    except KeyError as e:
        logger.error(f"Missing column during preprocessing: {e}")
        return
    except Exception as e:
        logger.error(f"Error during preprocessing: {e}")
        return

    # Split the data
    try:
        logger.info("Splitting data into training and testing sets...")
        X_train, X_test, y_train, y_test = split_data(
            df,
            label_column=args.label_column,
            test_size=config.DEFAULT_TEST_SIZE,
            random_state=config.DEFAULT_RANDOM_STATE,
        )
        logger.info(
            f"Data split completed. Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}"
        )
    except Exception as e:
        logger.error(f"Error splitting data: {e}")
        return

    # Initialize ClassBalancer if any balancing methods are specified
    if args.balance_methods:
        logger.info("Initializing ClassBalancer...")
        smote_kwargs = parse_kwargs(args.smote_kwargs)
        vae_config = {}
        if "vae" in args.balance_methods:
            # Ensure VAE configuration parameters are provided
            vae_config = {
                "input_dim": (
                    args.vae_input_dim if args.vae_input_dim else X_train.shape[1]
                ),
                "latent_dim": args.vae_latent_dim if args.vae_latent_dim else 10,
                "epochs": args.vae_epochs,
                "batch_size": args.vae_batch_size,
                "learning_rate": args.vae_learning_rate,
            }
        class_balancer = ClassBalancer(
            methods=args.balance_methods,
            random_state=config.DEFAULT_RANDOM_STATE,
            smote_kwargs=smote_kwargs,
            vae_config=vae_config,
        )

        # Apply balancing
        try:
            logger.info("Balancing classes...")
            X_balanced, y_balanced, class_weights = class_balancer.balance(
                X_train, y_train
            )
            logger.info("Class balancing completed.")
        except Exception as e:
            logger.error(f"Error during class balancing: {e}")
            sys.exit(1)
    else:
        X_balanced, y_balanced = X_train, y_train
        class_weights = None
        logger.info("No class balancing methods specified.")
    # Plot a count of classes
    # try:
    #     logger.info("Generating class distribution plot...")
    #     generate_plot(
    #         data=X_balanced,
    #         plot_type="count",
    #         x=y_balanced,
    #         title="Count of Classes",
    #         xlabel="Class",
    #         ylabel="Count",
    #     )
    #     logger.info("Class distribution plot generated.")
    # except Exception as e:
    #     logger.error(f"Error generating plot: {e}")

    # Initialize ModelTrainer
    model_trainer = ModelTrainer(
        config=config,
        directories=directories,
        scoring_metrics=config.SCORING_METRICS,
        random_state=config.DEFAULT_RANDOM_STATE,
    )

    # Iterate over models and train/evaluate
    for model_name, model in config.MODELS.items():
        model_trainer.train_and_evaluate(
            model_name=model_name,
            model=model,
            X_train=X_balanced,
            y_train=y_balanced,
            X_test=X_test,
            y_test=y_test,
            label_encoder=le,
        )

    # Save all metrics
    model_trainer.save_metrics()

    # Save training times
    model_trainer.save_training_times()


if __name__ == "__main__":
    main()
