# class_balancer.py

from typing import Optional, Union, List, Dict, Tuple
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.utils.class_weight import compute_class_weight

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from tqdm import tqdm

from LoggerSingleton import get_logger
from VAE import VAE

logger = get_logger(__name__)


# class_balancer.py

from typing import Optional, Union, List, Dict, Tuple
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.utils.class_weight import compute_class_weight

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from tqdm import tqdm

from LoggerSingleton import get_logger
from VAE import VAE, vae_loss

logger = get_logger(__name__)


class ClassBalancer:
    def __init__(
        self,
        methods: List[str],
        random_state: int = 42,
        smote_kwargs: Optional[Dict] = None,
        vae_config: Optional[Dict] = None,
        device: Optional[str] = None,
    ):
        """
        Initializes the ClassBalancer with specified balancing methods.

        Args:
            methods (List[str]): List of balancing methods to apply. Options: ['smote', 'weighted_loss', 'vae'].
            random_state (int, optional): Random state for reproducibility. Defaults to 42.
            smote_kwargs (dict, optional): Additional keyword arguments for SMOTE. Defaults to None.
            vae_config (dict, optional): Configuration dictionary for VAE. Should include 'input_dim', 'latent_dim', 'epochs', 'batch_size', 'learning_rate'. Defaults to None.
            device (str, optional): Device to run VAE on ('cpu' or 'cuda'). Defaults to None, which auto-selects.
        """
        self.methods = [m.lower() for m in methods]
        self.random_state = random_state
        self.smote_kwargs = smote_kwargs if smote_kwargs else {}
        self.vae_config = vae_config if vae_config else {}
        self.device = "cpu"
        # TODO:  uncomment this when the problem with VAE is figured out
        # self.device = (
        #     device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        # )
        self.logger = logger

        # Validate methods
        supported_methods = {"smote", "weighted_loss", "vae"}
        invalid_methods = set(self.methods) - supported_methods
        if invalid_methods:
            self.logger.error(
                f"Unsupported methods: {invalid_methods}. Supported methods are {supported_methods}."
            )
            raise ValueError(
                f"Unsupported methods: {invalid_methods}. Supported methods are {supported_methods}."
            )

        # Initialize SMOTE if selected
        if "smote" in self.methods:
            self._initialize_smote()

        # Initialize VAE if selected
        if "vae" in self.methods:
            self._initialize_vae()

    def _initialize_smote(self):
        """
        Initializes the SMOTE oversampler with provided keyword arguments.
        """
        try:
            self.smote = SMOTE(random_state=self.random_state, **self.smote_kwargs)
            self.logger.info("SMOTE initialized.")
        except Exception as e:
            self.logger.error(f"Failed to initialize SMOTE: {e}")
            raise

    def _initialize_vae(self):
        """
        Initializes the VAE model based on the provided configuration.
        """
        required_keys = [
            "input_dim",
            "latent_dim",
            "epochs",
            "batch_size",
            "learning_rate",
        ]
        if not all(key in self.vae_config for key in required_keys):
            self.logger.error(f"VAE configuration must include {required_keys}")
            raise ValueError(f"VAE configuration must include {required_keys}")

        self.input_dim = self.vae_config["input_dim"]
        self.latent_dim = self.vae_config["latent_dim"]
        self.epochs = self.vae_config["epochs"]
        self.batch_size = self.vae_config["batch_size"]
        self.learning_rate = self.vae_config["learning_rate"]

        try:
            self.vae = VAE(self.input_dim, self.latent_dim).to(self.device)
            self.optimizer = optim.Adam(self.vae.parameters(), lr=self.learning_rate)
            self.logger.info("VAE initialized.")
        except Exception as e:
            self.logger.error(f"Failed to initialize VAE: {e}")
            raise

    def balance(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series, Optional[Dict]]:
        """
        Applies the specified balancing methods to the dataset.

        Args:
            X (pd.DataFrame): Feature set.
            y (pd.Series): Labels.

        Returns:
            Tuple[pd.DataFrame, pd.Series, Optional[Dict]]: Balanced feature set and labels, and/or class weights.
        """
        X_balanced = X.copy()
        y_balanced = y.copy()
        class_weights = None

        # Apply SMOTE
        if "smote" in self.methods:
            X_balanced, y_balanced = self._apply_smote(X_balanced, y_balanced)

        # Apply VAE-based oversampling
        if "vae" in self.methods:
            X_balanced, y_balanced = self._apply_vae_oversampling(
                X_balanced, y_balanced
            )

        # Calculate class weights
        if "weighted_loss" in self.methods:
            class_weights = self._calculate_class_weights(y_balanced)

        return X_balanced, y_balanced, class_weights

    def _apply_smote(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Applies SMOTE to oversample minority classes.

        Args:
            X (pd.DataFrame): Feature set.
            y (pd.Series): Labels.

        Returns:
            Tuple[pd.DataFrame, pd.Series]: Oversampled feature set and labels.
        """
        self.logger.info("Applying SMOTE...")
        try:
            X_res, y_res = self.smote.fit_resample(X, y)
            self.logger.info(f"SMOTE applied. New shape: {X_res.shape}")
            return X_res, y_res
        except Exception as e:
            self.logger.error(f"SMOTE failed: {e}")
            raise ValueError(f"SMOTE failed: {e}")

    def _calculate_class_weights(self, y: pd.Series) -> Dict:
        """
        Calculates class weights for handling class imbalance.

        Args:
            y (pd.Series): Labels.

        Returns:
            Dict: Class weights.
        """
        self.logger.info("Calculating class weights...")
        try:
            classes = np.unique(y)
            class_weights_array = compute_class_weight(
                class_weight="balanced", classes=classes, y=y
            )
            class_weights = {
                cls: weight for cls, weight in zip(classes, class_weights_array)
            }
            self.logger.info(f"Class weights calculated: {class_weights}")
            return class_weights
        except Exception as e:
            self.logger.error(f"Class weight calculation failed: {e}")
            raise ValueError(f"Class weight calculation failed: {e}")

    def _apply_vae_oversampling(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Applies VAE-based oversampling to generate synthetic minority samples.

        Args:
            X (pd.DataFrame): Feature set.
            y (pd.Series): Labels.

        Returns:
            Tuple[pd.DataFrame, pd.Series]: Feature set and labels with synthetic samples.
        """
        self.logger.info("Applying VAE-based oversampling...")
        try:
            X_synthetic, y_synthetic = self._vae_oversample(X, y)
            if not X_synthetic.empty:
                X_balanced = pd.concat([X, X_synthetic], ignore_index=True)
                y_balanced = pd.concat([y, y_synthetic], ignore_index=True)
                self.logger.info(
                    f"VAE-based oversampling applied. New shape: {X_balanced.shape}"
                )
                return X_balanced, y_balanced
            else:
                self.logger.info("No synthetic samples generated by VAE.")
                return X, y
        except Exception as e:
            self.logger.error(f"VAE-based oversampling failed: {e}")
            raise ValueError(f"VAE-based oversampling failed: {e}")

    def _vae_oversample(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Trains a VAE on the minority class and generates synthetic samples.

        Args:
            X (pd.DataFrame): Feature set.
            y (pd.Series): Labels.

        Returns:
            Tuple[pd.DataFrame, pd.Series]: Synthetic feature set and labels.
        """
        # Identify minority class
        class_counts = Counter(y)
        minority_class = min(class_counts, key=class_counts.get)
        self.logger.info(
            f"Minority class identified: {minority_class} with {class_counts[minority_class]} samples."
        )

        # Extract minority class samples
        X_minority = X[y == minority_class].values.astype(np.float32)

        # Normalize features using Min-Max scaling to get values between 0 and 1
        X_minority_min = X_minority.min(axis=0)
        X_minority_max = X_minority.max(axis=0)
        # Avoid division by zero
        X_minority_max_replaced = np.where(
            X_minority_max == X_minority_min, X_minority_min + 1, X_minority_max
        )
        X_minority_norm = (X_minority - X_minority_min) / (
            X_minority_max_replaced - X_minority_min
        )

        # Create TensorDataset
        dataset = TensorDataset(torch.from_numpy(X_minority_norm))
        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Training loop with Early Stopping
        best_loss = float("inf")
        patience = 10
        trigger_times = 0
        best_model_path = "best_vae.pth"

        for epoch in range(1, self.epochs + 1):
            self.vae.train()
            epoch_loss = 0
            for batch in data_loader:
                x = batch[0].to(self.device)
                self.optimizer.zero_grad()
                recon_x, mu, logvar = self.vae(x)
                loss = vae_loss(recon_x, x, mu, logvar)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            avg_loss = epoch_loss / len(data_loader.dataset)
            self.logger.info(f"VAE Epoch {epoch}/{self.epochs}, Loss: {avg_loss:.4f}")

            # Early Stopping Check
            if avg_loss < best_loss:
                best_loss = avg_loss
                trigger_times = 0
                # Save the best model
                torch.save(self.vae.state_dict(), best_model_path)
                self.logger.info(f"New best VAE model saved at epoch {epoch}.")
            else:
                trigger_times += 1
                self.logger.info(f"No improvement in loss for {trigger_times} epochs.")
                if trigger_times >= patience:
                    self.logger.info("Early stopping triggered.")
                    break

        # Load the best model
        self.vae.load_state_dict(torch.load(best_model_path))
        Path(best_model_path).unlink()  # Remove the checkpoint file
        self.vae.eval()

        # Calculate the number of synthetic samples needed to balance the classes
        majority_class = max(class_counts, key=class_counts.get)
        num_synthetic = class_counts[majority_class] - class_counts[minority_class]
        self.logger.info(
            f"Generating {num_synthetic} synthetic samples for class {minority_class}."
        )

        if num_synthetic <= 0:
            self.logger.info("No synthetic samples needed to balance the classes.")
            return pd.DataFrame(), pd.Series()

        # Generate synthetic samples
        with torch.no_grad():
            z = torch.randn(num_synthetic, self.latent_dim).to(self.device)
            synthetic_data = self.vae.decoder(z).cpu().numpy()

        # Denormalize synthetic data back to the original range
        synthetic_data = (
            synthetic_data * (X_minority_max_replaced - X_minority_min) + X_minority_min
        )

        # Convert to DataFrame
        X_synthetic = pd.DataFrame(synthetic_data, columns=X.columns)
        y_synthetic = pd.Series([minority_class] * num_synthetic)

        return X_synthetic, y_synthetic
