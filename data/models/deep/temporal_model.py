"""
Temporal Prediction Model — LSTM with entity embeddings for F1 prediction.

Uses driver career sequences as temporal input, with learned embeddings
for drivers, teams, and circuits. Produces both predictions and embeddings
that can be injected into the XGBoost ensemble as additional features.

Architecture:
    Entity Embeddings (driver + team + circuit)
        ↓ concat
    Bi-LSTM (2 layers, 128 hidden)
        ↓
    Multi-head output:
        - Position regression (MSE)
        - Win classification (BCE)
        - Podium classification (BCE)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).parent.parent.parent / "cache" / "models" / "temporal"

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    logger.warning("PyTorch not installed. Deep learning features disabled.")


if HAS_TORCH:

    class F1TemporalModel(nn.Module):
        """
        Bi-LSTM model with entity embeddings for F1 race prediction.

        Designed to be small enough to avoid overfitting on F1's limited dataset
        (~8000 driver-race samples per decade) while learning meaningful
        temporal patterns in driver/team performance trajectories.
        """

        def __init__(
            self,
            n_features: int,
            n_drivers: int = 200,
            n_constructors: int = 50,
            n_circuits: int = 80,
            driver_embed_dim: int = 16,
            constructor_embed_dim: int = 8,
            circuit_embed_dim: int = 8,
            hidden_dim: int = 128,
            n_layers: int = 2,
            dropout: float = 0.3,
        ):
            super().__init__()

            self.driver_embed_dim = driver_embed_dim
            self.constructor_embed_dim = constructor_embed_dim
            self.circuit_embed_dim = circuit_embed_dim
            self.hidden_dim = hidden_dim

            # Entity embeddings
            self.driver_embedding = nn.Embedding(n_drivers + 1, driver_embed_dim, padding_idx=0)
            self.constructor_embedding = nn.Embedding(n_constructors + 1, constructor_embed_dim, padding_idx=0)
            self.circuit_embedding = nn.Embedding(n_circuits + 1, circuit_embed_dim, padding_idx=0)

            total_embed_dim = driver_embed_dim + constructor_embed_dim + circuit_embed_dim

            # Input projection: features + entity embeddings
            self.input_proj = nn.Linear(n_features + total_embed_dim, hidden_dim)

            # Bi-LSTM
            self.lstm = nn.LSTM(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                num_layers=n_layers,
                batch_first=True,
                bidirectional=True,
                dropout=dropout if n_layers > 1 else 0.0,
            )

            self.dropout = nn.Dropout(dropout)
            self.layer_norm = nn.LayerNorm(hidden_dim * 2)

            # Multi-head output
            self.position_head = nn.Sequential(
                nn.Linear(hidden_dim * 2, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, 1),
            )

            self.win_head = nn.Sequential(
                nn.Linear(hidden_dim * 2, 32),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(32, 1),
                nn.Sigmoid(),
            )

            self.podium_head = nn.Sequential(
                nn.Linear(hidden_dim * 2, 32),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(32, 1),
                nn.Sigmoid(),
            )

        def forward(
            self,
            sequences: torch.Tensor,
            driver_ids: torch.Tensor,
            constructor_ids: torch.Tensor,
            circuit_ids: torch.Tensor,
            seq_lengths: Optional[torch.Tensor] = None,
        ) -> Dict[str, torch.Tensor]:
            """
            Forward pass.

            Args:
                sequences: (batch, seq_len, n_features) temporal feature sequences
                driver_ids: (batch,) driver entity indices
                constructor_ids: (batch,) constructor entity indices
                circuit_ids: (batch,) circuit entity indices
                seq_lengths: (batch,) actual sequence lengths (for packing)

            Returns:
                Dict with 'position', 'win_prob', 'podium_prob', 'embedding' tensors
            """
            batch_size, seq_len, _ = sequences.shape

            # Entity embeddings — broadcast across sequence
            d_emb = self.driver_embedding(driver_ids).unsqueeze(1).expand(-1, seq_len, -1)
            c_emb = self.constructor_embedding(constructor_ids).unsqueeze(1).expand(-1, seq_len, -1)
            t_emb = self.circuit_embedding(circuit_ids).unsqueeze(1).expand(-1, seq_len, -1)

            # Concatenate features + embeddings
            x = torch.cat([sequences, d_emb, c_emb, t_emb], dim=-1)
            x = self.input_proj(x)

            # LSTM
            lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden*2)

            # Use the last timestep output
            if seq_lengths is not None:
                # Gather the output at each sequence's actual last timestep
                idx = (seq_lengths - 1).clamp(min=0).long()
                idx = idx.unsqueeze(1).unsqueeze(2).expand(-1, 1, lstm_out.size(2))
                last_out = lstm_out.gather(1, idx).squeeze(1)
            else:
                last_out = lstm_out[:, -1, :]

            last_out = self.layer_norm(last_out)
            last_out = self.dropout(last_out)

            # Multi-head predictions
            position = self.position_head(last_out).squeeze(-1)
            win_prob = self.win_head(last_out).squeeze(-1)
            podium_prob = self.podium_head(last_out).squeeze(-1)

            return {
                "position": position,
                "win_prob": win_prob,
                "podium_prob": podium_prob,
                "embedding": last_out.detach(),  # for XGBoost injection
            }

        def extract_embeddings(
            self,
            sequences: torch.Tensor,
            driver_ids: torch.Tensor,
            constructor_ids: torch.Tensor,
            circuit_ids: torch.Tensor,
            seq_lengths: Optional[torch.Tensor] = None,
        ) -> np.ndarray:
            """Extract learned embeddings as numpy array for XGBoost injection."""
            self.eval()
            with torch.no_grad():
                out = self.forward(sequences, driver_ids, constructor_ids, circuit_ids, seq_lengths)
            return out["embedding"].cpu().numpy()


    class F1TemporalTrainer:
        """Trains the F1TemporalModel with proper time-series validation."""

        def __init__(
            self,
            n_features: int,
            n_drivers: int = 200,
            n_constructors: int = 50,
            n_circuits: int = 80,
            hidden_dim: int = 128,
            learning_rate: float = 1e-3,
            batch_size: int = 64,
            n_epochs: int = 50,
            patience: int = 10,
            device: Optional[str] = None,
        ):
            self.batch_size = batch_size
            self.n_epochs = n_epochs
            self.patience = patience
            self.learning_rate = learning_rate

            if device:
                self.device = torch.device(device)
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")

            self.model = F1TemporalModel(
                n_features=n_features,
                n_drivers=n_drivers,
                n_constructors=n_constructors,
                n_circuits=n_circuits,
                hidden_dim=hidden_dim,
            ).to(self.device)

            self.optimizer = torch.optim.AdamW(
                self.model.parameters(), lr=learning_rate, weight_decay=1e-4,
            )
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="min", patience=5, factor=0.5,
            )

            logger.info("Temporal model on device: %s", self.device)
            n_params = sum(p.numel() for p in self.model.parameters())
            logger.info("Model parameters: %d", n_params)

        def train(
            self,
            sequences: np.ndarray,
            targets: np.ndarray,
            driver_indices: np.ndarray,
            constructor_indices: np.ndarray,
            circuit_indices: np.ndarray,
            seq_lengths: np.ndarray,
            val_fraction: float = 0.15,
        ) -> Dict[str, List[float]]:
            """
            Train the model with time-aware validation split.

            Uses the last `val_fraction` of samples as validation (since data
            is chronologically ordered, this prevents future data leakage).

            Returns:
                Dict with 'train_loss' and 'val_loss' histories
            """
            n_samples = len(sequences)
            split = int(n_samples * (1 - val_fraction))

            # Time-series split (not random)
            train_seq = torch.FloatTensor(sequences[:split]).to(self.device)
            train_tgt = torch.FloatTensor(targets[:split]).to(self.device)
            train_drv = torch.LongTensor(driver_indices[:split]).to(self.device)
            train_con = torch.LongTensor(constructor_indices[:split]).to(self.device)
            train_cir = torch.LongTensor(circuit_indices[:split]).to(self.device)
            train_len = torch.LongTensor(seq_lengths[:split]).to(self.device)

            val_seq = torch.FloatTensor(sequences[split:]).to(self.device)
            val_tgt = torch.FloatTensor(targets[split:]).to(self.device)
            val_drv = torch.LongTensor(driver_indices[split:]).to(self.device)
            val_con = torch.LongTensor(constructor_indices[split:]).to(self.device)
            val_cir = torch.LongTensor(circuit_indices[split:]).to(self.device)
            val_len = torch.LongTensor(seq_lengths[split:]).to(self.device)

            train_dataset = TensorDataset(train_seq, train_tgt, train_drv, train_con, train_cir, train_len)
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

            mse_loss = nn.MSELoss()
            bce_loss = nn.BCELoss()

            history = {"train_loss": [], "val_loss": []}
            best_val = float("inf")
            patience_counter = 0

            for epoch in range(self.n_epochs):
                # Training
                self.model.train()
                epoch_loss = 0.0
                n_batches = 0

                for batch in train_loader:
                    b_seq, b_tgt, b_drv, b_con, b_cir, b_len = batch

                    self.optimizer.zero_grad()
                    out = self.model(b_seq, b_drv, b_con, b_cir, b_len)

                    # Multi-task loss
                    loss_pos = mse_loss(out["position"], b_tgt)
                    loss_win = bce_loss(out["win_prob"], (b_tgt == 1).float())
                    loss_pod = bce_loss(out["podium_prob"], (b_tgt <= 3).float())

                    loss = loss_pos + 0.3 * loss_win + 0.3 * loss_pod
                    loss.backward()

                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()

                    epoch_loss += loss.item()
                    n_batches += 1

                avg_train = epoch_loss / max(n_batches, 1)
                history["train_loss"].append(avg_train)

                # Validation
                self.model.eval()
                with torch.no_grad():
                    val_out = self.model(val_seq, val_drv, val_con, val_cir, val_len)
                    val_loss_pos = mse_loss(val_out["position"], val_tgt)
                    val_loss_win = bce_loss(val_out["win_prob"], (val_tgt == 1).float())
                    val_loss_pod = bce_loss(val_out["podium_prob"], (val_tgt <= 3).float())
                    val_loss = val_loss_pos + 0.3 * val_loss_win + 0.3 * val_loss_pod
                    val_loss_val = val_loss.item()

                history["val_loss"].append(val_loss_val)
                self.scheduler.step(val_loss_val)

                # Early stopping
                if val_loss_val < best_val:
                    best_val = val_loss_val
                    patience_counter = 0
                    self._save_checkpoint("best")
                else:
                    patience_counter += 1

                if epoch % 5 == 0 or patience_counter == 0:
                    logger.info(
                        "Epoch %d/%d: train=%.4f val=%.4f (best=%.4f, patience=%d/%d)",
                        epoch + 1, self.n_epochs, avg_train, val_loss_val,
                        best_val, patience_counter, self.patience,
                    )

                if patience_counter >= self.patience:
                    logger.info("Early stopping at epoch %d", epoch + 1)
                    break

            # Load best model
            self._load_checkpoint("best")
            logger.info("Training complete. Best val loss: %.4f", best_val)

            return history

        def extract_embeddings(
            self,
            sequences: np.ndarray,
            driver_indices: np.ndarray,
            constructor_indices: np.ndarray,
            circuit_indices: np.ndarray,
            seq_lengths: np.ndarray,
        ) -> np.ndarray:
            """
            Extract learned embeddings for all samples.

            Returns:
                (n_samples, hidden_dim*2) numpy array of LSTM hidden states
            """
            self.model.eval()
            all_embeddings = []

            dataset = TensorDataset(
                torch.FloatTensor(sequences),
                torch.LongTensor(driver_indices),
                torch.LongTensor(constructor_indices),
                torch.LongTensor(circuit_indices),
                torch.LongTensor(seq_lengths),
            )
            loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

            with torch.no_grad():
                for batch in loader:
                    b_seq, b_drv, b_con, b_cir, b_len = [b.to(self.device) for b in batch]
                    out = self.model(b_seq, b_drv, b_con, b_cir, b_len)
                    all_embeddings.append(out["embedding"].cpu().numpy())

            embeddings = np.vstack(all_embeddings)
            logger.info("Extracted embeddings: shape=%s", embeddings.shape)
            return embeddings

        def get_entity_embeddings(self) -> Dict[str, np.ndarray]:
            """Extract learned entity embedding weights."""
            self.model.eval()
            return {
                "driver": self.model.driver_embedding.weight.detach().cpu().numpy(),
                "constructor": self.model.constructor_embedding.weight.detach().cpu().numpy(),
                "circuit": self.model.circuit_embedding.weight.detach().cpu().numpy(),
            }

        def _save_checkpoint(self, name: str = "latest"):
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            path = MODEL_DIR / f"temporal_{name}.pt"
            torch.save({
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
            }, path)

        def _load_checkpoint(self, name: str = "latest"):
            path = MODEL_DIR / f"temporal_{name}.pt"
            if path.exists():
                checkpoint = torch.load(path, map_location=self.device, weights_only=True)
                self.model.load_state_dict(checkpoint["model_state"])
                logger.info("Loaded checkpoint: %s", path)

        def save(self):
            """Save final model."""
            self._save_checkpoint("final")
            logger.info("Temporal model saved to %s", MODEL_DIR)

        def load(self):
            """Load saved model."""
            self._load_checkpoint("final")


def train_temporal_model(
    feature_matrix: pd.DataFrame,
    seq_length: int = 20,
    n_epochs: int = 50,
    batch_size: int = 64,
) -> Optional[F1TemporalTrainer]:
    """
    End-to-end training pipeline for the temporal model.

    Args:
        feature_matrix: Full feature matrix
        seq_length: Sequence length for LSTM input
        n_epochs: Training epochs
        batch_size: Batch size

    Returns:
        Trained F1TemporalTrainer or None if PyTorch unavailable
    """
    if not HAS_TORCH:
        logger.warning("PyTorch not installed. Skipping temporal model training.")
        return None

    from data.models.deep.sequences import build_driver_sequences, build_entity_vocabularies, get_entity_indices, MIN_SEQ_LENGTH

    logger.info("Building temporal sequences (seq_length=%d)...", seq_length)
    sequences, targets, driver_ids, seq_lengths = build_driver_sequences(
        feature_matrix, seq_length=seq_length,
    )

    if len(sequences) == 0:
        logger.warning("No valid sequences built. Skipping temporal training.")
        return None

    # Build entity vocabularies and indices
    vocabs = build_entity_vocabularies(feature_matrix)

    # Map driver_ids in sequences to entity indices
    driver_vocab = vocabs.get("driver", {})
    constructor_vocab = vocabs.get("constructor", {})
    circuit_vocab = vocabs.get("circuit", {})

    # For each sequence sample, we need the entity indices of the TARGET race.
    # Rebuild the same sliding-window iteration as build_driver_sequences to get
    # the correct constructor_id and circuit_id for each sample's target row.
    fm_sorted = feature_matrix.sort_values(["season", "round"])
    driver_indices = np.array([driver_vocab.get(d, 0) for d in driver_ids])

    sample_constructors = []
    sample_circuits = []
    for driver_id, driver_data in fm_sorted.groupby("driver_id"):
        driver_data = driver_data.sort_values(["season", "round"])
        if len(driver_data) < MIN_SEQ_LENGTH + 1:
            continue
        for i in range(MIN_SEQ_LENGTH, len(driver_data)):
            target_row = driver_data.iloc[i]
            if pd.isna(target_row.get("position")):
                continue
            cid = target_row.get("constructor_id", "")
            cid = cid if pd.notna(cid) else ""
            tid = target_row.get("circuit_id", "")
            tid = tid if pd.notna(tid) else ""
            sample_constructors.append(constructor_vocab.get(cid, 0))
            sample_circuits.append(circuit_vocab.get(tid, 0))

    constructor_indices = np.array(sample_constructors)
    circuit_indices = np.array(sample_circuits)

    n_features = sequences.shape[2]
    trainer = F1TemporalTrainer(
        n_features=n_features,
        n_drivers=len(driver_vocab),
        n_constructors=len(constructor_vocab),
        n_circuits=len(circuit_vocab),
        n_epochs=n_epochs,
        batch_size=batch_size,
    )

    history = trainer.train(
        sequences, targets, driver_indices, constructor_indices,
        circuit_indices, seq_lengths,
    )

    trainer.save()

    # Save vocabularies for inference
    import json
    vocab_path = MODEL_DIR / "vocabularies.json"
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(vocab_path, "w") as f:
        json.dump(vocabs, f)
    logger.info("Vocabularies saved to %s", vocab_path)

    return trainer


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    CACHE_DIR = Path(__file__).parent.parent.parent / "cache" / "processed"
    fm = pd.read_parquet(CACHE_DIR / "feature_matrix.parquet")

    logger.info("Feature matrix: %d rows × %d cols", len(fm), len(fm.columns))

    trainer = train_temporal_model(fm, n_epochs=50)

    if trainer:
        print("\nTraining complete!")
        embeddings = trainer.get_entity_embeddings()
        for name, emb in embeddings.items():
            print(f"  {name} embeddings: {emb.shape}")
    else:
        print("Training skipped (PyTorch not available)")
