"""
LSTM预测模型训练脚本
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prediction.lstm_predictor import TrajectoryLSTM
from prediction.trajectory_data import generate_dataset, compute_normalization


def train_model(config: dict, save_dir: str = "models"):
    """训练LSTM预测模型"""
    pred_cfg = config["prediction"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 生成训练数据
    print("Generating training data...")
    X, Y = generate_dataset(
        num_trajectories=500,
        seq_length=pred_cfg["sequence_length"],
        pred_horizon=pred_cfg["prediction_horizon"],
        speed_range=tuple(config["debris"]["linear_velocity_range"]),
        ang_speed_range=tuple(config["debris"]["angular_velocity_range"])
    )
    print(f"Dataset: X={X.shape}, Y={Y.shape}")

    # 归一化
    mean, std = compute_normalization(X)
    X_norm = (X - mean) / std
    Y_norm = (Y - mean) / std

    # 划分训练/验证集
    n = len(X_norm)
    n_train = int(n * 0.8)
    indices = np.random.permutation(n)

    X_train = torch.FloatTensor(X_norm[indices[:n_train]])
    Y_train = torch.FloatTensor(Y_norm[indices[:n_train]])
    X_val = torch.FloatTensor(X_norm[indices[n_train:]])
    Y_val = torch.FloatTensor(Y_norm[indices[n_train:]])

    train_loader = DataLoader(
        TensorDataset(X_train, Y_train),
        batch_size=pred_cfg["batch_size"],
        shuffle=True
    )

    # 创建模型
    model = TrajectoryLSTM(
        hidden_size=pred_cfg["hidden_size"],
        num_layers=pred_cfg["num_layers"],
        prediction_horizon=pred_cfg["prediction_horizon"]
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=pred_cfg["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    criterion = nn.MSELoss()

    # 训练
    best_val_loss = float('inf')
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(pred_cfg["epochs"]):
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            pred, _ = model(batch_x)
            loss = criterion(pred, batch_y)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # 验证
        model.eval()
        with torch.no_grad():
            X_val_d = X_val.to(device)
            Y_val_d = Y_val.to(device)
            val_pred, _ = model(X_val_d)
            val_loss = criterion(val_pred, Y_val_d).item()

        scheduler.step(val_loss)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{pred_cfg['epochs']}: "
                  f"train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(save_dir, "lstm_predictor.pth"))
            np.savez(os.path.join(save_dir, "normalization.npz"), mean=mean, std=std)

    print(f"Training complete. Best val loss: {best_val_loss:.6f}")
    print(f"Model saved to {save_dir}/lstm_predictor.pth")

    return model, mean, std


if __name__ == "__main__":
    import yaml
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               "config", "default_config.yaml")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    train_model(config)
