import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
import copy
import numpy as np
import torch
import torch.nn as nn

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# heats[i][j] gives the heat for episode index i and box index j. size is (5, 15) for the current dataset.
# convolutional autoencoder input shape: (N_samples, N_episodes, N_tile_sizes)
# dont use pooling layers, as they will reduce the size of the output too much

# define 2D convolutional autoencoder
class ConvAutoencoder2D(nn.Module):
    def __init__(self, latent_dim=4):
        super().__init__()

        # input shape: (N_samples, 1, N_episodes, N_boxes) = (30, 1, 5, 15)
        """ 
        ENCODER
        """
        self.encoder_conv = nn.Sequential(
            # layer 1: (N_samples, 1, 5, 15) --> (N_samples, 16, 5, 15)
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1), #--> (N_samples, 16, 5, 15)
            nn.BatchNorm2d(16),
            nn.GELU(), 

            # layer2: (N_samples, 16, 5, 15) --> (N_samples, 32, 3, 8)
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU()
        )
        # fully connected layer to latent vector z.
        self.flatten_dim = 32*3*8 #=768
        self.fc_to_latent = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_from_latent = nn.Linear(latent_dim, self.flatten_dim)
        """ 
        DECODER
        """
        self.decoder_conv = nn.Sequential(
            #(N_samples, 32, 3, 8)--> (N_samples, 16, 5, 15) 
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=(0,0)),
            nn.BatchNorm2d(16),
            nn.GELU(),
            #(N_samples, 16, 5, 15) --> (N_samples, 1, 5, 15)
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=1, padding=1)
            # NB: no activation or batch norm on final output.
        )

    # encode: heatmap x --> latent vector z
    def encode(self, x):
        feature_map = self.encoder_conv(x)
        flat_feature_map = torch.flatten(feature_map, start_dim=1)
        return self.fc_to_latent(flat_feature_map)
    # decode: latent vector z --> heatmap x
    def decode(self, z):
        flat_feature_map = self.fc_from_latent(z)
        feature_map = flat_feature_map.view(-1, 32,3,8) #reshape (N_samples, 768) --> (N_samples, 32, 3, 8)
        return self.decoder_conv(feature_map)

    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z


def split_files(files, data_split_mode, r_tens):
    # random split
    if data_split_mode == 0:
        N_samples = len(files)
        train_size = int(0.5 * N_samples)
        val_size = int(0.15 * N_samples)
        test_size = N_samples - train_size - val_size

        split = random_split(
            range(N_samples),
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )
        train_idx = split[0].indices
        val_idx = split[1].indices
        test_idx = split[2].indices

    # tests interpolation of Ri
    elif data_split_mode == 1:
        rng = np.random.default_rng(seed=25)
        n_test_per_seen_radius = 1

        #Unique radii in ascending order
        R_np = r_tens.numpy().flatten()
        unique_radii = np.sort(np.unique(R_np))
        train_idx = []
        val_idx = []
        test_idx = []
        train_radii = unique_radii[::2] 
        unseen_radii = unique_radii[1::2]

        # for each R in train_radii, shuffle the indices and select n_test_per_seen_radius for testing, and the rest for training/validation
        # for each R in unseen_radii, add all indices to the test set
        for i, R in enumerate(train_radii):
            radius_idx = np.where(R_np == R)[0]
            shuffled_idx = rng.permutation(radius_idx)
            test_idx.extend(shuffled_idx[:n_test_per_seen_radius])
            if i % 3 == 0:
                val_idx.extend(shuffled_idx[n_test_per_seen_radius:])
            else:
                train_idx.extend(shuffled_idx[n_test_per_seen_radius:])
        for R in unseen_radii:
            radius_idx = np.where(R_np == R)[0]
            test_idx.extend(radius_idx)

    train_idx = np.array(train_idx)
    val_idx = np.array(val_idx)
    test_idx = np.array(test_idx)
    return train_idx, val_idx, test_idx


def prepare_data(heat_choice, files, r_tens, train_idx, val_idx, test_idx):

    # Load selected flock_heat maps
    all_heats = []
    for file in files:
        with np.load(file) as data:
            all_heats.append(data[heat_choice])
    # X_tens shape: (N_samples, 1, len(Dt_set), len(ws))
    X_tens = torch.tensor(np.array(all_heats),dtype=torch.float32).unsqueeze(1)

    # Split into train, val, and test sets
    X_train, X_val, X_test  = X_tens[train_idx], X_tens[val_idx], X_tens[test_idx]
    r_train, r_val, r_test = r_tens[train_idx], r_tens[val_idx], r_tens[test_idx]

    # Normalise using training data only
    X_mean = X_train.mean()
    X_std = X_train.std() + 1e-8

    X_train = (X_train - X_mean) / X_std
    X_val = (X_val - X_mean) / X_std
    X_test = (X_test - X_mean) / X_std

    r_min = r_train.min()
    r_max = r_train.max()

    r_train = (r_train - r_min) / (r_max - r_min)
    r_val = (r_val - r_min) / (r_max - r_min)
    r_test = (r_test - r_min) / (r_max - r_min)

    # Create datasets
    train_dataset = TensorDataset(X_train, r_train)
    val_dataset = TensorDataset(X_val, r_val)
    test_dataset = TensorDataset(X_test, r_test)

    batch_size = min(4, len(train_dataset))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return (
        train_loader, val_loader, test_loader,
        train_dataset, val_dataset, test_dataset,
        r_min, r_max, X_train.shape[1:]
    )



"""
1) Train conv AE: heatmap <--> z
2) Train regressor: z <--> Ri
3) Evaluate results 
"""
def run_experiment(
        files,
        heat_choice,
        r_tens,
        train_idx,
        val_idx,
        test_idx,
        latent_dim,
        epochs_ae,
        epochs_reg
):

    # 1. PREPARE DATA
    (
        train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset, r_min, r_max, input_shape
    ) = prepare_data(
        heat_choice, files, r_tens, train_idx, val_idx, test_idx
        )


    # 2. INITIALISE AUTOENCODER
    AE = ConvAutoencoder2D(latent_dim=latent_dim)
    AE_optimiser = torch.optim.AdamW(AE.parameters(), lr=1e-3, weight_decay=1e-4)
    AE_criterion = nn.MSELoss()
    best_ae_val_loss = float("inf")
    best_ae_state = None


    #3. TRAIN AUTOENCODER
    for epoch in range(1, epochs_ae + 1):

        AE.train()
        train_loss = 0.0
        for x_batch, _ in train_loader:
            AE_optimiser.zero_grad() #reset from previous epoch
            x_hat, z = AE(x_batch) #torch calls forward
            loss = AE_criterion(x_hat, x_batch)
            loss.backward() #find gradients (backpropagation)
            AE_optimiser.step() #update params
            train_loss += loss.item() * len(x_batch)
        # average training loss:
        train_loss /= len(train_dataset)

        # Validation reconstruction loss
        AE.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_val, _ in val_loader:
                x_hat_val, _ = AE(x_val)
                loss = AE_criterion(x_hat_val, x_val)
                val_loss += loss.item() * len(x_val)
        val_loss /= len(val_dataset)

        # Save best autoencoder
        if val_loss < best_ae_val_loss:
            best_ae_val_loss = val_loss
            best_ae_state = copy.deepcopy(AE.state_dict())

        if epoch % 10 == 0 or epoch == 1:
            print(f"AE Epoch {epoch:3d}/{epochs_ae} | " f"Train: {train_loss:.5f} | " f"Val: {val_loss:.5f}")

    # Restore best autoencoder
    AE.load_state_dict(best_ae_state)
    AE.eval()
    print(f"\nBest AE validation loss: {best_ae_val_loss:.5f}")


    # 4. EXTRACT LATENT VECTORS
    def extract_latents(dataset):
        X = dataset.tensors[0]
        with torch.no_grad():
            z = AE.encode(X)
        return z

    z_train, z_val, z_test = extract_latents(train_dataset), extract_latents(val_dataset), extract_latents(test_dataset)
    r_train, r_val, r_test = train_dataset.tensors[1], val_dataset.tensors[1], test_dataset.tensors[1]


    # 5. DEFINE LATENT-SPACE REGRESSOR
    # latent_dim --> 8 --> 1
    regressor = nn.Sequential(nn.Linear(latent_dim, 8), nn.GELU(), nn.Linear(8, 1))
    reg_optimiser = torch.optim.AdamW(regressor.parameters(), lr=1e-3, weight_decay=1e-4)
    reg_criterion = nn.MSELoss()

    train_latent_dataset = TensorDataset(z_train, r_train)
    reg_loader = DataLoader(
        train_latent_dataset,
        batch_size=min(4, len(train_latent_dataset)),
        shuffle=True
    )
    best_reg_val_loss = float("inf")
    best_reg_state = None


    # 6. TRAIN REGRESSOR
    for epoch in range(1, epochs_reg + 1):
        regressor.train()
        train_loss = 0.0
        for z_batch, r_batch in reg_loader:
            reg_optimiser.zero_grad()
            r_pred = regressor(z_batch)
            loss = reg_criterion(r_pred, r_batch)
            loss.backward()
            reg_optimiser.step()
            train_loss += loss.item() * len(z_batch)
        train_loss /= len(train_latent_dataset)

        # Validation
        regressor.eval()
        with torch.no_grad():
            r_val_pred = regressor(z_val)
            val_loss = reg_criterion(r_val_pred, r_val).item()

        # Save best regressor
        if val_loss < best_reg_val_loss:
            best_reg_val_loss = val_loss
            best_reg_state = copy.deepcopy(regressor.state_dict())

        if epoch % 20 == 0 or epoch == 1:
            print(f"Reg Epoch {epoch:3d}/{epochs_reg} | " f"Train: {train_loss:.5f} | " f"Val: {val_loss:.5f}")

    # Restore best regressor
    regressor.load_state_dict(best_reg_state)
    regressor.eval()

    print(f"\nBest regressor validation loss: {best_reg_val_loss:.5f}")


    # 7. EVALUATE REGRESSOR ON TEST SET
    with torch.no_grad():
        r_test_pred = regressor(z_test)

    # Convert normalised radii back to physical units
    r_test_actual = (r_test.numpy().flatten() * (r_max.item() - r_min.item()) + r_min.item())
    r_test_pred = (r_test_pred.numpy().flatten() * (r_max.item() - r_min.item()) + r_min.item())

    # Calculate final metrics
    mae = mean_absolute_error(r_test_actual, r_test_pred)
    rmse = np.sqrt(mean_squared_error(r_test_actual, r_test_pred))
    r2 = r2_score(r_test_actual, r_test_pred)
    print(f"\nTest results:\nMAE: {mae:.4f}\nRMSE: {rmse:.4f}\nR2: {r2:.4f}")


    # 8. RETURN RESULTS
    train_radii = np.unique(r_tens[train_idx].cpu().numpy().flatten())
    val_radii = np.unique(r_tens[val_idx].cpu().numpy().flatten())
    return {
        "heat_choice": heat_choice,
        "best_ae_val_loss": best_ae_val_loss,
        "best_reg_val_loss": best_reg_val_loss,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "r_actual": r_test_actual,
        "r_pred": r_test_pred,
        "train_radii": train_radii,
        "val_radii": val_radii,
        "autoencoder": AE,
        "regressor": regressor,

    }