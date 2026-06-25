import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import minari

from pong_decision_transformer.dataset import MinariDataset
from pong_decision_transformer.model import MuJoCoDecisionTransformer

def train(env: str = "mujoco/halfcheetah/medium-v0", batch_size: int = 256, epochs: int = 100, lr: float = 1e-4, context_length: int = 20):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Extract Environment Specs
    temp_dataset = minari.load_dataset(env, download=True)
    temp_env = temp_dataset.recover_environment()
    state_dim = temp_env.observation_space.shape[0]
    act_dim = temp_env.action_space.shape[0]
    max_ep_len = temp_env.spec.max_episode_steps
    temp_env.close()

    # Load Dataset
    dataset = MinariDataset(env_name=env, context_length=context_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize Model
    model = MuJoCoDecisionTransformer(
        state_dim=state_dim, 
        act_dim=act_dim, 
        max_ep_len=max_ep_len
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.1, 
        patience=100,
        min_lr=1e-5
    )
    criterion = nn.MSELoss() # Continuous action matching uses MSE

    print(f"Starting training on {env}...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in dataloader:
            states = batch['states'].to(device)
            actions = batch['actions'].to(device)
            rtg = batch['rtg'].to(device)
            timesteps = batch['timesteps'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            optimizer.zero_grad()
            action_preds = model(states, actions, rtg, timesteps, attention_mask)

            # Flatten and compute loss only on non-padded timesteps
            action_preds = action_preds.view(-1, act_dim)
            targets = actions.view(-1, act_dim)
            mask_flat = attention_mask.view(-1)

            active_logits = action_preds[mask_flat == 1]
            active_labels = targets[mask_flat == 1]

            loss = criterion(active_logits, active_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.25)
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        scheduler.step(avg_loss)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch + 1}/{epochs} | Avg MSE Loss: {avg_loss:.4f} | LR: {current_lr:.2e}")

    # Save model weights alongside the dataset normalization statistics
    safe_env_name = env.replace("/", "_")
    save_path = f"{safe_env_name}_dt.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'state_mean': dataset.state_mean,
        'state_std': dataset.state_std
    }, save_path)
    print(f"Training complete! Artifacts saved to {save_path}")

