import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import minari

from pong_decision_transformer.dataset import MinariDataset
from pong_decision_transformer.model import MuJoCoDecisionTransformer
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs

def train(env: str = "mujoco/halfcheetah/medium-v0", batch_size: int = 256, epochs: int = 100, lr: float = 1e-4, context_length: int = 20):
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
    accelerator.print(f"Total processes: {accelerator.num_processes}")
    print(f"This process is running on: {accelerator.device}")

    # Extract Environment Specs and Load Dataset safely across processes
    with accelerator.main_process_first():
        temp_dataset = minari.load_dataset(env, download=True)
        temp_env = temp_dataset.recover_environment()
        state_dim = temp_env.observation_space.shape[0]
        act_dim = temp_env.action_space.shape[0]
        max_ep_len = temp_env.spec.max_episode_steps
        temp_env.close()

        dataset = MinariDataset(env_name=env, context_length=context_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize Model (Accelerate handles device placement dynamically)
    model = MuJoCoDecisionTransformer(
        state_dim=state_dim, 
        act_dim=act_dim, 
        max_ep_len=max_ep_len
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.1, 
        patience=100,
        min_lr=1e-5
    )
    criterion = nn.MSELoss() # Continuous action matching uses MSE

    model, optimizer, dataloader = accelerator.prepare(
        model, optimizer, dataloader
    )

    accelerator.print(f"Starting training on {env}...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in dataloader:
            states = batch['states']
            actions = batch['actions']
            rtg = batch['rtg']
            timesteps = batch['timesteps']
            attention_mask = batch['attention_mask']

            optimizer.zero_grad()
            action_preds = model(states, actions, rtg, timesteps, attention_mask)

            # Flatten and compute loss only on non-padded timesteps
            action_preds = action_preds.view(-1, act_dim)
            targets = actions.view(-1, act_dim)
            mask_flat = attention_mask.view(-1)

            active_logits = action_preds[mask_flat == 1]
            active_labels = targets[mask_flat == 1]

            loss = criterion(active_logits, active_labels)
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), max_norm=0.25)
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)

        # Synchronize the loss across all GPUs/cores for the scheduler
        local_loss_tensor = torch.tensor([avg_loss], dtype=torch.float32, device=accelerator.device)
        gathered_losses = accelerator.gather(local_loss_tensor)
        global_avg_loss = torch.mean(gathered_losses).item()

        scheduler.step(global_avg_loss)
        current_lr = optimizer.param_groups[0]['lr']
        accelerator.print(f"Epoch {epoch + 1}/{epochs} | Avg MSE Loss: {global_avg_loss:.4f} | LR: {current_lr:.2e}")

    # Save model weights alongside the dataset normalization statistics
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        safe_env_name = env.replace("/", "_")
        save_path = f"{safe_env_name}_dt.pt"
        unwrapped_model = accelerator.unwrap_model(model)
        torch.save({
            'model_state_dict': unwrapped_model.state_dict(),
            'state_mean': dataset.state_mean,
            'state_std': dataset.state_std
        }, save_path)
        accelerator.print(f"Training complete! Artifacts saved to {save_path}")

