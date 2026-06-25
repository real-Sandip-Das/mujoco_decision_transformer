import argparse
import minari
import numpy as np
import torch

from pong_decision_transformer.model import MuJoCoDecisionTransformer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="mujoco/halfcheetah/medium-v0")
    parser.add_argument("--model-path", type=str, default="mujoco_halfcheetah_medium-v0_dt.pt")
    # Recommended Defaults: HalfCheetah=6000, Hopper=3600, Walker2d=5000
    parser.add_argument("--target-rtg", type=float, default=6000.0) 
    parser.add_argument("--context-length", type=int, default=20)
    parser.add_argument("--num-episodes", type=int, default=5)
    parser.add_argument("--render", action="store_true", help="Render the environment UI")
    return parser.parse_args()

def run_evaluation():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating {args.env} targeting RTG: {args.target_rtg}")

    dataset = minari.load_dataset(args.env, download=True)
    
    env_kwargs = {"eval_env": True}
    if args.render:
        env_kwargs["render_mode"] = "human"
        
    env = dataset.recover_environment(**env_kwargs)
    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    max_ep_len = env.spec.max_episode_steps

    model = MuJoCoDecisionTransformer(
        state_dim=state_dim,
        act_dim=act_dim,
        max_ep_len=max_ep_len
    ).to(device)

    # Load weights and normalization stats
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    state_mean = checkpoint['state_mean']
    state_std = checkpoint['state_std']
    model.eval()

    def normalize_state(obs):
        return (obs - state_mean) / state_std

    for episode in range(args.num_episodes):
        obs, _ = env.reset()
        
        history_states = [normalize_state(obs)]
        history_rtgs = [args.target_rtg]
        history_timesteps = [0]
        # Seed placeholder continuous action vector
        history_actions = [np.zeros(act_dim, dtype=np.float32)] 

        episode_reward = 0
        t = 0

        while True:
            s_history = history_states[-args.context_length:]
            a_history = history_actions[-args.context_length:]
            rtg_history = history_rtgs[-args.context_length:]
            t_history = history_timesteps[-args.context_length:]

            pad_len = args.context_length - len(s_history)
            mask = np.concatenate([np.zeros(pad_len), np.ones(len(s_history))])

            if pad_len > 0:
                s_pad = np.zeros((pad_len, state_dim), dtype=np.float32)
                a_pad = np.zeros((pad_len, act_dim), dtype=np.float32)
                rtg_pad = np.zeros(pad_len, dtype=np.float32)
                t_pad = np.zeros(pad_len, dtype=np.int64)

                s_history = np.concatenate([s_pad, s_history], axis=0)
                a_history = np.concatenate([a_pad, a_history], axis=0)
                rtg_history = np.concatenate([rtg_pad, rtg_history], axis=0)
                t_history = np.concatenate([t_pad, t_history], axis=0)

            t_states = torch.tensor(s_history, dtype=torch.float32).unsqueeze(0).to(device)
            t_actions = torch.tensor(a_history, dtype=torch.float32).unsqueeze(0).to(device)
            t_rtg = torch.tensor(rtg_history, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)
            t_timesteps = torch.tensor(t_history, dtype=torch.long).unsqueeze(0).to(device)
            t_mask = torch.tensor(mask, dtype=torch.long).unsqueeze(0).to(device)

            with torch.no_grad():
                action_preds = model(t_states, t_actions, t_rtg, t_timesteps, t_mask)
            
            # Action is a continuous vector
            action = action_preds[0, -1, :].cpu().numpy()
            history_actions[-1] = action

            next_obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            next_rtg = history_rtgs[-1] - reward
            t += 1

            if terminated or truncated:
                break

            history_states.append(normalize_state(next_obs))
            history_rtgs.append(next_rtg)
            history_timesteps.append(t)
            history_actions.append(np.zeros(act_dim, dtype=np.float32))

        print(f"Episode {episode + 1} | Raw Reward: {episode_reward:.1f}")

    env.close()

if __name__ == "__main__":
    run_evaluation()
