import torch
import torch.nn as nn
from transformers import DecisionTransformerModel, DecisionTransformerConfig

class MuJoCoDecisionTransformer(nn.Module):
    def __init__(self, state_dim, act_dim, hidden_size=128, max_ep_len=1000):
        super().__init__()
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.hidden_size = hidden_size

        config = DecisionTransformerConfig(
            state_dim=state_dim,
            act_dim=act_dim,
            hidden_size=hidden_size,
            max_ep_len=max_ep_len,
            vocab_size=1, # Unused for continuous actions, but required by HF config
            n_layer=3,
            n_head=1,
            n_inner=4 * hidden_size,
            activation_function="relu",
            resid_pdrop=0.1,
            attn_pdrop=0.1,
        )
        self.dt = DecisionTransformerModel(config)
        
        # Embedders for continuous MuJoCo states and actions
        self.embed_state = nn.Linear(state_dim, hidden_size)
        self.embed_action = nn.Linear(act_dim, hidden_size)
        self.embed_rtg = nn.Linear(1, hidden_size)
        self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)
        
        # Action prediction head outputs continuous vector bounded by Tanh
        self.predict_action = nn.Sequential(
            nn.Linear(hidden_size, act_dim),
            nn.Tanh()
        )

    def forward(self, states, actions, returns_to_go, timesteps, attention_mask=None):
        batch_size, seq_length = states.shape[0], states.shape[1]

        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)
        returns_embeddings = self.embed_rtg(returns_to_go)
        time_embeddings = self.embed_timestep(timesteps)

        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        returns_embeddings = returns_embeddings + time_embeddings

        stacked_inputs = torch.stack(
            (returns_embeddings, state_embeddings, action_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(batch_size, 3 * seq_length, self.hidden_size)
        stacked_inputs = self.dt.encoder.drop(stacked_inputs)

        if attention_mask is not None:
            stacked_attention_mask = torch.stack(
                (attention_mask, attention_mask, attention_mask), dim=1
            ).permute(0, 2, 1).reshape(batch_size, 3 * seq_length)
        else:
            stacked_attention_mask = None

        transformer_outputs = self.dt.encoder(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
        )
        x = transformer_outputs[0]

        x = x.reshape(batch_size, seq_length, 3, self.hidden_size)
        # Predict continuous actions from state token (index 1 in the stack)
        action_preds = self.predict_action(x[:, :, 1])
        return action_preds
