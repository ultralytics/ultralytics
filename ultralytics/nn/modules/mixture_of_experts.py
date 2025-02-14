import torch
import torch.nn as nn
import torch.nn.functional as F

class MixtureOfExperts(nn.Module):
    def __init__(self, input_dim, num_experts, expert_dim, gating_hidden_dim):
        super(MixtureOfExperts, self).__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([nn.Linear(input_dim, expert_dim) for _ in range(num_experts)])
        self.gating_network = nn.Sequential(
            nn.Linear(input_dim, gating_hidden_dim),
            nn.ReLU(),
            nn.Linear(gating_hidden_dim, num_experts),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        gating_weights = self.gating_network(x)
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=-1)
        output = torch.sum(gating_weights.unsqueeze(-2) * expert_outputs, dim=-1)
        return output
