import torch
import torch.nn as nn
import torch.nn.functional as F

class CommNet(nn.Module):
    """
    A simple implementation of a CommNet network for multi-agent communication.
    All agents share the same network but also exchange message vectors.
    """
    def __init__(self, input_dim, hidden_dim, message_dim, n_agents, n_layers=2):
        super(CommNet, self).__init__()
        self.n_agents = n_agents
        self.n_layers = n_layers
        self.message_dim = message_dim

        # Input to hidden
        self.fc_in = nn.Linear(input_dim, hidden_dim)

        # Communication layers
        self.fc_comm = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(n_layers)])
        self.fc_out = nn.Linear(hidden_dim, message_dim)

    def forward(self, x):
        """
        x shape: (n_agents, input_dim)
        """
        h = F.relu(self.fc_in(x))
        for layer in range(self.n_layers):
            # Compute messages from all agents
            m = self.fc_comm[layer](h)
            m_sum = torch.mean(m, dim=0, keepdim=True)  # shared message (can be weighted)
            h = F.relu(h + m_sum.expand_as(h))
        out = self.fc_out(h)
        return out

# Example usage:
if __name__ == "__main__":
    n_agents = 4
    input_dim = 50
    hidden_dim = 64
    message_dim = 16
    net = CommNet(input_dim, hidden_dim, message_dim, n_agents)
    dummy_input = torch.randn(n_agents, input_dim)
    output = net(dummy_input)
    print("CommNet output shape:", output.shape)
