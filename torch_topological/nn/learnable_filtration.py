import torch
import torch.nn as nn


class LearnableFiltration(nn.Module):
    """
    A learnable function f(w_i, w_j, d(x_i, x_j)) that adaptively determines 
    how weights influence the distance between points.
    
    This module transforms pairwise distances based on point weights, learning
    the relationship between weights and distances through a neural network.
    """
    def __init__(self, hidden_dim=32, activation=nn.ReLU()):
        """Initialize the learnable filtration function.
        
        Parameters
        ----------
        hidden_dim : int
            Dimension of the hidden layer(s)
            
        activation : torch.nn.Module
            Activation function to use in hidden layers
        """
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(3, hidden_dim),
            activation,
            nn.Linear(hidden_dim, hidden_dim),
            activation,
            nn.Linear(hidden_dim, 1),
            nn.Softplus()  # Ensure positive distances
        )
        
    def forward(self, weights, distances):
        """
        Apply the learnable filtration function.
        
        Parameters
        ----------
        weights : torch.Tensor
            Point weights [batch_size, n_points]
            
        distances : torch.Tensor
            Pairwise distances [batch_size, n_points, n_points]
            
        Returns
        -------
        torch.Tensor
            Modified distances [batch_size, n_points, n_points]
        """
        batch_size, n_points = weights.shape
        
        # Expand weights to match pairwise distances matrix
        w_i_expanded = weights.unsqueeze(2).expand(batch_size, n_points, n_points)
        w_j_expanded = weights.unsqueeze(1).expand(batch_size, n_points, n_points)
        
        # Create input tensor for the network [batch_size, n_points, n_points, 3]
        inputs = torch.stack([w_i_expanded, w_j_expanded, distances], dim=-1)
        
        # Process all pairs at once by flattening
        flat_inputs = inputs.view(-1, 3)
        flat_outputs = self.network(flat_inputs)
        
        # Reshape back to original structure
        modified_distances = flat_outputs.view(batch_size, n_points, n_points)
        
        return modified_distances