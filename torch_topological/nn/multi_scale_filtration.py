import torch
import torch.nn as nn


class MultiScaleFiltration(nn.Module):
    """
    Computes filtrations at multiple scales and learns how to combine them.
    
    This module extends topological analysis by considering structures at 
    different scales, allowing for more robust feature extraction.
    """
    def __init__(self, num_scales=5, scale_min=0.5, scale_max=2.0, learnable_weights=True):
        """
        Initialize the multi-scale filtration.
        
        Parameters
        ----------
        num_scales : int
            Number of different scales to use
            
        scale_min : float
            Minimum scale factor
            
        scale_max : float
            Maximum scale factor
            
        learnable_weights : bool
            Whether to make scale factors and importance weights learnable
        """
        super().__init__()
        
        # Initialize scale factors and importance weights
        if learnable_weights:
            self.scale_factors = nn.Parameter(torch.linspace(scale_min, scale_max, num_scales))
            self.scale_importance = nn.Parameter(torch.ones(num_scales) / num_scales)
        else:
            self.register_buffer('scale_factors', torch.linspace(scale_min, scale_max, num_scales))
            self.register_buffer('scale_importance', torch.ones(num_scales) / num_scales)
            
        self.num_scales = num_scales
    
    def forward(self, filtration_function, weights, distances):
        """
        Apply filtration at multiple scales.
        
        Parameters
        ----------
        filtration_function : callable
            Function that computes filtration from weights and distances
            
        weights : torch.Tensor
            Point weights [batch_size, n_points]
            
        distances : torch.Tensor
            Pairwise distances [batch_size, n_points, n_points]
            
        Returns
        -------
        tuple
            List of modified distances at different scales and their importance weights
        """
        multi_scale_distances = []
        
        # Apply filtration at each scale
        for i in range(self.num_scales):
            scaled_distances = distances * self.scale_factors[i]
            modified_distances = filtration_function(weights, scaled_distances)
            multi_scale_distances.append(modified_distances)
            
        # Return both the modified distances and their importance weights
        return multi_scale_distances, self.scale_importance