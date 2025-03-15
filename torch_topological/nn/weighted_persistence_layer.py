import torch
import torch.nn as nn

from torch_topological.nn.data import batch_handler, PersistenceInformation, make_tensor
from torch_topological.nn.vietoris_rips_complex import VietorisRipsComplex
from torch_topological.nn.learnable_filtration import LearnableFiltration
from torch_topological.nn.multi_scale_filtration import MultiScaleFiltration


class WeightedPersistenceLayer(nn.Module):
    """
    A layer that computes weighted persistent homology with learnable
    filtration and multi-scale capabilities.
    
    This layer combines learnable filtration functions with multi-scale
    analysis to produce topologically-informed features.
    """
    def __init__(
        self, 
        dim=1,
        p=2,
        hidden_dim=32, 
        num_scales=5,
        threshold=float('inf'),
        keep_infinite_features=False
    ):
        """
        Initialize weighted persistence layer.
        
        Parameters
        ----------
        dim : int
            Maximum homology dimension to compute
            
        p : float
            Exponent for Minkowski distance calculation
            
        hidden_dim : int
            Dimension of hidden layers in the filtration function
        
        num_scales : int
            Number of scales to use for multi-scale filtration
            
        threshold : float
            Maximum filtration value
            
        keep_infinite_features : bool
            Whether to keep infinite persistence features
        """
        super().__init__()
        
        self.learnable_filtration = LearnableFiltration(hidden_dim=hidden_dim)
        self.multi_scale = MultiScaleFiltration(num_scales=num_scales)
        self.vr_complex = VietorisRipsComplex(
            dim=dim, 
            p=p, 
            threshold=threshold, 
            keep_infinite_features=keep_infinite_features
        )
        
    def forward(self, x, weights):
        """
        Compute persistent homology with learnable weighted filtration.
        
        Parameters
        ----------
        x : torch.Tensor or list
            Input point cloud(s) [batch_size, n_points, dim]
            
        weights : torch.Tensor
            Weights for each point [batch_size, n_points]
            
        Returns
        -------
        tuple
            List of persistence diagrams at multiple scales and their importance weights
        """
        # Handle non-batch inputs
        if not isinstance(x, torch.Tensor):
            return self._forward_list_input(x, weights)
            
        # Calculate pairwise distances
        distances = torch.cdist(x, x, p=self.vr_complex.p)
        
        # Compute weighted distances at multiple scales
        multi_scale_distances, scale_importance = self.multi_scale(
            self.learnable_filtration, weights, distances
        )
        
        # Compute persistence diagrams at each scale
        multi_scale_persistence = []
        for scale_distances in multi_scale_distances:
            # For each point in batch, create a weighted distance matrix
            persistence_info = self.vr_complex(scale_distances, treat_as_distances=True)
            multi_scale_persistence.append(persistence_info)
        
        return multi_scale_persistence, scale_importance
    
    def _forward_list_input(self, x_list, weights_list):
        """Handle list inputs (point clouds of varying sizes)."""
        # Process each point cloud individually
        results = []
        importances = []
        
        for x, w in zip(x_list, weights_list):
            # Convert to tensor if needed
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x, dtype=torch.float32)
            if not isinstance(w, torch.Tensor):
                w = torch.tensor(w, dtype=torch.float32)
                
            # Add batch dimension
            x = x.unsqueeze(0)
            w = w.unsqueeze(0)
            
            # Process single point cloud
            pers_infos, importance = self(x, w)
            
            # Remove batch dimension for results
            unbatched_pers_infos = [
                [pi[0] for pi in scale_infos]
                for scale_infos in pers_infos
            ]
            
            results.append(unbatched_pers_infos)
            importances.append(importance)
            
        return results, importances
        
    def combine_multi_scale_features(self, multi_scale_persistence, scale_importance=None):
        """
        Combine persistence information from multiple scales.
        
        Parameters
        ----------
        multi_scale_persistence : list
            List of persistence information at different scales
            
        scale_importance : torch.Tensor or None
            Importance weights for each scale
            
        Returns
        -------
        list
            Combined persistence information
        """
        if scale_importance is None:
            # Use equal weighting if no importance provided
            scale_importance = torch.ones(len(multi_scale_persistence)) / len(multi_scale_persistence)
            
        # Get the first scale's persistence info to determine batch size and dimensions
        batch_size = len(multi_scale_persistence[0])
        
        # For each batch item, combine the persistence diagrams from all scales
        combined_persistence = []
        
        for batch_idx in range(batch_size):
            # For each batch item, get persistence info across all scales
            batch_persistence = []
            
            # For each homology dimension, combine features across scales
            dims_seen = set()
            for scale_idx, scale_pers in enumerate(multi_scale_persistence):
                for pers_info in scale_pers[batch_idx]:
                    dim = pers_info.dimension
                    dims_seen.add(dim)
            
            # Process each dimension separately
            for dim in sorted(dims_seen):
                # For each scale, get persistence diagrams for this dimension
                diagrams = []
                for scale_idx, scale_pers in enumerate(multi_scale_persistence):
                    for pers_info in scale_pers[batch_idx]:
                        if pers_info.dimension == dim:
                            diagrams.append(scale_importance[scale_idx] * pers_info.diagram)
                
                if diagrams:
                    # Combine diagrams using weighted average
                    combined_diagram = torch.stack(diagrams).sum(dim=0)
                    
                    # Create new persistence information with combined diagram
                    combined_info = PersistenceInformation(
                        diagram=combined_diagram,
                        dimension=dim
                    )
                    batch_persistence.append(combined_info)
            
            combined_persistence.append(batch_persistence)
            
        return combined_persistence