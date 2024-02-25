"""
See original implementation (quite far from this one)
at https://github.com/jakesnell/prototypical-networks

Modified by Kim Bjerge (25 Feb. 2024)
"""

import torch

from torch import Tensor

from easyfsl.methods import FewShotClassifier

class PrototypicalNetworksNovelty(FewShotClassifier):
    """
    Jake Snell, Kevin Swersky, and Richard S. Zemel.
    "Prototypical networks for few-shot learning." (2017)
    https://arxiv.org/abs/1703.05175

    Prototypical networks extract feature vectors for both support and query images. Then it
    computes the mean of support features for each class (called prototypes), and predict
    classification scores for query images based on their euclidean distance to the prototypes.
    """
    def __init__(
        self,
        *args,
        use_normcorr: int = 0, # Default cosine distance 0-2, 3 euclidian distance
        **kwargs,
    ):
        """
        Build Prototypical Networks Novelty by calling the constructor of FewShotClassifier.
        Args:
            use_normcorr: use euclidian distance or normalized correlation to compute scores (0, 1, or 2)
        """
        super().__init__(*args, **kwargs)
        
        self.use_normcorr = use_normcorr
        
    
    def forward(
        self,
        query_images: Tensor,
    ) -> Tensor:
        """
        Overrides forward method of FewShotClassifier.
        Predict query labels based on their distance to class prototypes in the feature space.
        Classification scores are the negative of euclidean distances.
        """
        
        # Extract the features of query images
        self.query_features = self.compute_features(query_images)
        self._raise_error_if_features_are_multi_dimensional(self.query_features)


        if self.use_normcorr == 3: # Euclidian distance to prototypes
            scores = self.l2_distance_to_prototypes(self.query_features)
        else: # Default cosine distance to prototypes 0, 1 or 2
            scores = self.cosine_distance_to_prototypes(self.query_features)
                
        return self.softmax_if_specified(scores)
    
    
    def multivariantScatterLoss(self):
        
        num_centers = len(self.prototypes)
        center_points = self.prototypes
        
        scatterBetweenSum = 0
        for i in range(num_centers-1):
            for j in range(num_centers - (i+1)):
                scatterDiff = center_points[i] - center_points[i+j+1]
                scatterBetween = scatterDiff @ torch.t(scatterDiff)
                scatterBetweenSum += scatterBetween
        
        support_features = self.support_features
        support_labels = self.support_labels
        
        scatterWithinSum = 0
        for i in range(num_centers):
            support_features_center = support_features[support_labels == i]
            for j in range(len(support_features_center)):
                scatterDiff = support_features_center[j] - center_points[i]
                scatterWithin = scatterDiff @ torch.t(scatterDiff)
                scatterWithinSum += scatterWithin
            
        #scatterWithinLoss = torch.sqrt(scatterWithinSum)
        #scatterBetweenLoss = torch.sqrt(scatterBetweenSum)
        scatterLoss = scatterWithinSum/scatterBetweenSum
        
        return scatterWithinSum, scatterBetweenSum, scatterLoss
    
      
    @staticmethod
    def is_transductive() -> bool:
        return False
