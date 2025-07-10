import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class SimilarityMatcher:
    def __init__(self, threshold=0.7):
        self.threshold = threshold
    
    def find_best_match(self, features, player_database):
        """Find best matching player from database"""
        if not player_database:
            return None
        
        best_match_id = None
        best_similarity = 0
        
        for player_id, player_info in player_database.items():
            similarity = self._calculate_similarity(
                features, player_info['features']
            )
            
            if similarity > best_similarity and similarity > self.threshold:
                best_similarity = similarity
                best_match_id = player_id
        
        return best_match_id
    
    def _calculate_similarity(self, features1, features2):
        """Calculate cosine similarity between feature vectors"""
        features1 = features1.reshape(1, -1)
        features2 = features2.reshape(1, -1)
        return cosine_similarity(features1, features2)[0][0]