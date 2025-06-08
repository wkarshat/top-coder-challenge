"""
Clustering Analysis

Performs clustering analysis to identify patterns and groups in data
using various clustering algorithms and evaluation metrics.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.decomposition import PCA

from core.interfaces import IAnalyzer


class ClusteringAnalyzer(IAnalyzer):
    """Clustering analysis for pattern discovery."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize clustering analyzer."""
        self.config = config
        self.algorithms = config.get('algorithms', ['kmeans', 'dbscan'])
        self.max_clusters = config.get('max_clusters', 10)
        self.min_samples = config.get('min_samples', 5)
        self.scale_features = config.get('scale_features', True)
    
    def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Perform clustering analysis."""
        results = {}
        
        # Get numeric columns for clustering
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 2:
            return {'error': 'Need at least 2 numeric columns for clustering'}
        
        # Prepare data
        cluster_data = data[numeric_cols].dropna()
        
        if len(cluster_data) < self.min_samples:
            return {'error': f'Need at least {self.min_samples} samples for clustering'}
        
        # Scale features if requested
        if self.scale_features:
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(cluster_data)
            scaled_df = pd.DataFrame(scaled_data, columns=numeric_cols, 
                                   index=cluster_data.index)
        else:
            scaled_df = cluster_data
        
        # Perform clustering with different algorithms
        if 'kmeans' in self.algorithms:
            results['kmeans'] = self._kmeans_analysis(scaled_df, cluster_data)
        
        if 'dbscan' in self.algorithms:
            results['dbscan'] = self._dbscan_analysis(scaled_df, cluster_data)
        
        if 'hierarchical' in self.algorithms:
            results['hierarchical'] = self._hierarchical_analysis(scaled_df, cluster_data)
        
        # Dimensionality reduction for visualization
        results['dimensionality_reduction'] = self._dimensionality_reduction(scaled_df)
        
        # Feature importance for clustering
        results['feature_importance'] = self._analyze_feature_importance(scaled_df)
        
        return results
    
    def _kmeans_analysis(self, scaled_data: pd.DataFrame, 
                        original_data: pd.DataFrame) -> Dict[str, Any]:
        """Perform K-means clustering analysis."""
        kmeans_results = {}
        
        # Determine optimal number of clusters
        inertias = []
        silhouette_scores = []
        calinski_scores = []
        
        k_range = range(2, min(self.max_clusters + 1, len(scaled_data)))
        
        for k in k_range:
            try:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(scaled_data)
                
                inertias.append(kmeans.inertia_)
                
                if len(set(labels)) > 1:  # Need at least 2 clusters for silhouette
                    sil_score = silhouette_score(scaled_data, labels)
                    cal_score = calinski_harabasz_score(scaled_data, labels)
                    silhouette_scores.append(sil_score)
                    calinski_scores.append(cal_score)
                else:
                    silhouette_scores.append(0)
                    calinski_scores.append(0)
                    
            except Exception as e:
                kmeans_results[f'k_{k}_error'] = str(e)
        
        # Find optimal k using elbow method and silhouette score
        optimal_k = self._find_optimal_k(k_range, inertias, silhouette_scores)
        
        # Perform final clustering with optimal k
        try:
            final_kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
            final_labels = final_kmeans.fit_predict(scaled_data)
            
            # Cluster statistics
            cluster_stats = self._calculate_cluster_statistics(
                original_data, final_labels
            )
            
            kmeans_results.update({
                'optimal_k': optimal_k,
                'inertias': inertias,
                'silhouette_scores': silhouette_scores,
                'calinski_scores': calinski_scores,
                'final_silhouette_score': silhouette_score(scaled_data, final_labels) if len(set(final_labels)) > 1 else 0,
                'cluster_centers': final_kmeans.cluster_centers_.tolist(),
                'cluster_labels': final_labels.tolist(),
                'cluster_statistics': cluster_stats
            })
            
        except Exception as e:
            kmeans_results['final_clustering_error'] = str(e)
        
        return kmeans_results
    
    def _dbscan_analysis(self, scaled_data: pd.DataFrame, 
                        original_data: pd.DataFrame) -> Dict[str, Any]:
        """Perform DBSCAN clustering analysis."""
        dbscan_results = {}
        
        # Try different eps values
        eps_values = np.linspace(0.1, 2.0, 10)
        best_eps = None
        best_score = -1
        best_labels = None
        
        for eps in eps_values:
            try:
                dbscan = DBSCAN(eps=eps, min_samples=max(2, self.min_samples // 2))
                labels = dbscan.fit_predict(scaled_data)
                
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                n_noise = list(labels).count(-1)
                
                if n_clusters > 1 and n_noise < len(scaled_data) * 0.5:
                    # Calculate silhouette score (excluding noise points)
                    non_noise_mask = labels != -1
                    if np.sum(non_noise_mask) > 1:
                        score = silhouette_score(
                            scaled_data[non_noise_mask], 
                            labels[non_noise_mask]
                        )
                        
                        if score > best_score:
                            best_score = score
                            best_eps = eps
                            best_labels = labels
                            
            except Exception:
                continue
        
        if best_labels is not None:
            n_clusters = len(set(best_labels)) - (1 if -1 in best_labels else 0)
            n_noise = list(best_labels).count(-1)
            
            cluster_stats = self._calculate_cluster_statistics(
                original_data, best_labels
            )
            
            dbscan_results.update({
                'best_eps': float(best_eps),
                'n_clusters': n_clusters,
                'n_noise_points': n_noise,
                'noise_ratio': n_noise / len(scaled_data),
                'silhouette_score': float(best_score),
                'cluster_labels': best_labels.tolist(),
                'cluster_statistics': cluster_stats
            })
        else:
            dbscan_results['error'] = 'Could not find suitable clustering parameters'
        
        return dbscan_results
    
    def _hierarchical_analysis(self, scaled_data: pd.DataFrame, 
                             original_data: pd.DataFrame) -> Dict[str, Any]:
        """Perform hierarchical clustering analysis."""
        hierarchical_results = {}
        
        # Try different numbers of clusters
        silhouette_scores = []
        k_range = range(2, min(self.max_clusters + 1, len(scaled_data)))
        
        best_k = 2
        best_score = -1
        
        for k in k_range:
            try:
                hierarchical = AgglomerativeClustering(n_clusters=k)
                labels = hierarchical.fit_predict(scaled_data)
                
                if len(set(labels)) > 1:
                    score = silhouette_score(scaled_data, labels)
                    silhouette_scores.append(score)
                    
                    if score > best_score:
                        best_score = score
                        best_k = k
                else:
                    silhouette_scores.append(0)
                    
            except Exception:
                silhouette_scores.append(0)
        
        # Final clustering with best k
        try:
            final_hierarchical = AgglomerativeClustering(n_clusters=best_k)
            final_labels = final_hierarchical.fit_predict(scaled_data)
            
            cluster_stats = self._calculate_cluster_statistics(
                original_data, final_labels
            )
            
            hierarchical_results.update({
                'optimal_k': best_k,
                'silhouette_scores': silhouette_scores,
                'best_silhouette_score': float(best_score),
                'cluster_labels': final_labels.tolist(),
                'cluster_statistics': cluster_stats
            })
            
        except Exception as e:
            hierarchical_results['error'] = str(e)
        
        return hierarchical_results
    
    def _find_optimal_k(self, k_range: range, inertias: List[float], 
                       silhouette_scores: List[float]) -> int:
        """Find optimal number of clusters using elbow method and silhouette."""
        if not silhouette_scores:
            return 3  # Default
        
        # Find k with highest silhouette score
        max_sil_idx = np.argmax(silhouette_scores)
        optimal_k_sil = list(k_range)[max_sil_idx]
        
        # Simple elbow detection (look for largest decrease in inertia)
        if len(inertias) >= 2:
            decreases = [inertias[i] - inertias[i+1] for i in range(len(inertias)-1)]
            max_decrease_idx = np.argmax(decreases)
            optimal_k_elbow = list(k_range)[max_decrease_idx]
            
            # Prefer silhouette score but consider elbow
            if abs(optimal_k_sil - optimal_k_elbow) <= 1:
                return optimal_k_sil
            else:
                return optimal_k_sil  # Prioritize silhouette score
        
        return optimal_k_sil
    
    def _calculate_cluster_statistics(self, data: pd.DataFrame, 
                                    labels: np.ndarray) -> Dict[str, Any]:
        """Calculate statistics for each cluster."""
        cluster_stats = {}
        
        unique_labels = set(labels)
        
        for label in unique_labels:
            if label == -1:  # Noise points in DBSCAN
                cluster_name = 'noise'
            else:
                cluster_name = f'cluster_{label}'
            
            mask = labels == label
            cluster_data = data[mask]
            
            if len(cluster_data) > 0:
                stats = {}
                
                for col in data.columns:
                    series = cluster_data[col]
                    stats[col] = {
                        'mean': float(series.mean()),
                        'std': float(series.std()),
                        'min': float(series.min()),
                        'max': float(series.max()),
                        'count': len(series)
                    }
                
                cluster_stats[cluster_name] = {
                    'size': len(cluster_data),
                    'percentage': len(cluster_data) / len(data) * 100,
                    'feature_statistics': stats
                }
        
        return cluster_stats
    
    def _dimensionality_reduction(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Perform PCA for dimensionality reduction."""
        reduction_results = {}
        
        try:
            # PCA to 2 components for visualization
            pca_2d = PCA(n_components=2)
            pca_2d_result = pca_2d.fit_transform(data)
            
            # PCA to 3 components if we have enough features
            if data.shape[1] >= 3:
                pca_3d = PCA(n_components=3)
                pca_3d_result = pca_3d.fit_transform(data)
                
                reduction_results['pca_3d'] = {
                    'components': pca_3d_result.tolist(),
                    'explained_variance_ratio': pca_3d.explained_variance_ratio_.tolist(),
                    'cumulative_variance': np.cumsum(pca_3d.explained_variance_ratio_).tolist()
                }
            
            reduction_results['pca_2d'] = {
                'components': pca_2d_result.tolist(),
                'explained_variance_ratio': pca_2d.explained_variance_ratio_.tolist(),
                'cumulative_variance': np.cumsum(pca_2d.explained_variance_ratio_).tolist()
            }
            
        except Exception as e:
            reduction_results['error'] = str(e)
        
        return reduction_results
    
    def _analyze_feature_importance(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze feature importance for clustering."""
        importance_results = {}
        
        try:
            # Calculate variance of each feature
            feature_variances = data.var().to_dict()
            
            # Calculate correlation with first principal component
            pca = PCA(n_components=1)
            pc1 = pca.fit_transform(data)
            
            feature_pc1_corr = {}
            for col in data.columns:
                corr = np.corrcoef(data[col], pc1.flatten())[0, 1]
                feature_pc1_corr[col] = float(corr) if not np.isnan(corr) else 0.0
            
            importance_results.update({
                'feature_variances': feature_variances,
                'feature_pc1_correlations': feature_pc1_corr,
                'pc1_explained_variance': float(pca.explained_variance_ratio_[0])
            })
            
        except Exception as e:
            importance_results['error'] = str(e)
        
        return importance_results 