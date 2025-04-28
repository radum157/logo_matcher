import os
import pandas as pd
import numpy as np
from logger.logger_init import setup_logger
from extractor.logo_extractor import WebsiteLogoExtractor
import cv2
import logging
import time
from collections import defaultdict
from tqdm import tqdm
import argparse
import concurrent.futures
from functools import partial
import json
import networkx as nx
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt
from similarity.logo_similarity import LogoSimilarityMatcher

class LogoClusterer:
    def __init__(self, args, df: pd.DataFrame, logger=None):
        self.logger = logger or logging.Logger(name=__name__, level=logging.INFO)
        self.df = df
        self.output_file = args.output
        self.format = args.format
        self.threshold = args.threshold
        self.logos_dir = args.logos_dir
        self.batch_size = args.batch_size
        self.delay = args.delay
        self.chunk_size = args.chunk_size
        self.workers = args.workers

    def _extract_logo_for_url(self, url, extractor):
        try:
            if not isinstance(url, str) or not url.strip():
                return  # Skip invalid URLs
            url = url.strip()
            extractor.extract_logo("https://" + url, self.logos_dir)
        except Exception as e:
            self.logger.warning(f"Failed to extract logo for {url}: {e}")

    def _downloadLogos(self):
        extractor = WebsiteLogoExtractor(timeout=0.5, logger=setup_logger('Logo Extractor'))
        # Create output directory
        os.makedirs(self.logos_dir, exist_ok=True)

        # Prepare the list of valid URLs
        urls = [url for url in self.df["domain"] if isinstance(url, str) and url.strip()]

        # Process URLs in batches
        for i in range(0, len(urls), self.batch_size):
            batch_urls = urls[i:i + self.batch_size]
            self.logger.info(f"Processing batch {i//self.batch_size + 1} with {len(batch_urls)} URLs")

            # Extract logos in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.workers) as executor:
                list(tqdm(
                    executor.map(partial(self._extract_logo_for_url, extractor=extractor), batch_urls),
                    total=len(batch_urls),
                    desc=f"Batch {i//self.batch_size + 1}"
                ))

            if i + self.batch_size < len(urls):
                self.logger.info(f"Batch complete. Sleeping {self.delay} seconds")
                time.sleep(self.delay)  # Add delay between batches

    def _load_logo_image(self, logo_path):
        """Load a logo image from file path"""
        try:
            img = cv2.imread(logo_path)
            if img is None:
                self.logger.warning(f"Failed to load image: {logo_path}")
                return None
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img
        except Exception as e:
            self.logger.warning(f"Error loading image {logo_path}: {e}")
            return None

    def _compute_pairwise_similarities(self, logo_files, similarity_matcher):
        """Compute pairwise similarities between all logos"""
        n = len(logo_files)
        self.logger.info(f"Computing similarities between {n} logos")

        # Initialize similarity matrix
        similarity_matrix = np.zeros((n, n))
        similarity_details = {}

        # Load all images first
        images = {}
        for i, logo_file in enumerate(tqdm(logo_files, desc="Loading images")):
            company = os.path.basename(logo_file).split('_logo.jpg')[0]
            img = self._load_logo_image(logo_file)
            if img is not None:
                images[company] = img

        companies = list(images.keys())
        n_companies = len(companies)

        # Compute similarities in chunks to avoid memory issues
        for i in tqdm(range(n_companies), desc="Computing similarities"):
            for j in range(i+1, n_companies):
                company1 = companies[i]
                company2 = companies[j]

                # Compute similarity
                similarity = similarity_matcher.compute_similarity(
                    images[company1],
                    images[company2]
                )

                # Compute an overall similarity score (weighted average)
                weights = {
                    'color_histogram_bhattacharyya': 0.3,
                    'color_entropy_similarity': 0.1,
                    'shape_moment_similarity': 0.2,
                    'hog_similarity': 0.2,
                    'symmetry_similarity': 0.1,
                    'perceptual_hash_similarity': 0.1
                }

                overall_score = sum(
                    similarity[metric] * weight
                    for metric, weight in weights.items()
                )

                # Store in matrix
                similarity_matrix[i, j] = overall_score
                similarity_matrix[j, i] = overall_score

                # Store detailed similarity
                if overall_score >= self.threshold:
                    if company1 not in similarity_details:
                        similarity_details[company1] = {}
                    similarity_details[company1][company2] = {
                        'overall': overall_score,
                        'details': similarity
                    }

        # Set diagonal to 1 (self-similarity)
        np.fill_diagonal(similarity_matrix, 1.0)

        return similarity_matrix, similarity_details, companies

    def _cluster_logos(self, similarity_matrix, companies):
        """Cluster logos based on similarity matrix"""
        # Convert similarity to distance
        distance_matrix = 1 - similarity_matrix

        # Perform hierarchical clustering
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=1 - self.threshold,
            metric='precomputed',
            linkage='average'
        )

        cluster_labels = clustering.fit_predict(distance_matrix)

        # Group companies by cluster
        clusters = defaultdict(list)
        for company, label in zip(companies, cluster_labels):
            clusters[int(label)].append(company)

        return dict(clusters)

    def _alternative_graph_clustering(self, similarity_details, companies):
        """Alternative clustering approach using graph community detection"""
        G = nx.Graph()

        # Add all companies as nodes
        for company in companies:
            G.add_node(company)

        # Add edges for similar logos
        for company1, similar_companies in similarity_details.items():
            for company2, similarity in similar_companies.items():
                if similarity['overall'] >= self.threshold:
                    G.add_edge(company1, company2, weight=similarity['overall'])

        # Find communities
        communities = nx.community.greedy_modularity_communities(G)

        # Convert to dictionary
        clusters = {}
        for i, community in enumerate(communities):
            clusters[i] = list(community)

        return clusters

    def _save_results(self, clusters, similarity_details):
        """Save clustering results"""
        result = {
            "clusters": clusters,
            "similarities": similarity_details,
            "threshold": self.threshold,
            "total_clusters": len(clusters),
            "total_companies": sum(len(companies) for companies in clusters.values())
        }

        if self.format == 'json':
            with open(self.output_file, 'w') as f:
                json.dump(result, f, indent=2)
        elif self.format == 'csv':
            # Create dataframe
            rows = []
            for cluster_id, companies in clusters.items():
                for company in companies:
                    rows.append({
                        'company': company,
                        'cluster_id': cluster_id
                    })

            df = pd.DataFrame(rows)
            df.to_csv(self.output_file, index=False)

        self.logger.info(f"Results saved to {self.output_file}")
        self.logger.info(f"Found {len(clusters)} clusters from {result['total_companies']} companies")

    def clusterLogos(self):
        """Main method to cluster logos"""
        # Download logos if needed
        if not os.path.exists(self.logos_dir) or len(os.listdir(self.logos_dir)) == 0:
            self.logger.info("Downloading logos...")
            self._downloadLogos()
        else:
            self.logger.info(f"Using existing logos in {self.logos_dir}")

        # Get list of logo files
        logo_files = [
            os.path.join(self.logos_dir, f)
            for f in os.listdir(self.logos_dir)
            if f.endswith('_logo.jpg')
        ]

        if not logo_files:
            self.logger.error("No logo files found!")
            return

        self.logger.info(f"Found {len(logo_files)} logo files")

        # Initialize similarity matcher
        similarity_matcher = LogoSimilarityMatcher()

        # Compute pairwise similarities
        similarity_matrix, similarity_details, companies = self._compute_pairwise_similarities(
            logo_files, similarity_matcher
        )

        # Cluster logos
        self.logger.info("Clustering logos...")
        clusters = self._cluster_logos(similarity_matrix, companies)

        # Alternative clustering using graph approach
        if len(clusters) == 1:
            self.logger.info("Trying alternative clustering approach...")
            clusters = self._alternative_graph_clustering(similarity_details, companies)

        # Save results
        self._save_results(clusters, similarity_details)

        # Visualize clusters (optional)
        self._visualize_clusters(similarity_matrix, companies, clusters)

        return clusters

    def _visualize_clusters(self, similarity_matrix, companies, clusters):
        """Generate visualization of clusters"""
        try:
            # Create visualization directory
            viz_dir = os.path.join(os.path.dirname(self.output_file), "visualizations")
            os.makedirs(viz_dir, exist_ok=True)

            # Visualize similarity matrix
            plt.figure(figsize=(12, 10))
            plt.imshow(similarity_matrix, cmap='viridis')
            plt.colorbar(label='Similarity')
            plt.title('Logo Similarity Matrix')
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, 'similarity_matrix.png'))

            # Visualize clusters
            # Create a graph from similarity matrix
            G = nx.Graph()

            # Add companies as nodes
            for company in companies:
                G.add_node(company)

            # Add edges for similarities above threshold
            for i, company1 in enumerate(companies):
                for j, company2 in enumerate(companies):
                    if i < j and similarity_matrix[i, j] >= self.threshold:
                        G.add_edge(company1, company2, weight=similarity_matrix[i, j])

            # Set node colors based on cluster
            cluster_lookup = {}
            for cluster_id, cluster_companies in clusters.items():
                for company in cluster_companies:
                    cluster_lookup[company] = cluster_id

            node_colors = [cluster_lookup.get(company, -1) for company in companies]

            # Draw graph
            plt.figure(figsize=(15, 15))
            pos = nx.spring_layout(G, seed=42)
            nx.draw_networkx(
                G, pos,
                node_color=node_colors,
                cmap=plt.cm.tab20,
                node_size=100,
                with_labels=False,
                width=[G[u][v]['weight'] for u, v in G.edges()]
            )
            plt.title('Logo Similarity Network')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, 'similarity_network.png'))

            self.logger.info(f"Visualizations saved to {viz_dir}")
        except Exception as e:
            self.logger.warning(f"Failed to generate visualizations: {e}")

def main():
    """Main function for standalone operation"""
    parser = argparse.ArgumentParser(description="Cluster website logos")
    parser.add_argument("input_data", help="Path to parquet file with websites or JSON file with logo mappings")
    parser.add_argument("--output", "-o", default="logo_clusters.json", help="Path to save the output")
    parser.add_argument("--format", "-f", choices=["json", "csv"], default="json", help="Output format")
    parser.add_argument("--threshold", "-t", type=float, default=0.75, help="Similarity threshold")
    parser.add_argument("--workers", "-w", type=int, default=2, help="Maximum number of concurrent workers")
    parser.add_argument("--logos-dir", "-d", default="logos", help="Directory to store downloaded logos")
    parser.add_argument("--batch-size", "-b", type=int, default=50, help="Batch size for downloads")
    parser.add_argument("--delay", type=int, default=1, help="Delay between batches in seconds")
    parser.add_argument("--chunk-size", "-c", type=int, default=100, help="Chunk size for similarity computation")
    args = parser.parse_args()

    logger = setup_logger(name="Logo Clusterer")

    try:
        df = pd.read_parquet(args.input_data)
        if df.empty:
            logger.error("Input file is empty after discarding the first row.")
            return
    except Exception as e:
        logger.error(f"Failed to load input file: {e}")
        return

    clusterer = LogoClusterer(args, df, logger)
    clusterer.clusterLogos()

if __name__ == "__main__":
    main()
