# Logo Clustering System

A comprehensive system for extracting, analyzing, and clustering logo images from websites based on visual similarity.

## Overview

This system downloads logos from websites, analyzes them using multiple computer vision techniques, and groups similar logos into clusters. The clustering helps identify visually similar brand identities, which can be useful for brand monitoring, trademark analysis, and competitive intelligence.

## Architecture

The system consists of two main components:

1. **Logo Clusterer (`logo_clusterer.py`)**: Manages the overall process of downloading logos, computing similarities, clustering, and visualizing results.
2. **Logo Similarity Matcher (`logo_similarity.py`)**: Implements the computer vision algorithms used to analyze and compare logo images.

## Key Features

- **Parallel logo extraction** from websites with rate limiting and batching
- **Multi-dimensional similarity analysis** using color, shape, and perceptual features
- **Hierarchical clustering** with dynamic threshold-based grouping
- **Alternative graph-based clustering** as a fallback approach
- **Visualization** of similarity matrices and cluster networks
- **Flexible output formats** (JSON/CSV)

## How It Works

### 1. Logo Extraction

The system downloads logos from website domains using a `WebsiteLogoExtractor` (not shown in provided code). The extraction process:

- Takes a list of domains from a Parquet file
- Creates batches to manage load and respect website rate limits
- Uses multi-threading for parallel downloads
- Implements delays between batches to avoid overloading servers

### 2. Similarity Analysis

The `LogoSimilarityMatcher` implements a sophisticated multi-feature comparison approach:

#### Preprocessing Steps:
- Background removal to isolate the logo
- Image normalization for consistent comparison
- Noise reduction to improve feature extraction

#### Feature Extraction:
- **Color Features**:
  - Dominant colors using K-means clustering
  - Color histograms
  - Color entropy measurements

- **Shape Features**:
  - Histogram of Oriented Gradients (HOG)
  - Hu Moments for shape characteristics
  - Symmetry scoring

- **Perceptual Features**:
  - Multiple perceptual hashing algorithms (average hash, perceptual hash, difference hash)

#### Similarity Calculation:
- Bhattacharyya distance for color histograms
- Normalized differences for entropy measures
- L2 norm distances for HOG features
- Hamming distance for perceptual hashes

The system combines these features using a weighted approach to produce a comprehensive similarity score between any two logos.

### 3. Clustering

Two clustering approaches are implemented:

#### Primary: Hierarchical Agglomerative Clustering
- Uses a distance matrix derived from the similarity scores
- Employs average linkage to group similar logos
- Automatically determines the number of clusters based on a similarity threshold

#### Alternative: Graph-Based Community Detection
- Creates a graph where nodes are logos and edges represent similarity above threshold
- Uses greedy modularity optimization to find communities
- Serves as a fallback when hierarchical clustering produces unsatisfactory results

### 4. Visualization and Output

The system generates:
- A similarity matrix visualization showing relationships between all logos
- A network graph visualization showing cluster relationships
- Output data in either JSON or CSV format containing cluster assignments and similarity details

## Design Decisions

### 1. Multi-dimensional Similarity Approach

Rather than relying on a single comparison technique, the system uses multiple complementary features to improve accuracy:

- **Color features** work well for logos with distinctive color schemes
- **Shape features** capture structural similarities even with different colors
- **Perceptual hashing** provides resilience to minor variations and noise

This comprehensive approach allows the system to detect similarities that might be missed by simpler algorithms.

### 2. Weighted Feature Combination

Different features are weighted based on their importance and reliability:

```python
weights = {
    'color_histogram_bhattacharyya': 0.3,
    'color_entropy_similarity': 0.1,
    'shape_moment_similarity': 0.2,
    'hog_similarity': 0.2,
    'symmetry_similarity': 0.1,
    'perceptual_hash_similarity': 0.1
}
```

This weighting scheme:
- Emphasizes color distribution (0.3) as a primary feature
- Gives significant weight to shape features (0.4 combined)
- Uses entropy and symmetry as complementary signals
- Includes perceptual hashing for robustness

### 3. Two-Phase Clustering

The system implements a fallback clustering approach for cases where the primary method fails to produce meaningful clusters:

1. **Hierarchical clustering** works well for identifying clear groupings based on the full similarity matrix
2. **Graph-based clustering** can find community structures that might be missed in the hierarchical approach

This two-phase approach improves the system's robustness across varied datasets.

### 4. Parallel Processing with Safety Features

To efficiently process large numbers of websites while being respectful of resources:

- Batch processing limits the concurrent load
- Configurable delays prevent overwhelming target servers
- Worker pool management controls system resource usage
- Progress tracking with tqdm provides visibility into long-running operations

## Usage

```bash
python3 logo_clusterer.py input_data.parquet \
    --output results.json \
    --format json \
    --threshold 0.75 \
    --workers 2 \
    --logos-dir logos \
    --batch-size 50 \
    --delay 1 \
    --chunk-size 100
```

or run with docker

```bash
chmod u+x ./*.sh ./app/*.sh
./build.sh && ./docker_run.sh
```

or with run.sh:
```bash
chmod u+x ./*.sh ./app/*.sh
./app/run.sh ./app/ && ./app/cleanup.sh
```

### Arguments:

- `input_data`: Path to parquet file with website domain information
- `--output`: Path to save the results (default: logo_clusters.json)
- `--format`: Output format, either "json" or "csv" (default: json)
- `--threshold`: Similarity threshold for clustering (default: 0.75)
- `--workers`: Maximum number of concurrent download workers (default: 2)
- `--logos-dir`: Directory to store downloaded logos (default: logos)
- `--batch-size`: Number of websites to process in each batch (default: 50)
- `--delay`: Delay between batches in seconds (default: 1)
- `--chunk-size`: Processing chunk size for similarity computation (default: 100)

## Extending the System

The modular design allows for easy extension:

1. **Adding new similarity features**: Extend the `LogoSimilarityMatcher` class with additional feature extractors
2. **Alternative clustering algorithms**: Implement new clustering approaches in the `LogoClusterer` class
3. **Custom visualization**: Extend the `_visualize_clusters` method for domain-specific visualizations

## Performance

Space complexity:

Distance Matrix: O(n²) - stores all pairwise distances
Image Storage: O(n) - stores all logo images
Similarity Details: O(k²) where k is the number of logos above threshold

Time complexity:

1. Logo Extraction: O(n) where n is the number of domains

Each domain requires a separate HTTP request
With parallelization, this becomes O(n/p) where p is the number of parallel workers

2. Pairwise Similarity Computation: O(n²) where n is the number of logos

3. Feature Extraction: O(n × m) where m is the computational cost of feature extraction

For each logo: HOG features, color histograms, etc.
This scales linearly with the number of logos

4. Hierarchical Clustering: O(n²log(n))

Agglomerative clustering has quadratic memory requirements
The algorithm needs the complete distance matrix in memory

5. Graph-based Clustering: O(m × log(n)) where m is the number of edges

Edge creation is O(k²) where k is the number of nodes above threshold
Community detection has roughly linear complexity

## Possible improvements

1. Dimensionality Reduction

Current Issue: Computing and storing the full n×n similarity matrix is expensive.
Improvement:

Implement dimensionality reduction techniques (PCA, t-SNE) to project logos into a lower-dimensional space
Use approximate nearest neighbor algorithms (Locality-Sensitive Hashing, FAISS) for similarity search
This could reduce complexity from O(n²) to O(n×log(n))

2. Progressive Refinement

Current Issue: All logos go through the full pipeline of feature extraction and comparison.
Improvement:

Implement a multi-stage filtering approach:

First pass: Use fast but less accurate perceptual hashes to find potential matches
Second pass: Apply more expensive features only to candidates from first pass

This could reduce the effective complexity by limiting detailed comparisons

3. Incremental Processing

Current Issue: The system processes all logos in one batch.
Improvement:

Implement incremental clustering that can add new logos to existing clusters
Store pre-computed features for previously seen logos
This avoids recomputing the entire similarity matrix when new logos are added

4. Feature Optimization

Current Issue: Extracting all features for every logo is computationally expensive.
Improvement:

Profile to identify which features provide the most discrimination power
Potentially drop or simplify less useful features
Implement early stopping in the similarity calculation when logos are clearly dissimilar

5. Distributed Computing

Current Issue: The system is limited to a single machine.
Improvement:

Split the similarity matrix computation across multiple machines
Use MapReduce or similar paradigms for parallel processing
Consider using cloud services for on-demand scaling

6. Indexing and Caching

Current Issue: Repeated computations for the same logos.
Improvement:

Implement a feature cache to store computed features
Use database indexing for faster similarity lookups
Consider vector databases specialized for similarity search

7. Better Memory Management

Current Issue: Holding all images and the full similarity matrix in memory.
Improvement:

Implement streaming processing to reduce memory footprint
Use disk-based storage for the similarity matrix with memory-mapped files
Process in chunks to limit peak memory usage
