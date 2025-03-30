import numpy as np
import cv2
import imagehash
from PIL import Image
from dictances import bhattacharyya
from skimage.feature import hog
from skimage.color import rgb2gray
from typing import List, Dict, Any

class LogoSimilarityMatcher:
    def __init__(self):
        """
        Initialize logo similarity matching system with multiple feature extraction strategies
        """
        self.preprocessors = [
            self._remove_background,
            self._normalize_image,
            self._reduce_noise
        ]

    def _remove_background(self, image: np.ndarray) -> np.ndarray:
        """
        Remove background and isolate logo
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Threshold to create binary mask
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Find largest contour (assumed to be logo)
        largest_contour = max(contours, key=cv2.contourArea)

        # Create mask
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [largest_contour], -1, 255, -1)

        # Apply mask
        result = cv2.bitwise_and(image, image, mask=mask)
        return result

    def _normalize_image(self, image: np.ndarray, target_size: tuple = (224, 224)) -> np.ndarray:
        """
        Normalize image size and aspect ratio
        """
        return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

    def _reduce_noise(self, image: np.ndarray) -> np.ndarray:
        """
        Apply noise reduction techniques
        """
        return cv2.GaussianBlur(image, (3, 3), 0)

    def extract_color_signature(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Extract comprehensive color signature
        """
        # Dominant colors using K-means clustering
        pixels = image.reshape((-1, 3))
        pixels = np.float32(pixels)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        k = 5  # Number of dominant colors
        _, labels, palette = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        # Color histogram
        hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()

        return {
            'dominant_colors': palette.astype(int),
            'color_histogram': hist,
            'color_entropy': self._calculate_color_entropy(image)
        }

    def _calculate_color_entropy(self, image: np.ndarray) -> float:
        """
        Calculate color entropy
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist / hist.sum()
        entropy = -np.sum(hist * np.log2(hist + np.finfo(float).eps))
        return entropy

    def extract_shape_features(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Extract shape and structural features
        """
        # Convert to grayscale for shape analysis
        gray = rgb2gray(image)

        # HOG features
        hog_features = hog(gray,
                           orientations=9,
                           pixels_per_cell=(16, 16),
                           cells_per_block=(2, 2),
                           transform_sqrt=True)

        # Hu Moments
        moments = cv2.moments(gray)
        hu_moments = cv2.HuMoments(moments).flatten()

        return {
            'hog_features': hog_features,
            'hu_moments': hu_moments,
            'symmetry_score': self._calculate_symmetry(gray)
        }

    def _calculate_symmetry(self, image: np.ndarray) -> float:
        """
        Calculate image symmetry
        """
        height, width = image.shape
        vertical_mirror = np.fliplr(image)
        diff = np.abs(image - vertical_mirror)
        return 1 - (np.sum(diff) / (height * width * 255))

    def generate_perceptual_hash(self, image: np.ndarray) -> str:
        """
        Generate perceptual hash using multiple techniques
        """
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Multiple hash techniques
        ahash = str(imagehash.average_hash(pil_image))
        phash = str(imagehash.phash(pil_image))
        dhash = str(imagehash.dhash(pil_image))

        return f"{ahash}_{phash}_{dhash}"

    def compute_similarity(self, logo1: np.ndarray, logo2: np.ndarray) -> Dict[str, float]:
        """
        Compute multi-dimensional similarity between two logos
        """
        # Preprocess images
        for preprocessor in self.preprocessors:
            logo1 = preprocessor(logo1)
            logo2 = preprocessor(logo2)

        # Extract features
        color_sig1 = self.extract_color_signature(logo1)
        color_sig2 = self.extract_color_signature(logo2)

        shape_feat1 = self.extract_shape_features(logo1)
        shape_feat2 = self.extract_shape_features(logo2)

        hist_dict1 = {i: value for i, value in enumerate(color_sig1['color_histogram'])}
        hist_dict2 = {i: value for i, value in enumerate(color_sig2['color_histogram'])}

        # Compute similarities
        similarities = {
            'color_histogram_bhattacharyya': 1 - bhattacharyya(
                hist_dict1,
                hist_dict2
            ),
            'color_entropy_similarity': 1 - abs(
                color_sig1['color_entropy'] - color_sig2['color_entropy']
            ),
            'shape_moment_similarity': 1 - np.mean(
                np.abs(shape_feat1['hu_moments'] - shape_feat2['hu_moments'])
            ),
            'hog_similarity': 1 - np.linalg.norm(
                shape_feat1['hog_features'] - shape_feat2['hog_features']
            ),
            'symmetry_similarity': min(
                shape_feat1['symmetry_score'],
                shape_feat2['symmetry_score']
            ),
            'perceptual_hash_similarity': self._hash_similarity(
                self.generate_perceptual_hash(logo1),
                self.generate_perceptual_hash(logo2)
            )
        }

        return similarities

    def _hash_similarity(self, hash1: str, hash2: str) -> float:
        """
        Compute similarity between perceptual hashes
        """
        # Hamming distance-based similarity
        return 1 - sum(h1 != h2 for h1, h2 in zip(hash1, hash2)) / len(hash1)

    def classify_similarity(self, similarities: Dict[str, float], threshold: float = 0.7) -> bool:
        """
        Classify logos as similar based on multi-stage scoring
        """
        # Weighted similarity scoring
        weights = {
            'color_histogram_bhattacharyya': 0.2,
            'color_entropy_similarity': 0.15,
            'shape_moment_similarity': 0.2,
            'hog_similarity': 0.15,
            'symmetry_similarity': 0.1,
            'perceptual_hash_similarity': 0.2
        }

        total_score = sum(
            similarities.get(key, 0) * weight
            for key, weight in weights.items()
        )

        return total_score >= threshold
