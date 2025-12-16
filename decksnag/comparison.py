"""Image comparison functionality for detecting slide changes."""

import logging
from typing import Tuple, Optional, Literal
import numpy as np
from PIL import Image
from skimage.metrics import mean_squared_error, structural_similarity

logger = logging.getLogger("decksnag")

# Type for comparison methods
ComparisonMethod = Literal["mse", "ssim", "clip"]


class ImageComparator:
    """Compare images to detect significant changes between slides.

    Supports multiple comparison methods:
    - MSE (Mean Squared Error): Fast pixel-level comparison
    - SSIM (Structural Similarity): Better at detecting perceptual differences
    - CLIP (AI-powered): Semantic comparison using neural network embeddings
    """

    # Default thresholds per method
    DEFAULT_THRESHOLDS = {
        "mse": 0.005,  # Higher = more different (0 = identical)
        "ssim": 0.95,  # Lower = more different (1 = identical)
        "clip": 0.85,  # Lower = more different (1 = identical)
    }

    def __init__(
        self,
        threshold: Optional[float] = None,
        method: ComparisonMethod = "mse",
    ) -> None:
        """Initialize the comparator.

        Args:
            threshold: Comparison threshold. If None, uses default for method.
                      For MSE: Higher threshold = less sensitive (default 0.005)
                      For SSIM: Lower threshold = less sensitive (default 0.95)
                      For CLIP: Lower threshold = less sensitive (default 0.85)
            method: Comparison method - "mse", "ssim", or "clip".
        """
        self.method = method
        self.threshold = threshold if threshold is not None else self.DEFAULT_THRESHOLDS.get(method, 0.005)

        # Validate threshold based on method
        if method == "mse":
            if not 0 < self.threshold < 1:
                raise ValueError("MSE threshold must be between 0 and 1")
        elif method in ("ssim", "clip"):
            if not 0 < self.threshold <= 1:
                raise ValueError(f"{method.upper()} threshold must be between 0 and 1")

        # Lazy-loaded CLIP model
        self._clip_model = None

    def _to_grayscale_array(self, image: Image.Image) -> np.ndarray:
        """Convert PIL Image to grayscale numpy array normalized to 0-1.

        Args:
            image: PIL Image to convert.

        Returns:
            Grayscale numpy array with values in [0, 1].
        """
        # Convert to grayscale
        gray = image.convert("L")
        # Convert to numpy array and normalize to 0-1
        return np.array(gray, dtype=np.float64) / 255.0

    def _resize_to_match(
        self, img1: Image.Image, img2: Image.Image
    ) -> Tuple[Image.Image, Image.Image]:
        """Resize images to match dimensions if needed.

        Args:
            img1: First image.
            img2: Second image.

        Returns:
            Tuple of (img1, img2) with matching dimensions.
        """
        if img1.size == img2.size:
            return img1, img2

        # Use the smaller dimensions
        width = min(img1.width, img2.width)
        height = min(img1.height, img2.height)

        logger.debug(
            f"Resizing images to match: {img1.size} and {img2.size} -> ({width}, {height})"
        )

        img1_resized = img1.resize((width, height), Image.Resampling.LANCZOS)
        img2_resized = img2.resize((width, height), Image.Resampling.LANCZOS)

        return img1_resized, img2_resized

    def _load_clip_model(self) -> None:
        """Lazy load the CLIP model (only when first needed).

        The model is cached in the user data directory to avoid
        re-downloading (~350MB) on each use.
        """
        if self._clip_model is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer
            from decksnag.config_file import ensure_data_dir

            # Use persistent cache directory
            cache_dir = ensure_data_dir() / "models"
            cache_dir.mkdir(parents=True, exist_ok=True)

            logger.info("Loading CLIP model (first use, may take a moment)...")
            logger.debug(f"Model cache directory: {cache_dir}")

            self._clip_model = SentenceTransformer(
                "clip-ViT-B-32",
                cache_folder=str(cache_dir),
            )
            logger.info("CLIP model loaded successfully")
        except ImportError:
            raise ImportError(
                "CLIP comparison requires 'sentence-transformers' and 'torch'. "
                "Try reinstalling: pip install --upgrade decksnag"
            )

    def compute_mse(self, img1: Image.Image, img2: Image.Image) -> float:
        """Compute Mean Squared Error between two images.

        Args:
            img1: First image.
            img2: Second image.

        Returns:
            MSE value (0 = identical, higher = more different).
        """
        # Resize if needed
        img1, img2 = self._resize_to_match(img1, img2)

        # Convert to grayscale arrays
        arr1 = self._to_grayscale_array(img1)
        arr2 = self._to_grayscale_array(img2)

        # Compute MSE
        mse = mean_squared_error(arr1, arr2)

        logger.debug(f"MSE: {mse:.6f} (threshold: {self.threshold})")
        return float(mse)

    def compute_ssim(self, img1: Image.Image, img2: Image.Image) -> float:
        """Compute Structural Similarity Index between two images.

        SSIM is often better at detecting perceptual differences than MSE.

        Args:
            img1: First image.
            img2: Second image.

        Returns:
            SSIM value (1 = identical, lower = more different).
        """
        # Resize if needed
        img1, img2 = self._resize_to_match(img1, img2)

        # Convert to grayscale arrays
        arr1 = self._to_grayscale_array(img1)
        arr2 = self._to_grayscale_array(img2)

        # Compute SSIM
        # data_range is 1.0 since we normalized to 0-1
        ssim = structural_similarity(arr1, arr2, data_range=1.0)

        logger.debug(f"SSIM: {ssim:.6f} (threshold: {self.threshold})")
        return float(ssim)

    def compute_clip_similarity(self, img1: Image.Image, img2: Image.Image) -> float:
        """Compute CLIP embedding cosine similarity between two images.

        Uses the CLIP neural network to convert images to semantic vectors
        and compares their similarity. More robust to visual noise like
        animations, mouse cursors, and video player UI.

        Args:
            img1: First image.
            img2: Second image.

        Returns:
            Cosine similarity (1 = identical, 0 = completely different).
        """
        self._load_clip_model()

        # Resize images for CLIP (224x224 is standard input size)
        img1_rgb = img1.convert("RGB")
        img2_rgb = img2.convert("RGB")
        img1_resized = img1_rgb.resize((224, 224), Image.Resampling.LANCZOS)
        img2_resized = img2_rgb.resize((224, 224), Image.Resampling.LANCZOS)

        # Get embeddings
        emb1 = self._clip_model.encode(img1_resized)
        emb2 = self._clip_model.encode(img2_resized)

        # Compute cosine similarity
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

        logger.debug(f"CLIP similarity: {similarity:.4f} (threshold: {self.threshold})")
        return float(similarity)

    def is_different(self, img1: Image.Image, img2: Image.Image) -> bool:
        """Check if two images are significantly different.

        Uses the configured comparison method (MSE, SSIM, or CLIP).

        Args:
            img1: First image (e.g., previous screenshot).
            img2: Second image (e.g., current screenshot).

        Returns:
            True if images are different enough to be considered
            a new slide, False otherwise.
        """
        if self.method == "clip":
            similarity = self.compute_clip_similarity(img1, img2)
            is_diff = similarity < self.threshold
            if is_diff:
                logger.info(f"Change detected (CLIP: {similarity:.4f} < {self.threshold})")
            else:
                logger.debug(f"No significant change (CLIP: {similarity:.4f} >= {self.threshold})")
            return is_diff

        elif self.method == "ssim":
            ssim = self.compute_ssim(img1, img2)
            is_diff = ssim < self.threshold
            if is_diff:
                logger.info(f"Change detected (SSIM: {ssim:.4f} < {self.threshold})")
            else:
                logger.debug(f"No significant change (SSIM: {ssim:.4f} >= {self.threshold})")
            return is_diff

        else:  # Default: MSE
            mse = self.compute_mse(img1, img2)
            is_diff = mse > self.threshold
            if is_diff:
                logger.info(f"Change detected (MSE: {mse:.6f} > {self.threshold})")
            else:
                logger.debug(f"No significant change (MSE: {mse:.6f} <= {self.threshold})")
            return is_diff

    def set_threshold(self, threshold: float) -> None:
        """Update the comparison threshold.

        Args:
            threshold: New threshold value.
        """
        if self.method == "mse":
            if not 0 < threshold < 1:
                raise ValueError("MSE threshold must be between 0 and 1")
        elif self.method in ("ssim", "clip"):
            if not 0 < threshold <= 1:
                raise ValueError(f"{self.method.upper()} threshold must be between 0 and 1")

        self.threshold = threshold
        logger.debug(f"Threshold updated to: {threshold}")

    @staticmethod
    def threshold_from_sensitivity(
        sensitivity: str,
        method: ComparisonMethod = "mse",
    ) -> float:
        """Convert a sensitivity preset to a threshold value.

        Args:
            sensitivity: One of 'low', 'medium', 'high'.
            method: Comparison method ("mse", "ssim", or "clip").

        Returns:
            Corresponding threshold value for the method.
        """
        presets = {
            "mse": {
                "low": 0.01,  # Less sensitive (bigger changes needed)
                "medium": 0.005,  # Default
                "high": 0.001,  # More sensitive (catches subtle changes)
            },
            "ssim": {
                "low": 0.90,  # Less sensitive
                "medium": 0.95,  # Default
                "high": 0.98,  # More sensitive
            },
            "clip": {
                "low": 0.80,  # Less sensitive
                "medium": 0.85,  # Default
                "high": 0.92,  # More sensitive
            },
        }

        method_presets = presets.get(method, presets["mse"])

        if sensitivity not in method_presets:
            raise ValueError(
                f"Unknown sensitivity: {sensitivity}. "
                f"Use one of: {list(method_presets.keys())}"
            )

        return method_presets[sensitivity]
