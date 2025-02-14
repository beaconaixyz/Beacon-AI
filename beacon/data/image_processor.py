import torch
import numpy as np
import cv2
from typing import Dict, Any, Union, Tuple, Optional, List
from ..core.base import BeaconBase
import albumentations as A
from scipy import ndimage
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class ImageProcessor(BeaconBase):
    """Processor for medical image analysis"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize image processor
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self.transform = self._build_transform()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'target_size': (224, 224),
            'normalize': True,
            'augment': False,
            'batch_size': 32,
            'num_workers': 4,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225]
        }
    
    def _build_transform(self) -> transforms.Compose:
        """Build image transformation pipeline"""
        transform_list = []
        
        # Resize
        transform_list.append(
            transforms.Resize(self.config['target_size'])
        )
        
        # Data augmentation if enabled
        if self.config['augment']:
            transform_list.extend([
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.RandomAffine(
                    degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)
                )
            ])
        
        # Convert to tensor
        transform_list.append(transforms.ToTensor())
        
        # Normalize if enabled
        if self.config['normalize']:
            transform_list.append(
                transforms.Normalize(
                    mean=self.config['mean'],
                    std=self.config['std']
                )
            )
        
        return transforms.Compose(transform_list)
    
    def process_image(self, image: Union[str, Path, np.ndarray, Image.Image]) -> torch.Tensor:
        """
        Process a single image
        Args:
            image: Input image (file path or array)
        Returns:
            Processed image tensor
        """
        # Load image if path is provided
        if isinstance(image, (str, Path)):
            image = Image.open(str(image)).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        elif not isinstance(image, Image.Image):
            raise ValueError("Unsupported image type")
        
        # Apply transformations
        return self.transform(image)
    
    def process_batch(self, images: List[Union[str, Path, np.ndarray, Image.Image]]) -> torch.Tensor:
        """
        Process a batch of images
        Args:
            images: List of input images
        Returns:
            Batch tensor of processed images
        """
        processed = [self.process_image(img) for img in images]
        return torch.stack(processed).to(self.config['device'])
    
    def create_dataloader(self, image_paths: List[Union[str, Path]], labels: Optional[List[int]] = None) -> DataLoader:
        """
        Create a DataLoader for a list of images
        Args:
            image_paths: List of image file paths
            labels: Optional list of labels
        Returns:
            DataLoader object
        """
        dataset = MedicalImageDataset(
            image_paths,
            labels,
            transform=self.transform
        )
        
        return DataLoader(
            dataset,
            batch_size=self.config['batch_size'],
            num_workers=self.config['num_workers'],
            shuffle=labels is not None
        )
    
    def validate_image(self, image: Union[str, Path, np.ndarray, Image.Image]) -> Tuple[bool, Optional[str]]:
        """
        Validate an input image
        Args:
            image: Input image
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Load image if path is provided
            if isinstance(image, (str, Path)):
                img = Image.open(str(image))
            elif isinstance(image, np.ndarray):
                img = Image.fromarray(image)
            elif isinstance(image, Image.Image):
                img = image
            else:
                return False, "Unsupported image type"
            
            # Check image mode
            if img.mode not in ['RGB', 'L']:
                return False, f"Unsupported image mode: {img.mode}"
            
            # Check image size
            if any(s < 32 for s in img.size):
                return False, f"Image too small: {img.size}"
            
            return True, None
            
        except Exception as e:
            return False, str(e)
    
    def get_image_stats(self, image: Union[str, Path, np.ndarray, Image.Image]) -> Dict[str, Any]:
        """
        Calculate image statistics
        Args:
            image: Input image
        Returns:
            Dictionary of image statistics
        """
        # Load image if needed
        if isinstance(image, (str, Path)):
            img = cv2.imread(str(image))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif isinstance(image, np.ndarray):
            img = image
        elif isinstance(image, Image.Image):
            img = np.array(image)
        else:
            raise ValueError("Unsupported image type")
        
        # Calculate statistics
        stats = {
            'shape': img.shape,
            'mean': img.mean(axis=(0, 1)),
            'std': img.std(axis=(0, 1)),
            'min': img.min(axis=(0, 1)),
            'max': img.max(axis=(0, 1)),
            'dynamic_range': img.max() - img.min()
        }
        
        # Calculate histogram
        if len(img.shape) == 3:
            stats['histogram'] = [
                np.histogram(img[:,:,i], bins=256, range=(0,255))[0]
                for i in range(img.shape[2])
            ]
        else:
            stats['histogram'] = np.histogram(
                img, bins=256, range=(0,255)
            )[0]
        
        return stats


class MedicalImageDataset(Dataset):
    """Dataset class for medical images"""
    
    def __init__(self, image_paths: List[Union[str, Path]], 
                 labels: Optional[List[int]] = None,
                 transform: Optional[transforms.Compose] = None):
        """
        Initialize dataset
        Args:
            image_paths: List of image file paths
            labels: Optional list of labels
            transform: Optional transform to apply
        """
        self.image_paths = [str(p) for p in image_paths]
        self.labels = labels
        self.transform = transform
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Union[torch.Tensor, Tuple[torch.Tensor, int]]:
        # Load image
        image = Image.open(self.image_paths[idx]).convert('RGB')
        
        # Apply transform if available
        if self.transform:
            image = self.transform(image)
        
        # Return image and label if available
        if self.labels is not None:
            return image, self.labels[idx]
        return image

class MedicalImageProcessor(BeaconBase):
    """Advanced medical image processing and enhancement"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize image processor
        Args:
            config: Configuration dictionary with processing parameters
        """
        super().__init__(config)
        self.augmentation = self._setup_augmentation()
    
    def _setup_augmentation(self) -> A.Compose:
        """Setup image augmentation pipeline"""
        aug_config = self.config.get('augmentation', {})
        return A.Compose([
            A.RandomBrightnessContrast(
                brightness_limit=aug_config.get('brightness_limit', 0.2),
                contrast_limit=aug_config.get('contrast_limit', 0.2),
                p=aug_config.get('brightness_contrast_prob', 0.5)
            ),
            A.GaussNoise(
                var_limit=aug_config.get('noise_var_limit', (10.0, 50.0)),
                p=aug_config.get('noise_prob', 0.3)
            ),
            A.Rotate(
                limit=aug_config.get('rotation_limit', 15),
                p=aug_config.get('rotation_prob', 0.5)
            ),
            A.RandomGamma(
                gamma_limit=aug_config.get('gamma_limit', (80, 120)),
                p=aug_config.get('gamma_prob', 0.3)
            )
        ])
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess medical image
        Args:
            image: Input image array
        Returns:
            Preprocessed image array
        """
        # Ensure float32 type and scale to [0, 1]
        image = image.astype(np.float32)
        if image.max() > 1.0:
            image /= 255.0
        
        # Apply preprocessing steps based on configuration
        if self.config.get('normalize', True):
            image = self._normalize(image)
        
        if self.config.get('denoise', True):
            image = self._denoise(image)
        
        if self.config.get('enhance_contrast', True):
            image = self._enhance_contrast(image)
        
        return image
    
    def augment(self, image: np.ndarray) -> np.ndarray:
        """
        Apply data augmentation
        Args:
            image: Input image array
        Returns:
            Augmented image array
        """
        if not self.config.get('use_augmentation', False):
            return image
        
        # Apply augmentation
        augmented = self.augmentation(image=image)
        return augmented['image']
    
    def _normalize(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize image intensity
        Args:
            image: Input image array
        Returns:
            Normalized image array
        """
        # Z-score normalization
        mean = np.mean(image)
        std = np.std(image)
        if std > 0:
            return (image - mean) / std
        return image
    
    def _denoise(self, image: np.ndarray) -> np.ndarray:
        """
        Apply denoising
        Args:
            image: Input image array
        Returns:
            Denoised image array
        """
        method = self.config.get('denoise_method', 'gaussian')
        
        if method == 'gaussian':
            return cv2.GaussianBlur(
                image,
                ksize=(3, 3),
                sigmaX=self.config.get('gaussian_sigma', 0.5)
            )
        elif method == 'median':
            return cv2.medianBlur(
                (image * 255).astype(np.uint8),
                ksize=self.config.get('median_kernel', 3)
            ).astype(np.float32) / 255.0
        elif method == 'bilateral':
            return cv2.bilateralFilter(
                image,
                d=self.config.get('bilateral_diameter', 9),
                sigmaColor=self.config.get('bilateral_sigma_color', 75),
                sigmaSpace=self.config.get('bilateral_sigma_space', 75)
            )
        
        return image
    
    def _enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance image contrast
        Args:
            image: Input image array
        Returns:
            Contrast-enhanced image array
        """
        method = self.config.get('contrast_method', 'clahe')
        
        if method == 'clahe':
            # Convert to uint8 for CLAHE
            img_uint8 = (image * 255).astype(np.uint8)
            clahe = cv2.createCLAHE(
                clipLimit=self.config.get('clahe_clip_limit', 2.0),
                tileGridSize=self.config.get('clahe_grid_size', (8, 8))
            )
            enhanced = clahe.apply(img_uint8)
            return enhanced.astype(np.float32) / 255.0
        elif method == 'adaptive':
            return self._adaptive_histogram_equalization(image)
        
        return image
    
    def _adaptive_histogram_equalization(self, image: np.ndarray) -> np.ndarray:
        """
        Apply adaptive histogram equalization
        Args:
            image: Input image array
        Returns:
            Enhanced image array
        """
        # Convert to uint8
        img_uint8 = (image * 255).astype(np.uint8)
        
        # Apply adaptive histogram equalization
        window_size = self.config.get('adaptive_window_size', 32)
        enhanced = cv2.equalizeHist(img_uint8)
        
        # Convert back to float32
        return enhanced.astype(np.float32) / 255.0
    
    def segment_tissue(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Segment tissue regions in medical image
        Args:
            image: Input image array
        Returns:
            Tuple of (segmented image, mask)
        """
        # Convert to uint8 if needed
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        
        # Apply Otsu's thresholding
        _, mask = cv2.threshold(
            image,
            0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        
        # Clean up mask with morphological operations
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Apply mask to image
        segmented = cv2.bitwise_and(image, image, mask=mask)
        
        return segmented.astype(np.float32) / 255.0, mask.astype(bool)
    
    def extract_features(self, image: np.ndarray) -> Dict[str, float]:
        """
        Extract image features
        Args:
            image: Input image array
        Returns:
            Dictionary of extracted features
        """
        features = {}
        
        # Basic statistical features
        features['mean_intensity'] = np.mean(image)
        features['std_intensity'] = np.std(image)
        features['min_intensity'] = np.min(image)
        features['max_intensity'] = np.max(image)
        
        # Texture features
        if self.config.get('extract_texture', True):
            glcm = self._compute_glcm(image)
            features.update(self._compute_texture_features(glcm))
        
        return features
    
    def _compute_glcm(self, image: np.ndarray) -> np.ndarray:
        """Compute Gray-Level Co-occurrence Matrix"""
        # Quantize image to fewer gray levels
        bins = self.config.get('glcm_gray_levels', 16)
        img_quantized = np.digitize(image, np.linspace(0, 1, bins))
        
        # Compute GLCM
        glcm = np.zeros((bins, bins))
        for i in range(img_quantized.shape[0]-1):
            for j in range(img_quantized.shape[1]-1):
                glcm[img_quantized[i,j]-1, img_quantized[i+1,j+1]-1] += 1
        
        # Normalize GLCM
        if glcm.sum() > 0:
            glcm = glcm / glcm.sum()
        
        return glcm
    
    def _compute_texture_features(self, glcm: np.ndarray) -> Dict[str, float]:
        """Compute texture features from GLCM"""
        features = {}
        
        # Compute GLCM properties
        features['contrast'] = np.sum(np.square(np.subtract.outer(np.arange(glcm.shape[0]), np.arange(glcm.shape[1]))) * glcm)
        features['homogeneity'] = np.sum(glcm / (1 + np.square(np.subtract.outer(np.arange(glcm.shape[0]), np.arange(glcm.shape[1])))))
        features['energy'] = np.sum(np.square(glcm))
        features['correlation'] = np.sum(np.multiply.outer(np.arange(glcm.shape[0]), np.arange(glcm.shape[1])) * glcm)
        
        return features 