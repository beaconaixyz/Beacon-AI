import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional
import torchvision.models as models
from .base_model import BaseModel

class MedicalCNN(BaseModel):
    """CNN model for medical image classification"""
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        config = super()._get_default_config()
        config.update({
            'backbone': 'resnet50',
            'pretrained': True,
            'num_classes': 2,
            'dropout_rate': 0.5,
            'freeze_backbone': False
        })
        return config
    
    def _build_model(self) -> nn.Module:
        """Build model architecture"""
        # Get backbone model
        if self.config['backbone'].lower() == 'resnet18':
            backbone = models.resnet18(pretrained=self.config['pretrained'])
        elif self.config['backbone'].lower() == 'resnet34':
            backbone = models.resnet34(pretrained=self.config['pretrained'])
        elif self.config['backbone'].lower() == 'resnet50':
            backbone = models.resnet50(pretrained=self.config['pretrained'])
        elif self.config['backbone'].lower() == 'densenet121':
            backbone = models.densenet121(pretrained=self.config['pretrained'])
        elif self.config['backbone'].lower() == 'efficientnet_b0':
            backbone = models.efficientnet_b0(pretrained=self.config['pretrained'])
        else:
            raise ValueError(f"Unsupported backbone: {self.config['backbone']}")
        
        # Freeze backbone if specified
        if self.config['freeze_backbone']:
            for param in backbone.parameters():
                param.requires_grad = False
        
        # Modify final layer for classification
        if isinstance(backbone, models.ResNet):
            in_features = backbone.fc.in_features
            backbone.fc = nn.Sequential(
                nn.Dropout(self.config['dropout_rate']),
                nn.Linear(in_features, self.config['num_classes'])
            )
        elif isinstance(backbone, models.DenseNet):
            in_features = backbone.classifier.in_features
            backbone.classifier = nn.Sequential(
                nn.Dropout(self.config['dropout_rate']),
                nn.Linear(in_features, self.config['num_classes'])
            )
        elif isinstance(backbone, models.EfficientNet):
            in_features = backbone.classifier[1].in_features
            backbone.classifier = nn.Sequential(
                nn.Dropout(self.config['dropout_rate']),
                nn.Linear(in_features, self.config['num_classes'])
            )
        
        return backbone
    
    def get_feature_maps(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Get intermediate feature maps
        Args:
            x: Input tensor
        Returns:
            List of feature maps
        """
        feature_maps = []
        
        # Move input to device
        x = x.to(self.device)
        
        # Get feature maps for ResNet
        if isinstance(self.model, models.ResNet):
            # Layer 1
            x = self.model.conv1(x)
            x = self.model.bn1(x)
            x = self.model.relu(x)
            x = self.model.maxpool(x)
            feature_maps.append(x)
            
            # Layer 2-5
            x = self.model.layer1(x)
            feature_maps.append(x)
            x = self.model.layer2(x)
            feature_maps.append(x)
            x = self.model.layer3(x)
            feature_maps.append(x)
            x = self.model.layer4(x)
            feature_maps.append(x)
        
        # Get feature maps for DenseNet
        elif isinstance(self.model, models.DenseNet):
            # Initial convolution
            x = self.model.features.conv0(x)
            x = self.model.features.norm0(x)
            x = self.model.features.relu0(x)
            feature_maps.append(x)
            
            # Dense blocks
            x = self.model.features.pool0(x)
            x = self.model.features.denseblock1(x)
            feature_maps.append(x)
            x = self.model.features.transition1(x)
            x = self.model.features.denseblock2(x)
            feature_maps.append(x)
            x = self.model.features.transition2(x)
            x = self.model.features.denseblock3(x)
            feature_maps.append(x)
            x = self.model.features.transition3(x)
            x = self.model.features.denseblock4(x)
            feature_maps.append(x)
        
        # Get feature maps for EfficientNet
        elif isinstance(self.model, models.EfficientNet):
            x = self.model.features[0](x)
            feature_maps.append(x)
            
            for i in range(1, len(self.model.features)):
                x = self.model.features[i](x)
                if i % 2 == 0:  # Add feature maps at regular intervals
                    feature_maps.append(x)
        
        return feature_maps
    
    def get_attention_maps(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get attention maps using Grad-CAM
        Args:
            x: Input tensor
        Returns:
            Attention maps tensor
        """
        self.model.eval()
        x = x.to(self.device)
        
        # Register hooks for gradient and activation
        gradients = []
        activations = []
        
        def save_gradient(grad):
            gradients.append(grad)
        
        def save_activation(module, input, output):
            activations.append(output)
        
        # Get target layer
        if isinstance(self.model, models.ResNet):
            target_layer = self.model.layer4
        elif isinstance(self.model, models.DenseNet):
            target_layer = self.model.features.denseblock4
        elif isinstance(self.model, models.EfficientNet):
            target_layer = self.model.features[-1]
        
        # Register hooks
        handle_activation = target_layer.register_forward_hook(save_activation)
        handle_gradient = target_layer.register_backward_hook(
            lambda module, grad_input, grad_output: save_gradient(grad_output[0])
        )
        
        # Forward pass
        output = self.model(x)
        
        # Backward pass
        if output.dim() == 2:
            score = output[:, output.argmax(dim=1)]
        else:
            score = output
        
        self.model.zero_grad()
        score.backward(torch.ones_like(score))
        
        # Calculate attention maps
        gradients = gradients[0]
        activations = activations[0]
        
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        attention_maps = torch.sum(weights * activations, dim=1, keepdim=True)
        attention_maps = torch.relu(attention_maps)
        
        # Normalize attention maps
        if attention_maps.sum() > 0:
            attention_maps = attention_maps / attention_maps.sum()
        
        # Remove hooks
        handle_activation.remove()
        handle_gradient.remove()
        
        return attention_maps
    
    def get_layer_activations(self, x: torch.Tensor, layer_name: str) -> torch.Tensor:
        """
        Get activations for a specific layer
        Args:
            x: Input tensor
            layer_name: Name of the layer
        Returns:
            Layer activations tensor
        """
        self.model.eval()
        x = x.to(self.device)
        activations = None
        
        def get_activation(name):
            def hook(module, input, output):
                nonlocal activations
                activations = output.detach()
            return hook
        
        # Get target layer
        if hasattr(self.model, layer_name):
            target_layer = getattr(self.model, layer_name)
        else:
            raise ValueError(f"Layer not found: {layer_name}")
        
        # Register hook
        handle = target_layer.register_forward_hook(get_activation(layer_name))
        
        # Forward pass
        self.model(x)
        
        # Remove hook
        handle.remove()
        
        return activations 