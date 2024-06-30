import torch
import torchvision
import torch.nn as nn
from lightweight_model import LightweightModule

class Models:
    def __init__(
        self, model_name, num_classes, feature_extract, pre_trained=True
    ) -> None:
        self.model_name = model_name
        self.pre_trained = pre_trained
        self.feature_extract = feature_extract
        self.num_classes = num_classes

    def set_parameter_requires_grad(self, model: torch.nn.Module):
        if self.feature_extract:
            for param in model.parameters():
                param.requires_grad = False

    def get_model(self) -> torch.nn.Module:
        model = None

        if self.model_name == "efficientnet":
            weight = (
                None
                if not self.pre_trained
                else torchvision.models.EfficientNet_V2_S_Weights.DEFAULT
            )

            model = torchvision.models.efficientnet_v2_s(weights=weight)
            self.set_parameter_requires_grad(model)
            num_ftrs = model.classifier[-1].in_features
            model.classifier[-1] = torch.nn.Linear(num_ftrs, self.num_classes)
        
        elif self.model_name == "resnet":
            weight = (
                None
                if not self.pre_trained
                else torchvision.models.ResNet50_Weights.DEFAULT
            )
            model = torchvision.models.resnet50(weights=weight)
            self.set_parameter_requires_grad(model)
            num_ftrs = model.fc.in_features
            model.fc = torch.nn.Linear(num_ftrs, self.num_classes)
        
        elif self.model_name == "maxvit":
            weight = (
                None
                if not self.pre_trained
                else torchvision.models.MaxVit_T_Weights.DEFAULT
            )
            model = torchvision.models.maxvit_t(weights=weight)
            self.set_parameter_requires_grad(model)
            num_ftrs = model.classifier[-1].in_features
            model.classifier[-1] = torch.nn.Linear(num_ftrs, self.num_classes)

        elif self.model_name == "efficientnet_b0":
            weight = (
                None
                if not self.pre_trained
                else torchvision.models.EfficientNet_B0_Weights.DEFAULT
            )

            model = torchvision.models.efficientnet_b0(weights=weight)
            self.set_parameter_requires_grad(model)
            num_ftrs = model.classifier[-1].in_features
            model.classifier[-1] = torch.nn.Linear(num_ftrs, self.num_classes)

        elif self.model_name == "mobilenetv3":
            weight = (
                None
                if not self.pre_trained
                else torchvision.models.MobileNet_V3_Large_Weights.DEFAULT
            )

            model = torchvision.models.mobilenet_v3_large(weights=weight)
            self.set_parameter_requires_grad(model)
            num_ftrs = model.classifier[-1].in_features
            model.classifier[-1] = torch.nn.Linear(num_ftrs, self.num_classes)

        elif self.model_name == "shuffernetv2":
            weight = (
                None
                if not self.pre_trained
                else torchvision.models.ShuffleNet_V2_X2_0_Weights.DEFAULT
            )
            model = torchvision.models.shufflenet_v2_x2_0(weights=weight)
            self.set_parameter_requires_grad(model)
            num_ftrs = model.fc.in_features
            model.fc = torch.nn.Linear(num_ftrs, self.num_classes)
            
        else:
            print("Invalid model name, exiting...")
            exit()
        return model
    def get_lightweight_model(self, trained_model) -> torch.nn.Module:
        # Extract the features from the last layer of the trained model
        modules = list(trained_model.children())[:-1]  # Remove the last layer (classifier)
        feature_extractor = nn.Sequential(*modules)
        lightweight_model = nn.Sequential(
            feature_extractor,
            LightweightModule(num_classes=self.num_classes)
        )
        return lightweight_model