import timm
import torch.nn as nn

import torch.nn.functional as F
import torchvision.models as models

# ------------------------------------------------------------------

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

# ------------------------------------------------------------------

class Resnet18(nn.Module):
    def __init__(self, num_classes):
        super(Resnet18, self).__init__()
        self.model = models.resnet18(pretrained=True)
        
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)

class Resnet50(nn.Module):
    def __init__(self, num_classes):
        super(Resnet50, self).__init__()
        self.model = models.resnet50(pretrained=True)
        
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        return self.model(x)

class EfficientNet(nn.Module):
    def __init__(self, num_classes=60, model_name='efficientnet_b0'):
        super(EfficientNet, self).__init__()
        self.model = timm.create_model(model_name, pretrained=True)
        
        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)

# ------------------------------------------------------------------

# base model
class EfficientNetB3(nn.Module):
    def __init__(self, num_classes=60, model_name='efficientnet_b3'):
        super(EfficientNetB3, self).__init__()
        self.model = timm.create_model(model_name, pretrained=True)
        
        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)

# plus linear layer model
class EfficientNetB3(nn.Module):
    def __init__(self, num_classes=60, model_name='efficientnet_b3'):
        super(EfficientNetB3, self).__init__()
        self.model = timm.create_model(model_name, pretrained=True)
        
        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        return self.model(x)

# plus convolution layer model
class EfficientNetB3(nn.Module):
    def __init__(self, num_classes=60, model_name='efficientnet_b3'):
        super(EfficientNetB3, self).__init__()
        self.model = timm.create_model(model_name, pretrained=True)
        
        in_features = self.model.classifier.in_features
        self.conv1 = nn.Conv2d(in_features, 1024, kernel_size=3, padding=1)  # 3x3 convolution
        self.bn1 = nn.BatchNorm2d(1024)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.model.forward_features(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# plus SE layer model
class EfficientNetB3(nn.Module):
    def __init__(self, num_classes=60, model_name='efficientnet_b3'):
        super(EfficientNetB3, self).__init__()
        self.model = timm.create_model(model_name, pretrained=True)
        
        in_features = self.model.classifier.in_features

        self.se_layer = SELayer(in_features)
        self.model.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = self.model.forward_features(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = self.se_layer(x)
        x = x.view(x.size(0), -1) # Flatten the output
        return self.model.classifier(x)

# ------------------------------------------------------------------

class EfficientNetB4(nn.Module):
    def __init__(self, num_classes=60, model_name='efficientnet_b4'):
        super(EfficientNetB4, self).__init__()
        self.model = timm.create_model(model_name, pretrained=True)
        
        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)
    
class EfficientNetB5(nn.Module):
    def __init__(self, num_classes=60, model_name='efficientnet_b5'):
        super(EfficientNetB5, self).__init__()
        self.model = timm.create_model(model_name, pretrained=True)
        
        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)