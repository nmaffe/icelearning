from CFG import CFG
import torch
import torch.nn as nn
import torchvision.models as models

# ====================================================
# MODEL
# ====================================================
class CustomModel(nn.Module):
    def __init__(self, model_name=CFG.model_name, pretrained=CFG.if_pretrained):

        super().__init__()

        if model_name == 'resnet18':
            self.base = models.resnet18(weights='ResNet18_Weights.DEFAULT')
        if model_name == 'resnet34':
            self.base = models.resnet34(weights='ResNet34_Weights.DEFAULT')
        if model_name == 'resnet152':
            self.base = models.resnet152(weights='ResNet152_Weights.DEFAULT')
        if model_name == 'resnet101':
            self.base = timm.create_model(model_name)
        elif model_name == 'resnext50_32x4d':
            self.base = timm.create_model(model_name)


        n_features = self.base.fc.in_features  # 512

        self.base.fc = nn.Linear(n_features, 64)

        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)

        self.meta_net = nn.Sequential(nn.Linear(34, 128),
                                      nn.BatchNorm1d(128),
                                      nn.ReLU(),
                                      nn.Dropout(p=0.5),
                                      nn.Linear(128, 64),
                                      nn.BatchNorm1d(64),
                                      nn.ReLU(),
                                      nn.Dropout(p=0.5),
                                      nn.Linear(64, 32)
                                      )

        self.fc3 = nn.Linear(96, 40)
        self.bn3 = nn.BatchNorm1d(40)

        self.layer_out = nn.Linear(40, CFG.target_size)

    def forward(self, imgs, metas):
        cnn1 = self.base(imgs)
        x = self.bn1(cnn1)
        x = self.relu(x)
        x = self.dropout(x)

        meta_ = self.meta_net(metas)

        x = torch.cat((x, meta_), 1)

        x = self.fc3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.layer_out(x)
        return x

# ====================================================
# Hooks
# ====================================================
class Hook():
    def __init__(self, name, module, backward=False):

        self.name = name

        if backward == False:
            self.hook = module.register_forward_hook(self.hook_fn)
        else:
            self.hook = module.register_backward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.input = input
        self.output = output

    def close(self):
        self.hook.remove()