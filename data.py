from CFG import CFG
import torch
from torch.utils.data import Dataset
import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2

# ====================================================
# Dataset
# ====================================================
class ParticleDataset(Dataset):

    def __init__(self, df, transform=None):

        self.df = df
        self.imgpaths = df['imgpaths'].to_numpy()
        self.labels = df[CFG.target_cols].to_numpy()
        self.transform = transform
        self.X_features = df[CFG.cols_mva].to_numpy()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        imgpath = self.imgpaths[idx]

        image = cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        label = torch.tensor(self.labels[idx]).float()
        xfeatures = torch.from_numpy(self.X_features[idx]).float()

        # print(type(image), type(label), type(xfeatures))
        return image, label, imgpath, xfeatures

# ====================================================
# Transformations
# ====================================================
def get_transforms(*, data):
    if data == 'train':
        return A.Compose([
            A.Flip(p=0.5),
            A.Resize(CFG.size, CFG.size),
            A.Normalize(mean=[94., 94., 94.], std=[12., 12., 12.], max_pixel_value=1.0),
            ToTensorV2()
        ])
    elif data == 'valid':
        return A.Compose([
            A.Resize(CFG.size, CFG.size),
            A.Normalize(mean=[94., 94., 94.], std=[12., 12., 12.], max_pixel_value=1.0),
            ToTensorV2()
        ])