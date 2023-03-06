import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy
import cv2
import time

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from CFG import CFG
from data import ParticleDataset, get_transforms
from model import CustomModel, Hook
from utils import plot_confusion_matrix

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# ====================================================
# Load model
# ====================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'working with {device}')

# initialize a model
model = CustomModel(CFG.model_name, pretrained=CFG.if_pretrained)
model.to(device)

# load checkpoint and load the saved model
checkpoint = torch.load(CFG.OUTPUT_DIR+f'{CFG.model_name_saved}.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])

# make a copy of the model (we don't want to update the trained model)
model = copy.deepcopy(model)

# ====================================================
# Create GRIP test dataset
# ====================================================
PATH_TEST = 'datasets/test/GRIP_raw/'
li = []
for filename in ['GRIP_3046_raw.csv', 'GRIP_3136_raw.csv', 'GRIP_3303_raw.csv', 'GRIP_3306_raw.csv']:
    df = pd.read_csv(PATH_TEST + filename)
    li.append(df)
test_bag = pd.concat(li, axis=0, ignore_index=True)

### Add all zeros to label columns
for col in CFG.target_cols:
    test_bag[col] = [0] * len(test_bag)

def getsubbag(stringa):
    folder = stringa.split('/')[8]
    bag = folder.split('_')[1]
    subbag = folder.split('_')[2]
    repet = folder.split('_')[4]
    return int(subbag)

test = test_bag

scaler = StandardScaler()
test_meta = copy.deepcopy(test)
test_meta[CFG.cols_mva] = scaler.fit_transform(test_meta[CFG.cols_mva])
print(f'Size of GRIP test dataset: {len(test)} particles')

# ====================================================
# Inference loop
# ====================================================
def test_loop(model, test_meta):

    start = time.time()
    model.eval()

    hookfeatures = []

    print('Start inference on test dataset...')

    # Note we don't want shuffle here since we want to preserve order of dataframe
    test_dataset = ParticleDataset(test_meta, transform=get_transforms(data='valid'))
    test_loader = DataLoader(test_dataset, batch_size=2*CFG.batch_size, shuffle=False,
                             num_workers=2*CFG.num_workers, pin_memory=True, drop_last=False)

    results_test = {'paths': [], 'preds': [], 'probs': []}

    with torch.no_grad():
        for i, (images, labels, paths, xfeatures) in enumerate(test_loader):

            images = images.to(device)
            xfeatures = xfeatures.to(device)

            y_preds = model(images, xfeatures)

            preds = (y_preds == y_preds.max(dim=1, keepdim=True)[0]).to(dtype=torch.int32).cpu().numpy()
            probs = F.softmax(y_preds, dim=1).cpu().numpy()

            hook_emb = hook[0].output.detach().cpu().numpy()
            hookfeatures.append(hook_emb)

            # Fill dictionary with results
            results_test['paths'] += paths
            results_test['preds'].append(preds)
            results_test['probs'].append(probs)

        results_test['hookfeatures'] = np.vstack(hookfeatures)

    # Create ndarrays of test size
    imgpaths = results_test['paths']
    results = np.concatenate(results_test['preds'], axis=0)
    probabilities = np.concatenate(results_test['probs'], axis=0)

    # Transform into dataframes and merge the predictions and probabilities into final dataframe
    df_imgpaths = pd.DataFrame(imgpaths, columns=['imgpaths'])
    df_preds = pd.DataFrame(results, columns=CFG.target_cols)
    df_probs = pd.DataFrame(probabilities, columns=CFG.prob_cols)

    df_final = pd.concat([df_imgpaths, df_preds, df_probs], axis=1)
    df_final['hookfeatures'] = results_test['hookfeatures'].tolist()

    print(f'Finished in {time.time() - start:.2f} sec.')
    return df_final

# Initialize layers to be monitored: FC layer of the resnet network
hook = [Hook(name, layer, backward=False) for name, layer in model.named_modules() if name=='base.fc']
print(f'How many layers we are monitoring: {len(hook)}')
print(f'We are monitoring: {hook[0].name}')

# inference
test_results = test_loop(model, test_meta)

# print total per-class predictions
for col in CFG.target_cols:
    print(f'{col}\t{test_results[col].sum()}\t{test_results[col].sum()/len(test_results):.4f}')

# add predictions and probabilities of the test_results dataframe to original test dataframe
test[CFG.target_cols] = test_results[CFG.target_cols]
test[CFG.prob_cols] = test_results[CFG.prob_cols]

# save results
if CFG.save_inference_csv_files:
    test.to_csv('datasets/test/inference_on_GRIP_samples.csv', index=False)
    test_results.to_csv('datasets/test/inference_on_GRIP_samples_no_metadata.csv', index=False)
