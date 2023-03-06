import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import copy
import os
import cv2
import time

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from umap import UMAP

from CFG import CFG
from data import ParticleDataset, get_transforms
from model import CustomModel, Hook
from functs import AverageMeter, train_fn, valid_fn
from utils import plot_confusion_matrix

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler

# # ====================================================
# Create train / val / test datasets
# # ====================================================
camp = pd.read_csv('datasets/camp.csv').sample(frac=1)
grim = pd.read_csv('datasets/grim.csv').sample(frac=1)
dust = pd.read_csv('datasets/dust.csv').sample(frac=1)
qrob = pd.read_csv('datasets/qrob.csv').sample(frac=1)
qsub = pd.read_csv('datasets/qsub.csv').sample(frac=1)
corylus = pd.read_csv('datasets/corylus.csv').sample(frac=1)
cont = pd.read_csv('datasets/cont.csv').sample(frac=1)

Nval = 500
Ntest = 500

camp_val, camp_test, camp_train = camp[:Nval], camp[Nval:Nval+Ntest], camp[Nval+Ntest:]

corylus_val, corylus_test, corylus_train = corylus[:Nval], corylus[Nval:Nval+Ntest], corylus[Nval+Ntest:]

dust_val, dust_test, dust_train = dust[:Nval], dust[Nval:Nval+Ntest], dust[Nval+Ntest:].sample(8000)

grim_val, grim_test, grim_train = grim[:Nval], grim[Nval:Nval+Ntest], grim[Nval+Ntest:]

qrob_val, qrob_test, qrob_train = qrob[:Nval], qrob[Nval:Nval+Ntest], qrob[Nval+Ntest:]

qsub_val, qsub_test, qsub_train = qsub[:Nval], qsub[Nval:Nval+Ntest], qsub[Nval+Ntest:]

cont_val, cont_test, cont_train = cont[:Nval], cont[Nval:Nval+Ntest], cont[Nval+Ntest:]

train_label_sizes = [len(i) for i in (camp_train, corylus_train, dust_train, grim_train, qrob_train, qsub_train, cont_train)]
weights = np.max(train_label_sizes) / train_label_sizes
print(f'Train dataset sizes {train_label_sizes}')

train = pd.concat([camp_train, corylus_train, dust_train, grim_train, qrob_train, qsub_train, cont_train], ignore_index = True)
val = pd.concat([camp_val, corylus_val, dust_val, grim_val, qrob_val, qsub_val, cont_val], ignore_index = True)
test = pd.concat([camp_test, corylus_test, dust_test, grim_test, qrob_test, qsub_test, cont_test], ignore_index = True)

print(f'Train dataset: {len(train)} items')
print(f'Val dataset: {len(val)} items')
print(f'Test dataset: {len(test)} items')

train_meta = copy.deepcopy(train)
val_meta = copy.deepcopy(val)
test_meta = copy.deepcopy(test)

# Scaler
scaler = StandardScaler()

train_meta[CFG.cols_mva] = scaler.fit_transform(train_meta[CFG.cols_mva])
val_meta[CFG.cols_mva] = scaler.transform(val_meta[CFG.cols_mva])
test_meta[CFG.cols_mva] = scaler.transform(test_meta[CFG.cols_mva])

# ====================================================
# model & optimizer
# ====================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'working with {device}')

model = CustomModel(CFG.model_name, pretrained=CFG.if_pretrained)
model.to(device)

optimizer = AdamW(model.parameters(), lr=CFG.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)

if CFG.load_model:
    # first, load checkpoint in GPU
    checkpoint = torch.load(CFG.OUTPUT_DIR+f'{CFG.model_name_saved}.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #epoch = checkpoint['epoch_state_dict']

# ====================================================
# Train/val loop
# ====================================================
def train_loop(model, optimizer):

    train_dataset = ParticleDataset(train_meta, transform=get_transforms(data='train'))
    valid_dataset = ParticleDataset(val_meta, transform=get_transforms(data='valid'))

    train_loader = DataLoader(train_dataset, batch_size=CFG.batch_size, shuffle=True,
                              num_workers=CFG.num_workers, pin_memory=True, drop_last=True)

    valid_loader = DataLoader(valid_dataset, batch_size=CFG.batch_size, shuffle=True,
                              num_workers=CFG.num_workers, pin_memory=True, drop_last=True)

    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    classweights = torch.Tensor(weights).to(device)

    criterion = nn.BCEWithLogitsLoss(weight=classweights)

    # ====================================================
    # loop
    # ====================================================

    print('Start training loop.')
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    train_loss_history = []
    train_acc_history = []

    val_loss_history = []
    val_acc_history = []

    for epoch in range(CFG.epochs):
        print(f'-' * 10)
        print(f'Epoch {epoch}/{CFG.epochs - 1}')

        # train
        train_epoch_loss_history, train_epoch_acc_history = train_fn(train_loader, model, criterion, optimizer,
                                                                     scheduler, device)

        train_loss_history.append(np.mean(train_epoch_loss_history))
        train_acc_history.append(np.mean(train_epoch_acc_history))

        # val
        val_epoch_loss_history, val_epoch_acc_history = valid_fn(valid_loader, model, criterion, device)

        val_loss_history.append(np.mean(val_epoch_loss_history))
        val_acc_history.append(np.mean(val_epoch_acc_history))

        # If epoch validation accuracy has improved, deep copy the model
        if np.mean(val_epoch_acc_history) > best_acc:
            best_acc = np.mean(val_epoch_acc_history)
            best_model_wts = copy.deepcopy(model.state_dict())


    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

    # load best model weights
    model.load_state_dict(best_model_wts)

    # save model
    if CFG.save_model is True:
        if os.path.isdir(CFG.OUTPUT_DIR): None
        else: os.makedirs(CFG.OUTPUT_DIR)

        torch.save({
            'epoch_state_dict': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, CFG.OUTPUT_DIR + f'{CFG.model_name_saved}.pth')

    return train_loss_history, train_acc_history, val_loss_history, val_acc_history, model


# train
train_loss_history, train_acc_history, val_loss_history, val_acc_history, model = train_loop(model, optimizer)

model_train_val_loss_acc = pd.DataFrame({'epoch': list(range(CFG.epochs)),
                                        'train_loss':train_loss_history,
                                        'train_acc':train_acc_history,
                                        'val_loss':val_loss_history,
                                        'val_acc':val_acc_history,
                                         })
model_train_val_loss_acc.to_csv('model_training_performance.csv', index=False)


# ====================================================
# Test loop.
# Test dataset = 500 items/class
# ====================================================
def test_loop(model, test_meta):

    print('Start test loop.')
    start = time.time()
    model.eval()

    hookfeatures = []

    # Initialize val dataset and dataloader
    test_dataset = ParticleDataset(test_meta, transform=get_transforms(data='valid'))
    test_loader = DataLoader(test_dataset, batch_size=CFG.batch_size, shuffle=True,
                              num_workers=CFG.num_workers, pin_memory=True, drop_last=False)

    # Initialize a dictionary of results
    results_test = {'paths': [], 'preds': np.array([]), 'labels': np.array([])}

    with torch.no_grad():
        for i, (images, labels, paths, xfeatures) in enumerate(test_loader):
            images = images.to(device)
            labels = labels.to(device)
            xfeatures = xfeatures.to(device)

            y_preds = model(images, xfeatures)
            preds = (y_preds == y_preds.max(dim=1, keepdim=True)[0]).to(dtype=torch.int32)

            pred_indexes = torch.max(preds, 1)[1].cpu().numpy()
            label_indexes = torch.max(labels, 1)[1].cpu().numpy()

            hook_emb = hook[0].output.detach().cpu().numpy()  # (CFG.batch_size, 64)
            hookfeatures.append(hook_emb)

            # Fill dictionary with validation dataset results
            results_test['paths'] += paths
            results_test['preds'] = np.concatenate((results_test['preds'], pred_indexes), axis=0)
            results_test['labels'] = np.concatenate((results_test['labels'], label_indexes), axis=0)

    results_test['hookfeatures'] = np.vstack(hookfeatures).tolist() # (3500, 64)

    print(f'Test complete in {time.time() - start:.2f}s')

    return results_test


# Initialize layers to be monitored: FC layer of the resnet network
hook = [Hook(name, layer, backward=False) for name, layer in model.named_modules() if name=='base.fc']
print(f'How many layers we are monitoring: {len(hook)}')
print(f'We are monitoring: {hook[0].name}')

# Test loop and save results
results_test = test_loop(model, test_meta)
results_test_df = pd.DataFrame.from_dict(results_test)
results_test_df.to_csv ('model_test_results.csv', index=False, header=True)

# Confusion Matrix
cm_orig = confusion_matrix(results_test['labels'], results_test['preds'])
# we shuffle the confusion matrix according to labels list
cm = confusion_matrix(results_test['labels'], results_test['preds'], labels=[2,0,3,1,4,5,6])

accuracy = np.trace(cm) / float(np.sum(cm))
misclass = 1 - accuracy
print(f'Confusion matrix: \n {cm}')
print(f'Test accuracy: {accuracy:.4f}')

# Plot Confusion Matrix
ax = plot_confusion_matrix(cm,
                      target_names = ['Dust', 'Tephra F.', 'Tephra B.', 'C. avellana', 'Q. robur', 'Q. suber', 'Contam.'],
                      axis_titles = None,
                      title        = '',
                      normalize    = True,
                      savefig = CFG.save_conf_matrix
                      )
plt.show()

# UMAP on the hook embeddings
if CFG.run_umap_test:
    umap = UMAP(n_components=2, set_op_mix_ratio=.8, n_neighbors=25, min_dist=0.6, init='random', random_state=0)
    X_2d_umap_test = umap.fit_transform(results_test['hookfeatures']) # (3500, 2)

    target_ids = [0, 1, 2, 3, 4, 5, 6]
    colors = ['cyan', 'lime', 'peru', 'k', 'darkgreen', 'mediumseagreen', 'blue']

    # plot
    fig, ax1 = plt.subplots(1, 1, figsize=(6, 6))

    for i, c, label in zip(target_ids, colors, CFG.target_cols):
        ax1.scatter(X_2d_umap_test[results_test['labels'] == i, 0],
                    X_2d_umap_test[results_test['labels'] == i, 1],
                    c=c, s=8, alpha=0.5, label=label, zorder=2)

    ax1.set_xlabel('UMAP1', fontsize=14)
    ax1.set_ylabel('UMAP2', fontsize=14)

    ax1.set_xlim(-15, 23)
    ax1.set_ylim(-15, 23)

    ax1.grid(True, zorder=1)

    legend_elements = [Line2D([0], [0], marker='o', c='w', label='Dust', mfc='peru', ms=10),
                       Line2D([0], [0], marker='o', c='w', label='Tephra F.', mfc='cyan', ms=10),
                       Line2D([0], [0], marker='o', c='w', label='Tephra B.', mfc='k', ms=10),
                       Line2D([0], [0], marker='o', c='w', label='Contam/Blurry', mfc='blue', ms=10),
                       Line2D([0], [0], marker='o', c='w', label='Corylus Av.', mfc='lime', ms=10),
                       Line2D([0], [0], marker='o', c='w', label='Quercus R.', mfc='darkgreen', ms=10),
                       Line2D([0], [0], marker='o', c='w', label='Quercus S.', mfc='mediumseagreen', ms=10)
                       ]

    leg = ax1.legend(handles=legend_elements)
    leg.set_title('Test set (N=3500)', prop={'size': 12, 'weight': 'bold'})

    plt.savefig('umap_test.pdf', bbox_inches='tight', pad_inches=.1)
    plt.show()


