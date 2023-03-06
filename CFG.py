import numpy as np

class CFG:
    target_cols = ['camp', 'corylus', 'dust', 'grim', 'qrob', 'qsub', 'cont']
    target_size = len(target_cols)
    prob_cols = ['p_' + i for i in target_cols]
    cols_mva = ['Area (ABD)', 'Area (Filled)', 'Aspect Ratio', 'Biovolume (Cylinder)',
                'Biovolume (P. Spheroid)', 'Circle Fit',
                'Circularity', 'Circularity (Hu)', 'Compactness', 'Convex Perimeter',
                'Convexity', 'Diameter (ABD)', 'Diameter (ESD)', 'Edge Gradient',
                'Elongation', 'Feret Angle Max', 'Feret Angle Min', 'Fiber Curl',
                'Fiber Straightness', 'Geodesic Aspect Ratio', 'Geodesic Length',
                'Geodesic Thickness', 'Intensity', 'Length', 'Particles Per Chain',
                'Perimeter', 'Roughness', 'Sigma Intensity', 'Sum Intensity',
                'Symmetry', 'Transparency', 'Volume (ABD)', 'Volume (ESD)', 'Width']

    size = 128
    n_fold = 1
    num_workers = 8
    batch_size = 512
    model_name = 'resnet18'
    if_pretrained = True

    lr = 1e-4
    epochs = 40

    run_umap_test = True

    save_model = True
    load_model = False
    model_name_saved = 'ICELEARNING_net'
    OUTPUT_DIR = 'saved_model/'
    save_conf_matrix = 'confusion_matrix_test_dataset.pdf'

    save_inference_csv_files = True