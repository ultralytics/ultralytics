---
comments: true
description: Learn how to perform K-Fold cross validation on the Ultralytics object detection dataset with Python packages like Ultralytics, sklearn, pandas, and PyYAML.
keywords: K-Fold cross validation, Ultralytics, object detection, YOLO detection format, sklearn, pandas, PyYAML
---
​
# K-Fold cross validation for Ultralytics
​
## Introduction
​
- TODO
​
## Setup
​
- Annotations in [YOLO detection format](https://docs.ultralytics.com/datasets/detect/)
​
- Assumes annotation files are local
​
    - Tutorial uses [Fruit Detection](https://www.kaggle.com/datasets/lakshaytyagi01/fruit-detection/code) dataset
    
        - Total 8479 images
        
        - 6 class labels (shown with total instance counts)
​
            | Class Label | Instance Count |
            | :---------- | :------------: |
            |    Apple    |      7049      |
            |    Grapes   |      7202      |
            |  Pineapple  |      1613      |
            |    Orange   |      15549     |
            |    Banana   |      3536      |
            |  Watermelon |      1976      |
​
- Python packages required
​
    - Ultralytics
​
    - sklearn
​
    - pandas
​
    - PyYaml
​
- This tutorial uses `k=5` folds, but you will need to assess what number of folds works best for your dataset
​
1. Create and activate a new `venv` for your project and use `pip` (or your preferred package manager) to install:
​
1. `ultralytics` library using `pip install -U ultralytics` or by cloning the offical [repo](https://github.com/ultralytics/ultralytics)
​
1. Also scikit-learn, Pandas, and PyYAML `pip install -U scikit-learn pandas pyyaml`
​
1. Ensure your annotations are in [YOLO detection format](https://docs.ultralytics.com/datasets/detect/)
​
    - All annotation files for this example are located in directory `Fruit-Detection/labels`
​
​
## Generate Feature-Vectors for Object Detection Dataset
​
1. Create a new python file and import libraries
​
    ```py
    import datetime
    import shutil
    from pathlib import Path
    from collections import Counter
​
    import yaml
    import numpy as np
    import pandas as pd
    from ultralytics import YOLO
    from sklearn.model_selection import KFold
    ```
​
1. Next, get all label files for your dataset
​
    ```py
    dataset_path = Path('./Fruit-detection') # use 'path/to/dataset' for your custom data
    labels = sorted(dataset_path.rglob("*labels/*.txt")) # all data in 'labels'
    ```
​
1. Read contents of dataset YAML file and get indices of class labels
​
    ```py
    with open(yaml_file, 'r', encoding="utf8") as y:
        classes = yaml.safe_load(y)['names']
    cls_idx = sorted(classes.keys())
    ```
​
1. Create empty `pandas` Dataframe
​
    ```py
    indx = [l.stem for l in labels] # uses base filename as ID (no extension)
    labels_df = pd.DataFrame([], columns=cls_idx, index=indx)
    ```
​
1. Count instances of each class-label present in annotation files
​
    ```py
    for label in labels:
        lbl_counter = Counter()
​
        with open(label,'r') as lf:
            lines = lf.readlines()
​
        for l in lines:
            # classes for YOLO label uses integer at first position of each line
            lbl_counter[int(l.split(' ')[0])] += 1
​
        labels_df.loc[label.stem] = lbl_counter
​
    labels_df = labels_df.fillna(0.0) # replace `nan` values with `0.0`
    ```
​
1. Here is a preview of the populated DataFrame
​
    ```ps
                                                           0    1    2    3    4    5
    '0000a16e4b057580_jpg.rf.00ab48988370f64f5ca8ea4...'  0.0  0.0  0.0  0.0  0.0  7.0
    '0000a16e4b057580_jpg.rf.7e6dce029fb67f01eb19aa7...'  0.0  0.0  0.0  0.0  0.0  7.0
    '0000a16e4b057580_jpg.rf.bc4d31cdcbe229dd022957a...'  0.0  0.0  0.0  0.0  0.0  7.0
    '00020ebf74c4881c_jpg.rf.508192a0a97aa6c4a3b6882...'  0.0  0.0  0.0  1.0  0.0  0.0
    '00020ebf74c4881c_jpg.rf.5af192a2254c8ecc4188a25...'  0.0  0.0  0.0  1.0  0.0  0.0
     ...                                                  ...  ...  ...  ...  ...  ...
    'ff4cd45896de38be_jpg.rf.c4b5e967ca10c7ced3b9e97...'  0.0  0.0  0.0  0.0  0.0  2.0
    'ff4cd45896de38be_jpg.rf.ea4c1d37d2884b3e3cbce08...'  0.0  0.0  0.0  0.0  0.0  2.0
    'ff5fd9c3c624b7dc_jpg.rf.bb519feaa36fc4bf630a033...'  1.0  0.0  0.0  0.0  0.0  0.0
    'ff5fd9c3c624b7dc_jpg.rf.f0751c9c3aa4519ea3c9d6a...'  1.0  0.0  0.0  0.0  0.0  0.0
    'fffe28b31f2a70d4_jpg.rf.7ea16bd637ba0711c53b540...'  0.0  6.0  0.0  0.0  0.0  0.0
    ```
​
Rows index the label files (corresponding to each image in your dataset), with columns corresponding to your class-label indices. Each row is a pseudo feature-vector with the instance count of each class-label present in your dataset. This is what allows for K-Fold cross validation to be performed on an object detection dataset.
​
## K-Fold Dataset Split
​
1. Now using the `sklearn.model_selection.KFold`class, generate `k` splits of the dataset
​
    - NOTE:
​
        - Enable `shuffle=True` to ensure a mixed distribution of classes in your splits
​
        - Use `random_state=M` where `M` is an integer of your choice, this allows for repeatable results
​
    ```py
    ksplit = 5
    kf = KFold(n_splits=ksplit,
                shuffle=True,     # shuffles data for improved distribution of classes
                random_state=20   # random state seed
                )
    result = kf.split(labels_df,
                        y=None,
                        groups=None
                        )
    kfolds = dict()
    for i,(k_train,k_val) in enumerate(result):
        kfolds[i] = {'train':k_train, 'val':k_val}
    ```
​
1. Dataset has now been split into `k` folds with a list of `train` and `val` indexes, build a 'report' DataFrame to view the results
​
    ```py
    folds = [f'split_{n}' for n in range(1,ksplit + 1)]
    folds_df = pd.DataFrame([],
                            columns=folds,
                            index=indx
                            )
​
    for n,k in enumerate(kfolds,start=1):
        folds_df[f'split_{n}'].loc[labels_df.iloc[kfolds[k]['train']].index.to_series()] = 'train'
        folds_df[f'split_{n}'].loc[labels_df.iloc[kfolds[k]['val']].index.to_series()] = 'val'
    ```
​
    - Result:
​
    ```ps
                                                         split_1  split_2  split_3  split_4  split_5
    '0000a16e4b057580_jpg.rf.00ab48988370f64f5ca8ea4...'     val    train    train    train    train
    '0000a16e4b057580_jpg.rf.7e6dce029fb67f01eb19aa7...'   train    train    train    train      val
    '0000a16e4b057580_jpg.rf.bc4d31cdcbe229dd022957a...'   train    train    train      val    train
    '00020ebf74c4881c_jpg.rf.508192a0a97aa6c4a3b6882...'   train      val    train    train    train
    '00020ebf74c4881c_jpg.rf.5af192a2254c8ecc4188a25...'   train    train    train    train      val
    ...                                                      ...      ...      ...      ...      ...
    'ff4cd45896de38be_jpg.rf.c4b5e967ca10c7ced3b9e97...'     val    train    train    train    train
    'ff4cd45896de38be_jpg.rf.ea4c1d37d2884b3e3cbce08...'   train    train    train    train      val
    'ff5fd9c3c624b7dc_jpg.rf.bb519feaa36fc4bf630a033...'   train    train    train    train      val
    'ff5fd9c3c624b7dc_jpg.rf.f0751c9c3aa4519ea3c9d6a...'   train    train      val    train    train
    'fffe28b31f2a70d4_jpg.rf.7ea16bd637ba0711c53b540...'   train    train      val    train    train
    ```
​
1. Calculate class label distribution for each fold as a ratio of classes present in `val` and `train` 
​
    ```py
    fold_lbl_distrb = pd.DataFrame([],columns=cols,index=folds)
​
    for n,k in enumerate(kfolds,start=1):
        val_totals = np.array([labels_df.iloc[kfolds[k]['val']][c].sum() for c in labels_df.iloc[kfolds[k]['val']].columns])
        train_totals = np.array([labels_df.iloc[kfolds[k]['train']][c].sum() for c in labels_df.iloc[kfolds[k]['train']].columns])
        
        ratio = val_totals / (train_totals + 1E-7)
        fold_lbl_distrb.loc[f'split_{n}'] = ratio
    ```
​
    ### Class Ratio (val / train) per fold
​
    ```ps
                    0         1         2         3         4         5
    split_1  0.277919  0.220152  0.268181  0.198012  0.213695  0.281453
    split_2  0.246508  0.262857  0.204549  0.288129  0.315661  0.188214
    split_3  0.224422  0.246387   0.23048   0.29424  0.306073  0.261814
    split_4  0.275145  0.254791  0.242367  0.233069  0.211871  0.286458
    split_5  0.228049  0.266929  0.309455  0.241735  0.211871   0.23732
    ```
​
    - Ideally all class ratios will be _reasonably_ close for each split and across classes, but this will be subjective to your specific dataset.
    <p></p>
​
1. Create the directories and dataset YAML files for each split
​
    ```py
    # Create new directory to move/copy files based on splits
    save_path = Path(dataset_path.as_posix() + f'/{datetime.date.today().isoformat()}_{ksplit}-Fold_Cross-val')
    _ = save_path.mkdir()
    # Get all images
    image_path = Path(dataset_path.as_posix() + '/images')
    images = sorted(image_path.glob("*.jpg")) # update with correct file extension as needed
​
    # Create directories and dataset YAML files
    ds_yamls = list()
    for split in folds_df.columns:
        # New directories
        n_split = Path(save_path.as_posix() + f'/{split}')
        _ = n_split.mkdir()
        _ = [Path(n_split.as_posix() + f'/{sub}').mkdir(parents=True) for sub in ['train/images','train/labels','val/images','val/labels']]
        # Dataset YAML file
        dataset_yaml = n_split.as_posix() + f'/{split}_dataset.yaml'
        ds_yamls.append(ds_yamls)
        
        with open(dataset_yaml,'w') as ds_y:
            yaml.safe_dump(
                                {
                                'path':save_path.as_posix(),
                                'train':'/train',
                                'val':'/val',
                                'names':classes
                                }
                                ,ds_y
                            )
    ```
​
1. Copy images and labels into the appropriate directory ('train' or 'val') for each split
​
    - __NOTE:__ The time required for this portion of the code will vary based on the size of your dataset and your system hardware
​
    ```py
    for ii,image in enumerate(images):
        label = labels[ii] if labels[ii].stem == image.stem else [l for l in labels if l.stem == image.stem][0]
        
        for split,k_split in folds_df.loc[image.stem].items():
            # Path directory to copy into
            img_to_path = save_path.as_posix() + '/'.join(['',split,k_split,'images'])
            lbl_to_path = save_path.as_posix() + '/'.join(['',split,k_split,'labels'])
        
            # Copy image and label file into new path directory 
            # NOTE may throw SamefileError if file already exists
            _ = shutil.copy(image.as_posix(), img_to_path + f'/{image.name}')
            _ = shutil.copy(label.as_posix(), lbl_to_path + f'/{label.name}')
    ```
​
## Save Records (Optional)
​
1. Save records of K-Fold split and Label distribution DataFrames as CSV files
​
```py
folds_df.to_csv(save_path.as_posix() + "/kfold_datasplit.csv")
fold_lbl_distrb.to_csv(save_path.as_posix() + "/kfold_label_distribution.csv")
```
​
## Train YOLO using K-Fold datasplits
​
1. Load YOLO model
​
    ```py
    weights = 'path/to/weights.pt'
    m = YOLO(weights,task='detect')
    ```
​
1. Loop over dataset YAML files to run training, results will be saved to directory as specified by `project` and `name` kwargs, (default is 'exp/runs#' where # is an integer index)
​
    ```py
    results = dict()
    for k in range(ksplit):
        k_yaml = ds_yamls[k]
        m.train(data=k_yaml,
                            args,   # Include any train args
                            kwargs, # Include any train kwargs
                            )
        results[k] = m.metrics # save output for any calculations
    ```
