import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset, Dataset

from datasets.base import CombinedDataSet
from datasets.visda import CustomVisdaTarget, CustomVisdaSource

DATA_SETS = {
    CustomVisdaTarget.code(): CustomVisdaTarget,
    CustomVisdaSource.code(): CustomVisdaSource,
}


Source_train = pd.read_csv("/content/drive/MyDrive/DTA/data/Source_train.csv")
Source_test = pd.read_csv("/content/drive/MyDrive/DTA/data/Source_test.csv")
Target_train = pd.read_csv("/content/drive/MyDrive/DTA/data/Target_train.csv")
Target_test = pd.read_csv("/content/drive/MyDrive/DTA/data/Target_test.csv")



class PytorchDataSet(Dataset):
    
    def __init__(self, df):
        FEATURES = list(i for i in df.columns if i!= 'labels')
        TARGET = "labels"

        from sklearn.preprocessing import StandardScaler
        Normarizescaler = StandardScaler()
        Normarizescaler.fit(np.array(df[FEATURES]))
        
        # for test data, In test data, it's easier to fill it with something on purpose.
        
        if "labels" not in df.columns:
            df["labels"] = 9999
        
        self.df = df
        
        self.train_X = np.array(self.df[FEATURES])
        self.train_Y = np.array(self.df[TARGET])
        self.train_X = Normarizescaler.transform(self.train_X)
        
        
        self.train_X = torch.from_numpy(self.train_X).float()
        self.train_Y = torch.from_numpy(self.train_Y).long()
    
    def __len__(self):
        
        return len(self.df)
    
    def __getitem__(self, idx):
        
        return self.train_X[idx], self.train_Y[idx]

Source_train = PytorchDataSet(Source_train)
Source_test = PytorchDataSet(Source_test)
Target_train = PytorchDataSet(Target_train)
Target_test = PytorchDataSet(Target_test)

'''
def dataset_factory(dataset_code, transform_type, is_train=True, **kwargs):
    cls = DATA_SETS[dataset_code]
    transform = cls.train_transform_config(transform_type) if is_train else cls.eval_transform_config(transform_type)

    print("{} has been created.".format(cls.code()))
    return cls(transform=transform, **kwargs)
'''

def dataloaders_factory(args):
    source_train_dataset = Source_train
    target_train_dataset = Target_train

    train_dataset = CombinedDataSet(source_train_dataset, target_train_dataset)
    target_val_dataset = Target_test

    if args.test:
        train_dataset = Subset(train_dataset, np.random.randint(0, len(train_dataset), args.batch_size * 5))
        target_val_dataset = Subset(target_val_dataset,
                                    np.random.randint(0, len(target_val_dataset), args.batch_size * 5))

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                  pin_memory=True)

    target_val_dataloader = DataLoader(target_val_dataset,
                                       batch_size=args.batch_size, shuffle=False, pin_memory=True)
    return {
        'train': train_dataloader,
        'val': target_val_dataloader
    }
