import os, torch
from dassl.data.datasets import DATASET_REGISTRY
from .base_dataset import read_data
from .wrapper import FASD_RGB, FASD_RGB_VAL
import random

def build_dataset(data_root, protocol):
    protocols = protocol.split('@')[0]
    train_name = protocol.split('@')[1]
    val_name = protocol.split('@')[2]
    test_name = protocol.split('@')[3]

    data_train = read_data(data_root, train_name, protocols, split='train')
    data_val = read_data(data_root, val_name, protocols, split='val')
    data_test = read_data(data_root, test_name, protocols, split='test')

    return data_train, data_val, data_test

# RandomSampler SequentialSampler RandomDomainSampler SeqDomainSampler RandomClassSampler
@DATASET_REGISTRY.register()
class UniAttackData:
    def __init__(self, cfg):
        train, val, test = build_dataset(cfg.DATASET.ROOT, cfg.DATASET.PROTOCOL)

        # Build data loader
        train_loader = torch.utils.data.DataLoader(
            FASD_RGB(cfg.MODEL.BACKBONE.NAME,
                    data_source=train,
                    image_size=cfg.INPUT.SIZE[0],
                    depth_size=cfg.INPUT.SIZE[0],
                    # modals=modals,
                    preprocess=cfg.DATASET.PREPROCESS),
                    batch_size=cfg.DATALOADER.TRAIN_X.BATCH_SIZE,
                    # sampler=sampler,
                    shuffle=True,
                    num_workers=cfg.DATALOADER.NUM_WORKERS,
                    drop_last=True,
                    pin_memory=False
                    )
        val_loader = torch.utils.data.DataLoader(
            FASD_RGB_VAL(data_source=val,
                    image_size=cfg.INPUT.SIZE[0],
                    preprocess='resize'),
                    batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,
                    # sampler=sampler,
                    shuffle=False,
                    num_workers=1,
                    drop_last=False,
                    pin_memory=False
                    )
        test_loader = torch.utils.data.DataLoader(
            FASD_RGB_VAL(data_source=test,
                    image_size=cfg.INPUT.SIZE[0],
                    preprocess='resize'),
                    batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,
                    # sampler=sampler,
                    shuffle=False,
                    num_workers=1,
                    drop_last=False,
                    pin_memory=False
                    )

        self.train_loader_x = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.lab2cname = {0: 'fake', 1: 'live'}
        self.classnames = ['fake', 'live']
        self.templates = [
            'This is an example of a {} face',
            'This is a {} face',
            'This is how a {} face looks like',
            'A photo of a {} face',
            'Is not this a {} face ?',
            'A printout shown to be a {} face'
        ]