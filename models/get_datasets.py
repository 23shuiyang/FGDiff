import numpy as np
import torch
import data

def get_datasets(args):
    if args.dataset == 'salicon':
        args.output_size = [480, 640]
        args.input_size = 384
        train_dataset = data.SaliconDataset(args.dataset_dir, train=True, input_size_h=args.input_size,
                                            input_size_w=args.input_size)
        val_dataset = data.SaliconDataset(args.dataset_dir, train=False, input_size_h=args.input_size,
                                          input_size_w=args.input_size)

    elif args.dataset == 'mit1003':
        args.output_size = [384, 384]
        args.input_size = 384
        train_dataset = data.Mit1003Dataset(args.dataset_dir, train=True, input_size_h=args.input_size,
                                            input_size_w=args.input_size)
        val_dataset = data.Mit1003Dataset(args.dataset_dir, train=False, input_size_h=args.input_size,
                                          input_size_w=args.input_size)

    elif args.dataset == 'cat2000':
        args.output_size = [384, 384]
        args.input_size = 384
        train_dataset = data.CAT2000Dataset(args.dataset_dir, train=True, input_size_h=args.input_size,
                                            input_size_w=args.input_size)
        val_dataset = data.CAT2000Dataset(args.dataset_dir, train=False, input_size_h=args.input_size,
                                          input_size_w=args.input_size)

    elif args.dataset == 'pascals':
        args.output_size = [384, 384]
        args.input_size = 384
        train_dataset = data.PASCALSDataset(args.dataset_dir, train=True, input_size_h=args.input_size,
                                            input_size_w=args.input_size)
        val_dataset = data.PASCALSDataset(args.dataset_dir, train=False, input_size_h=args.input_size,
                                          input_size_w=args.input_size)

    elif args.dataset == 'osie':
        args.output_size = [384, 384]
        args.input_size = 384
        train_dataset = data.OSIEDataset(args.dataset_dir, train=True, input_size_h=args.input_size,
                                         input_size_w=args.input_size)
        val_dataset = data.OSIEDataset(args.dataset_dir, train=False, input_size_h=args.input_size,
                                       input_size_w=args.input_size)

    elif args.dataset == 'dutomron':
        args.output_size = [480, 640]
        args.input_size = 384
        train_dataset = data.DUTOMRONDataset(args.dataset_dir, train=True, input_size_h=args.input_size,
                                             input_size_w=args.input_size)
        val_dataset = data.DUTOMRONDataset(args.dataset_dir, train=False, input_size_h=args.input_size,
                                           input_size_w=args.input_size)

    elif args.dataset == 'fiwi':
        args.output_size = [384, 384]
        args.input_size = 384
        train_dataset = data.FIWIDataset(args.dataset_dir, train=True, input_size_h=args.input_size,
                                         input_size_w=args.input_size)
        val_dataset = data.FIWIDataset(args.dataset_dir, train=False, input_size_h=args.input_size,
                                       input_size_w=args.input_size)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.no_workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                             num_workers=args.no_workers, pin_memory=True)
    #train_loader = val_loader
    return train_loader, val_loader
