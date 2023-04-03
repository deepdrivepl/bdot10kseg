import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import matplotlib
matplotlib.use("Agg")

import argparse
from collections import defaultdict
from tqdm import tqdm

import cv2
import torch
import numpy as np
import segmentation_models_pytorch as smp

from segmentation_models_pytorch import utils as smp_utils
from torch.utils import tensorboard


from bdot10kseg import BDOT10k
from aug import get_training_augmentation, get_preprocessing



def image2tb(image, output, writer, epoch, ds, name="train"):
    image = image[0,:,:,:].detach().cpu().numpy()
    output = output[0,:,:,:].detach().cpu().numpy()
    
    image = image.transpose(1,2,0)
    output = output.transpose(1,2,0)
    vis = ds.plot_sample(image, output, show=False)
    writer.add_image("%s" % name, torch.tensor(vis[:,:,:3]), epoch, dataformats="HWC")
    
    
    
def train(model, dl, loss, optimizer, scheduler, metrics, device, writer, out_dir, epoch, classes, ds):
    model.train()
    
    tbar = tqdm(dl)
    for i, data in enumerate(tbar):
        total_batch = epoch*len(dl)+i
        img, mask = data
        
        img, mask = img.to(device), mask.to(device)
        with torch.amp.autocast(device_type=device, dtype=torch.float16):
            output = model(img)
            batch_loss = loss(output, mask)
        
        writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], total_batch)
        writer.add_scalar("train/loss", batch_loss.cpu().detach().numpy(), total_batch)
        for metric in metrics:
            m_value = metric(output, mask).cpu().detach().numpy()
            writer.add_scalar("train/%s" % (metric.__name__), m_value, total_batch)
                
        for metric in metrics:
            for idx, cls in zip(range(output.shape[1]), classes):
                m_value = metric(output[:,idx,:,:], mask[:,idx,:,:]).cpu().detach().numpy()
                writer.add_scalar("train/%s_%s" % (metric.__name__, cls), m_value, total_batch)

        if i==0:
            image2tb(img, output, writer, total_batch, ds, name="train")  


        scaler.scale(batch_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        optimizer.zero_grad()
        scheduler.step()
    


@torch.no_grad()
def val(model, dl, loss, metrics, device, writer, out_dir, epoch, classes, ds):
    model.eval()
    
    epochm = defaultdict(list)
    total_batch = epoch*len(dl)
    
    tbar = tqdm(dl)

    for i, data in enumerate(tbar):
        img, mask = data
        img, mask = img.to(device), mask.to(device)
        output = model(img)
        
        batch_loss =  loss(output, mask)
        epochm["loss"].append(batch_loss.cpu().detach().numpy())
        
        for metric in metrics:
            m_value = metric(output, mask).cpu().detach().numpy()
            epochm[metric.__name__].append(m_value)
            
        for metric in metrics:
            for idx, cls in zip(range(output.shape[1]), classes):
                m_value = metric(output[:,idx,:,:], mask[:,idx,:,:]).cpu().detach().numpy()
                epochm[metric.__name__+"_%s"%cls].append(m_value)
                
        if i%10==0:
            image2tb(img, output, writer, total_batch, ds, name="val/%02d" % i)
            
    epochm = {k:np.mean(v) for k,v in epochm.items()}
    for k, v in epochm.items():
        writer.add_scalar("val/%s" %k, v, total_batch)
    
    return epochm




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_imgs", type=str, required=True)
    parser.add_argument("--val_imgs", type=str, required=True)
    parser.add_argument("--test_imgs", type=str, required=False)
    parser.add_argument("--npy_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="logs/000")
    parser.add_argument("--cat_csv", type=str, default="../data/BDOT10k-categories.csv")

    # general
    parser.add_argument("--img_size", type=int, default=1024)
    parser.add_argument("--bs_train", type=int, default=16)
    parser.add_argument("--bs_val", type=int, default=8)
    parser.add_argument("--bs_test", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--val_frq", type=int, default=2)

    # model
    parser.add_argument("--encoder", type=str, choices=["tu-efficientnet_b3", "resnet50"], default="tu-efficientnet_b3")
    parser.add_argument("--encoder_weights", type=str, default="imagenet")
    parser.add_argument("--activation", type=str, default="softmax") # could be None for logits or "softmax2d" for multiclass segmentation
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--pooling", type=str, default="avg")
    parser.add_argument("--mode", type=str, default="multilabel")
    parser.add_argument("--iou_th", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()


    scaler = torch.cuda.amp.GradScaler()
    preprocessing_fn = smp.encoders.get_preprocessing_fn(args.encoder, args.encoder_weights)
    writer = tensorboard.SummaryWriter(args.out_dir)


    # datasets
    ds_train = BDOT10k(
        tiff_list=[x.rstrip() for x in open(args.train_imgs)],
        npy_dir=args.npy_dir,
        bdot10k_cats_fname=args.cat_csv,
        size=args.img_size,
        transform=get_training_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn)
    )
    dl_train = torch.utils.data.DataLoader(ds_train, batch_size=args.bs_train, shuffle=True, num_workers=20)

    ds_val = BDOT10k(
         tiff_list=[x.rstrip() for x in open(args.val_imgs)],
         npy_dir=args.npy_dir,
         bdot10k_cats_fname=args.cat_csv,
         size=args.img_size,
         transform=None,
         preprocessing=get_preprocessing(preprocessing_fn)
    )
    dl_val = torch.utils.data.DataLoader(ds_val, batch_size=args.bs_val, shuffle=False, num_workers=20)

    # model
    model = smp.Unet(encoder_name=args.encoder, encoder_weights=args.encoder_weights, classes=ds_train.num_classes)
    model = model.to(args.device)

    # loss
    loss = smp.losses.DiceLoss(mode=args.mode)
    metrics = [
        smp_utils.metrics.IoU(threshold=args.iou_th),
    ]

    # optimizer
    optimizer = torch.optim.RAdam([ 
        dict(params=model.parameters(), lr=args.lr),
    ])

    # scheduler
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=len(dl_train), epochs=args.epochs,
        pct_start=0.1, div_factor=30, final_div_factor=3)


    # train
    best_loss = 10e7

    for i in range(0, args.epochs):
        print("EPOCH: %02d" % i)
        train(model, dl_train, loss, optimizer, scheduler, metrics, args.device, writer, args.out_dir, i, ds_train.bdot10k_cats, ds_train)
        
        if i%args.val_frq==0:
            epochm = val(model, dl_val, loss, metrics, args.device, writer, args.out_dir, i, ds_train.bdot10k_cats, ds_val)
            
            if epochm["loss"] < best_loss:
                best_loss = epochm["loss"]
                torch.save(model, os.path.join(args.out_dir, "./best_model_loss.pth"))


        torch.save(model, os.path.join(args.out_dir, "./model_%04d.pth" % i))
