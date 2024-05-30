# import argparse
# import logging
# import os
# import time

# import torch
# import torch.distributed as dist
# import torch.nn.functional as F
# from torch.nn.parallel.distributed import DistributedDataParallel
# import torch.utils.data.distributed
# from torch.nn.utils import clip_grad_norm_
# from torch.nn import CrossEntropyLoss

# from utils import losses
# from config.config import config as cfg
# # from utils.dataset import FaceDatasetFolder, DataLoaderX
# from utils.dataset_celebA import FaceDatasetFolder, DataLoaderX
# from utils.utils_callbacks import CallBackVerification, CallBackLogging, CallBackModelCheckpoint
# from utils.utils_logging import AverageMeter, init_logging

# from backbones.iresnet import iresnet100, iresnet50


# torch.backends.cudnn.benchmark = True

# def main(args):
#     local_rank = 0
#     rank = 0
#     #torch.cuda.set_device(local_rank)

#     if not os.path.exists(cfg.output):
#         os.makedirs(cfg.output)
#     else:
#         time.sleep(2)

#     log_root = logging.getLogger()
#     init_logging(log_root, rank, cfg.output)

#     trainset = FaceDatasetFolder(root_dir='/media/statlab/SeagateHDD/Fateme Tavakoli/ElasticFace/ds/celebA', local_rank=local_rank)

#     train_loader = DataLoaderX(
#         local_rank=local_rank, dataset=trainset, batch_size=cfg.batch_size,
#         num_workers=0, pin_memory=True, drop_last=True)

#     # load model
#     if cfg.network == "iresnet100":
#         backbone = iresnet100(num_features=cfg.embedding_size, use_se=cfg.SE).to(local_rank)
#     elif cfg.network == "iresnet50":
#         backbone = iresnet50(dropout=0.4,num_features=cfg.embedding_size, use_se=cfg.SE).to(local_rank)
#     else:
#         backbone = None
#         logging.info("load backbone failed!")
#         exit()

#     if args.resume:
#         try:
#             backbone_pth = os.path.join(cfg.output, str(cfg.global_step) + "backbone.pth")
#             backbone.load_state_dict(torch.load(backbone_pth, map_location=torch.device(local_rank)))

#             if rank == 0:
#                 logging.info("backbone resume loaded successfully!")
#         except (FileNotFoundError, KeyError, IndexError, RuntimeError):
#             logging.info("load backbone resume init, failed!")

#     backbone.train()

#     # get header
#     if cfg.loss == "ElasticArcFace":
#         header = losses.ElasticArcFace(in_features=cfg.embedding_size, out_features=cfg.num_classes, s=cfg.s, m=cfg.m,std=cfg.std).to(local_rank)
#     elif cfg.loss == "ElasticArcFacePlus":
#         header = losses.ElasticArcFace(in_features=cfg.embedding_size, out_features=cfg.num_classes, s=cfg.s, m=cfg.m,
#                                        std=cfg.std, plus=True).to(local_rank)
#     elif cfg.loss == "ElasticCosFace":
#         header = losses.ElasticCosFace(in_features=cfg.embedding_size, out_features=cfg.num_classes, s=cfg.s, m=cfg.m,std=cfg.std).to(local_rank)
#     elif cfg.loss == "ElasticCosFacePlus":
#         header = losses.ElasticCosFace(in_features=cfg.embedding_size, out_features=cfg.num_classes, s=cfg.s, m=cfg.m,
#                                        std=cfg.std, plus=True).to(local_rank)
#     elif cfg.loss == "ArcFace":
#         header = losses.ArcFace(in_features=cfg.embedding_size, out_features=cfg.num_classes, s=cfg.s, m=cfg.m).to(local_rank)
#     elif cfg.loss == "CosFace":
#         header = losses.CosFace(in_features=cfg.embedding_size, out_features=cfg.num_classes, s=cfg.s, m=cfg.m).to(
#             local_rank)
#     else:
#         print("Header not implemented")
#     if args.resume:
#         try:
#             header_pth = os.path.join(cfg.output, str(cfg.global_step) + "header.pth")
#             header.load_state_dict(torch.load(header_pth, map_location=torch.device(local_rank)))

#             if rank == 0:
#                 logging.info("header resume loaded successfully!")
#         except (FileNotFoundError, KeyError, IndexError, RuntimeError):
#             logging.info("header resume init, failed!")
    

#     header.train()

#     opt_backbone = torch.optim.SGD(
#         params=[{'params': backbone.parameters()}],
#         lr=cfg.lr / 512 * cfg.batch_size,
#         momentum=0.9, weight_decay=cfg.weight_decay)
#     opt_header = torch.optim.SGD(
#         params=[{'params': header.parameters()}],
#         lr=cfg.lr / 512 * cfg.batch_size,
#         momentum=0.9, weight_decay=cfg.weight_decay)

#     scheduler_backbone = torch.optim.lr_scheduler.LambdaLR(
#         optimizer=opt_backbone, lr_lambda=cfg.lr_func)
#     scheduler_header = torch.optim.lr_scheduler.LambdaLR(
#         optimizer=opt_header, lr_lambda=cfg.lr_func)        

#     criterion = CrossEntropyLoss()

#     start_epoch = 0
#     total_step = int(len(trainset) / cfg.batch_size * cfg.num_epoch)
#     if rank == 0: logging.info("Total Step is: %d" % total_step)

#     if args.resume:
#         rem_steps = (total_step - cfg.global_step)
#         cur_epoch = cfg.num_epoch - int(cfg.num_epoch / total_step * rem_steps)
#         logging.info("resume from estimated epoch {}".format(cur_epoch))
#         logging.info("remaining steps {}".format(rem_steps))
        
#         start_epoch = cur_epoch
#         scheduler_backbone.last_epoch = cur_epoch
#         scheduler_header.last_epoch = cur_epoch

#         # --------- this could be solved more elegant ----------------
#         opt_backbone.param_groups[0]['lr'] = scheduler_backbone.get_lr()[0]
#         opt_header.param_groups[0]['lr'] = scheduler_header.get_lr()[0]

#         print("last learning rate: {}".format(scheduler_header.get_lr()))
#         # ------------------------------------------------------------

#     callback_verification = CallBackVerification(cfg.eval_step, rank, cfg.val_targets, cfg.rec)
#     callback_logging = CallBackLogging(50, rank, total_step, cfg.batch_size, 1, writer=None)
#     callback_checkpoint = CallBackModelCheckpoint(rank, cfg.output)

#     loss = AverageMeter()
#     global_step = cfg.global_step
#     for epoch in range(start_epoch, cfg.num_epoch):
#         for _, (img, label) in enumerate(train_loader):
#             ##
#             global_step += 1
#             img = img.cuda(local_rank, non_blocking=True)
#             label = label.cuda(local_rank, non_blocking=True)

#             features = F.normalize(backbone(img))

#             thetas = header(features, label)
#             loss_v = criterion(thetas, label)
#             loss_v.backward()

#             clip_grad_norm_(backbone.parameters(), max_norm=5, norm_type=2)

#             opt_backbone.step()
#             opt_header.step()

#             opt_backbone.zero_grad()
#             opt_header.zero_grad()

#             loss.update(loss_v.item(), 1)
            
#             callback_logging(global_step, loss, epoch)
#             callback_verification(global_step, backbone)

#         scheduler_backbone.step()
#         scheduler_header.step()

#         callback_checkpoint(global_step, backbone, header)



# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description='PyTorch margin penalty loss  training')
#     #parser.add_argument('--local_rank', type=int, default=0, help='local_rank')
#     parser.add_argument('--resume', type=int, default=0, help="resume training")
#     args_ = parser.parse_args()
#     main(args_)
import os 
import glob
import tarfile
import shutil
import random

import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as tt
from torchvision.utils import make_grid
import torch.nn as nn
import torch.nn.functional as F
import torchinfo

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import imageio

from sklearn.metrics import f1_score, confusion_matrix, classification_report

class SiameseNetwork(nn.Module):
    def __init__(self, train_backbone_params=True):
        super().__init__()
        self._train_backbone_params = train_backbone_params
        
        self.model = None
        self.loss = ContrastiveLoss()
        self._init_backbone()
        self._build_siamese()
        
    def _init_backbone(self):
        backbone = torchvision.models.resnet18(pretrained=True)
        
        if self._train_backbone_params:
            for param in backbone.parameters():
                param.requires_grad = True
        else:
            for param in backbone.parameters():
                param.requires_grad = False
        
        self._backbone = backbone
    
    def _build_siamese(self):
        self.model = self._backbone
        self.model.fc = nn.Sequential(
            nn.Linear(self._backbone.fc.in_features, 256),
            nn.ReLU(),
            nn.Linear(256, 128) # final feature vector
        )
        
    def forward_once(self, X):
        output = self.model(X)
        return output
    
    def forward(self, X1, X2):
        output1 = self.forward_once(X1)
        output2 = self.forward_once(X2)
        distance = F.pairwise_distance(output1, output2)    
        return distance
    
    def training_step(self, batch, threshhold=0.75):
        X, y = batch
        out = self(X[0], X[1]) 
        loss = self.loss(out, y) 
        acc = compute_accuracy(out, threshhold, y)
        
        return loss, acc
    
    def validation_step(self, batch, threshhold=0.75):
        X, y = batch
        out = self(X[0], X[1]) 
        val_loss = self.loss(out, y)
        val_acc = compute_accuracy(out, threshhold, y)
        return {"val_loss": val_loss, "val_acc": val_acc}
    
    def validation_epoch_end(self, outputs):
        batch_losses = [x["val_loss"] for x in outputs]
        batch_acc = [x["val_acc"] for x in outputs]

        epoch_loss = torch.stack(batch_losses).mean().item()
        epoch_acc = torch.stack(batch_acc).mean().item()
        return {"val_loss": epoch_loss, "val_acc": epoch_acc}
    
    def evaluate(self, dl):
        self.eval()
        with torch.no_grad():
            self.eval()
            outputs = [self.validation_step(batch) for batch in dl]
    
        return self.validation_epoch_end(outputs)
    
    def epoch_end_val(self, epoch, results):
        print(f"Epoch:[{epoch}]: validation loss: {results['val_loss']}, validation accuracy: {results['val_acc']}")

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1):
        super().__init__()
        self.margin = margin
    
    def forward(self, distance, labels):
        y = labels.float()
        
        loss = torch.mean(torch.square(distance) * (1-y) + torch.square(torch.max(self.margin - distance, torch.zeros_like(distance))) * (y)) 
        return loss

def compute_accuracy(distance, threshhold, true_labels):
    y = true_labels.float()
    prediction = distance.to(torch.device("cpu")).detach().apply_(lambda x: 0 if x < threshhold else 1)
    return sum((prediction == y.to(torch.device("cpu")))) / len(true_labels)

def get_default_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def train_generator(batch_size=64):
    n_batches = int(np.ceil(len(train_positives) / batch_size))
    device = get_default_device()
    
    random.shuffle(train_positives)
    X1 = torch.zeros((batch_size * 2,*img_shape)) 
    X2 = torch.zeros((batch_size * 2,*img_shape))
    y = torch.zeros((batch_size * 2))

    for i in range(n_batches): 
        pos_samples = train_positives[i * batch_size : batch_size * (i + 1)]
        neg_samples = np.random.choice(len(train_negatives), size=len(pos_samples))
        j = 0

        for x1, x2 in pos_samples:
            X1[j] = data[x1]
            X2[j] = data[x2]
            y[j] = 0
            j += 1

        for ind in neg_samples:
            X1[j] = data[train_negatives[ind][0]]
            X2[j] = data[train_negatives[ind][1]]
            y[j] = 1
            j += 1

        X1 = transforms_custom(X1).to(device)[:j]
        X2 = transforms_custom(X2).to(device)[:j]
        y = y.to(device)[:j]

        yield (X1, X2), y

def test_generator(batch_size=64):
    n_batches = int(np.ceil(len(test_positives) / batch_size))
    device = get_default_device()
    
    random.shuffle(train_positives)
    X1 = torch.zeros((batch_size * 2,*img_shape)) 
    X2 = torch.zeros((batch_size * 2,*img_shape))
    y = torch.zeros((batch_size * 2))
        
    for i in range(n_batches): 
        pos_samples = train_positives[i * batch_size : batch_size * (i + 1)]
        neg_samples = np.random.choice(len(train_negatives), size=len(pos_samples))
        j = 0

        for x1, x2 in pos_samples:
            X1[j] = data[x1]
            X2[j] = data[x2]
            y[j] = 0
            j += 1

        for ind in neg_samples:
            X1[j] = data[train_negatives[ind][0]]
            X2[j] = data[train_negatives[ind][1]]
            y[j] = 1
            j += 1

        X1 = transforms_custom(X1).to(device)[:j]
        X2 = transforms_custom(X2).to(device)[:j]
        y = y.to(device)[:j]

        yield (X1, X2), y

def fit(model, epochs, batch_size, train_generator, val_generator, optimizer, learning_rate, lr_scheduler,**kwargs):
    optimizer = optimizer(model.parameters(), lr=learning_rate)

    if lr_scheduler:
            lrs = lr_scheduler(optimizer, **kwargs)

    history = []
    min_val_loss = float('inf')

    model.train()
    for epoch in range(epochs):
        train_losses = []
        train_acc = []
        for num, batch in enumerate(train_generator(batch_size)):
            optimizer.zero_grad()
            loss, acc= model.training_step(batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.detach())
            train_acc.append(acc)

        result = model.evaluate(val_generator(batch_size))
        result["train_loss"] = torch.stack(train_losses).mean().item()
        result["train_acc"] = torch.stack(train_acc).mean().item()

        history.append(result)
       
        if lr_scheduler:
            lrs.step(metrics=result["val_loss"])

        model.epoch_end_val(epoch,result)

        if result["val_loss"] < min_val_loss:
            torch.save(model, 'best_model.pt')
            min_val_loss = result["val_loss"]

    return history

def get_train_accuracy(threshold=0.85):
    positive_distances = []
    negative_distances = []

    tp = 0
    tn = 0
    fp = 0
    fn = 0

    batch_size = 64
    
    X, y = next(train_generator())
    x1_positive = X[0][:batch_size].to(torch.device("cpu"))
    x2_positive = X[1][:batch_size].to(torch.device("cpu"))
    distances = model(x1_positive, x2_positive)
    positive_distances += distances.tolist()

    tp += (distances < threshold).sum()

    x1_negative = X[0][batch_size:].to(torch.device("cpu"))
    x2_negative = X[1][batch_size:].to(torch.device("cpu"))
    distances = model(x1_negative, x2_negative)
    negative_distances += distances.tolist()

    fp += (distances < threshold).sum()
    tn += (distances > threshold).sum()

    tpr = tp / (tp + fn)
    tnr = tn / (tn + fp)
    
    plt.hist(negative_distances, bins=20, density=True, label='negative_distances')
    plt.hist(positive_distances, bins=20, density=True, label='positive_distances')
    plt.suptitle("Train data evaluation")
    plt.title(f"sensitivity (tpr): {tpr}, specificity (tnr): {tnr}")
    plt.legend()
    plt.show()

def get_test_accuracy(threshold=0.85):
    positive_distances = []
    negative_distances = []

    tp = 0
    tn = 0
    fp = 0
    fn = 0

    batch_size = 64
    
    X, y = next(test_generator())
    x1_positive = X[0][:batch_size].to(torch.device("cpu"))
    x2_positive = X[1][:batch_size].to(torch.device("cpu"))
    distances = model(x1_positive, x2_positive)
    positive_distances += distances.tolist()

    tp += (distances < threshold).sum()
    x1_negative = X[0][batch_size:].to(torch.device("cpu"))
    x2_negative = X[1][batch_size:].to(torch.device("cpu"))
    distances = model(x1_negative, x2_negative)
    negative_distances += distances.tolist()

    fp += (distances < threshold).sum()
    tn += (distances > threshold).sum()

    tpr = tp / (tp + fn)
    tnr = tn / (tn + fp)
    
    plt.hist(negative_distances, bins=20, density=True, label='negative_distances')
    plt.hist(positive_distances, bins=20, density=True, label='positive_distances')
    plt.suptitle("Test data evaluation!")
    plt.title(f"sensitivity (tpr): {tpr}, specificity (tnr): {tnr}")
    plt.legend()
    plt.show()

def check_prediction(ind1=None, ind2=None):
    if ind1 is None:
        ind1 = np.random.choice(len(data))
        ind2 = np.random.choice(len(data))  
    fig, (ax1, ax2) = plt.subplots(1,2)
    x1, x2 = data[ind1].to(torch.device("cpu")), data[ind2].to(torch.device("cpu"))
    ax1.imshow(x1.permute(1,2,0))
    ax2.imshow(x2.permute(1,2,0))
    ax1.axis("off")
    ax2.axis("off")
    
    pair = (ind1, ind2)
    
    if (pair in positives):
        ax1.set_title("True label: Match!")
    else:
        ax1.set_title("True label: No match")
    
    distance = model(torch.unsqueeze(x1,0), torch.unsqueeze(x2,0))
    
    if distance < 0.75:
        ax2.set_title(f"Prediction: match with distance: {distance.item():.2f}")
    else:
        ax2.set_title(f"Prediction: no match with distance: {distance.item():.2f}")
    
    plt.show()

def main():
    # Data Preprocessing
    
    # Define paths and parameters
    path = "/media/statlab/SeagateHDD/Fateme Tavakoli/ElasticFace/ds/LFW/lfw"
    img_shape = (3,250,250)
    train_size = 0.7 

    # Find subdirectories
    people = glob.glob(path + "/*/")

    # Check if images are all the same size
    image_shape_tracker = {}
    for p in people:
        for file in os.listdir(p):
            img = imageio.v2.imread(os.path.join(p, file))
            img_shape = np.array(img).shape
            if img_shape not in image_shape_tracker.keys():
                image_shape_tracker[img_shape] = 1
            else:
                image_shape_tracker[img_shape] += 1
    assert len(image_shape_tracker.keys()) == 1, "All the images should be of the same size!"

    # Load all images into torch tensor
    len_data = 0
    for p in people:
        len_data += len(os.listdir(p))

    data = torch.zeros((len_data, *img_shape))
    index = 0
    for p in people:
        for file in os.listdir(p):
            data[index] = torchvision.io.read_image(os.path.join(p, file)) / 255.
            index += 1

    # Create pairs of positive and negative samples
    pair_mapping = {}
    counter = 0

    for ind, p in enumerate(people):
        pair_mapping[ind] = []

        for file in os.listdir(p):
            pair_mapping[ind].append(counter)
            counter += 1

    means = [np.mean(len(xs)) for xs in pair_mapping.values()]
    means_length = np.mean(means)

    positives = []
    negatives = []

    for key, indices in pair_mapping.items():
        indices = list(indices)
        other_indices = set(range(len_data)).difference(set(indices))

        for i, index in enumerate(indices):
            for subindex in indices[i + 1:]:
                positives.append((index, subindex))
            
            for subindex in other_indices:
                negatives.append((index, subindex))

    # Split data into train and test sets
    random.seed(1234)
    train_positives = random.sample(positives, k=int(train_size * len(positives)))
    train_negatives = random.sample(negatives, k=int(train_size * len(negatives)))
    test_positives = list(set(positives).difference(set(train_positives)))
    test_negatives = list(set(negatives).difference(set(train_negatives)))

    # Define transforms
    stats = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    base_transforms = [
        tt.Resize((256,256), interpolation = tt.InterpolationMode.BICUBIC),
    ]

    transforms_custom = tt.Compose([
        *base_transforms,
        tt.RandomCrop(250),
        tt.RandomHorizontalFlip(),
        tt.RandomRotation(30),
        tt.RandomVerticalFlip(),
        tt.Normalize(*stats)
    ])

    test_transforms = tt.Compose([
        tt.Normalize(*stats)
    ])

    # Model Training

    device = get_default_device()
    model = SiameseNetwork()
    model = model.to(device)

    history = fit(model, 10, 128, train_generator, test_generator, torch.optim.Adam, 0.001,
                  torch.optim.lr_scheduler.ReduceLROnPlateau, patience=5, factor=0.05)

    model = torch.load("best_model.pt", map_location=torch.device("cpu"))

    # Model Evaluation
    get_train_accuracy()
    get_test_accuracy()
    check_prediction(40, 41)

if __name__ == "__main__":
    main()
