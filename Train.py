import os
import argparse
from tqdm import tqdm
import pandas as pd
import joblib
import glob
from collections import OrderedDict
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from Networks.network import MODEL as net
from losses import ir_loss, vi_loss,ssim_loss,gra_loss
device = torch.device('cuda:0')
use_gpu = torch.cuda.is_available()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='...', help='model name: (default: arch+timestamp)')
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float)
    parser.add_argument('--betas', default=(0.9, 0.999), type=tuple)
    parser.add_argument('--eps', default=1e-8, type=float)
    parser.add_argument('--weight', default=[1, 1,10,100], type=float)
    args = parser.parse_args()
    return args

class GetDataset(Dataset):
    def __init__(self, imageFolderDataset, transform=None):
        self.imageFolderDataset = imageFolderDataset
        self.transform = transform

    def __getitem__(self, index):

        ir='...'
        vi='...'

        ir = Image.open(ir).convert('L')
        vi = Image.open(vi).convert('L')

        if self.transform is not None:
            tran = transforms.ToTensor()
            ir=tran(ir)
            vi= tran(vi)
            input = torch.cat((ir, vi), -3)
            return input, ir,vi

    def __len__(self):
        return len(self.imageFolderDataset)


class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train(args, train_loader_ir,train_loader_vi, model, criterion_ir, criterion_vi,criterion_ssim,criterion_gra,optimizer, epoch, scheduler=None):
    losses = AverageMeter()
    losses_ir = AverageMeter()
    losses_vi = AverageMeter()
    losses_ssim = AverageMeter()
    losses_gra= AverageMeter()
    weight = args.weight
    model.train()

    for i, (input,ir,vi)  in tqdm(enumerate(train_loader_ir), total=len(train_loader_ir)):


        input = input.cuda()

        ir=ir.cuda()
        vi=vi.cuda()

        out = model(input)

        loss_ir = weight[0] * criterion_ir(out, ir)
        loss_vi = weight[1] * criterion_vi(out, vi)
        loss_ssim= weight[2] * criterion_ssim(out,ir, vi)
        loss_gra = weight[3] * criterion_gra(out, ir,vi)
        loss = loss_ir + loss_vi+loss_ssim+ loss_gra

        losses.update(loss.item(), input.size(0))
        losses_ir.update(loss_ir.item(), input.size(0))
        losses_vi.update(loss_vi.item(), input.size(0))
        losses_ssim.update(loss_ssim.item(), input.size(0))
        losses_gra.update(loss_gra.item(), input.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    log = OrderedDict([
        ('loss', losses.avg),
        ('loss_ir', losses_ir.avg),
        ('loss_vi', losses_vi.avg),
        ('loss_ssim', losses_ssim.avg),
        ('loss_gra', losses_gra.avg),
    ])

    return log



def main():
    args = parse_args()

    if not os.path.exists('models/%s' %args.name):
        os.makedirs('models/%s' %args.name)


    with open('models/%s/args.txt' %args.name, 'w') as f:
        for arg in vars(args):
            print('%s: %s' %(arg, getattr(args, arg)), file=f)

    joblib.dump(args, 'models/%s/args.pkl' %args.name)
    cudnn.benchmark = True

    training_dir_ir = "..."
    folder_dataset_train_ir = glob.glob(training_dir_ir )
    training_dir_vi = "..."

    folder_dataset_train_vi= glob.glob(training_dir_vi )

    transform_train = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize((0.485, 0.456, 0.406),
                                                               (0.229, 0.224, 0.225))
                                          ])

    dataset_train_ir = GetDataset(imageFolderDataset=folder_dataset_train_ir,
                                                  transform=transform_train)
    dataset_train_vi = GetDataset(imageFolderDataset=folder_dataset_train_vi,
                                  transform=transform_train)

    train_loader_ir = DataLoader(dataset_train_ir,
                              shuffle=True,
                              batch_size=args.batch_size)
    train_loader_vi = DataLoader(dataset_train_vi,
                                 shuffle=True,
                                 batch_size=args.batch_size)
    model = net(in_channel=2)
    if use_gpu:
        model = model.cuda()
        model.cuda()

    else:
        model = model
    criterion_ir = ir_loss
    criterion_vi = vi_loss
    criterion_ssim = ssim_loss
    criterion_gra = gra_loss
    optimizer = optim.Adam(model.parameters(), lr=args.lr,
                           betas=args.betas, eps=args.eps)
    log = pd.DataFrame(index=[],
                       columns=['epoch',
                                'loss',
                                'loss_ir',
                                'loss_vi',
                                'loss_ssim',
                                'loss_gra',
                                ])

    for epoch in range(args.epochs):

        train_log = train(args, train_loader_ir,train_loader_vi, model, criterion_ir, criterion_vi,criterion_ssim,criterion_gra, optimizer, epoch)
        tmp = pd.Series([
            epoch + 1,
            train_log['loss'],
            train_log['loss_ir'],
            train_log['loss_vi'],
            train_log['loss_ssim'],
            train_log['loss_gra'],
        ], index=['epoch', 'loss', 'loss_ir', 'loss_vi', 'loss_ssim', 'loss_gra'])

        log = log.append(tmp, ignore_index=True)
        log.to_csv('models/%s/log.csv' %args.name, index=False)

        if (epoch+1) % 1 == 0:
            torch.save(model.state_dict(), 'models/%s/model_{}.pth'.format(epoch+1) %args.name)

if __name__ == '__main__':
    main()


