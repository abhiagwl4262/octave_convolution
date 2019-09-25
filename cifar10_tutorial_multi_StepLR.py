#!/usr/bin/env python

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from progress.bar import Bar
import torch.optim as optim
import shutil
import os
import argparse
from  torchsummary import summary

from ptflops import get_model_complexity_info


best_acc = 0

classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

class AverageMeter(object):
    """Computes and stores the average and current value"""
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


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class OctConv(nn.Module): 
    def __init__(self, ch_in, ch_out, kernel_size, stride=1, padding = 1, alphas=[0.0, 0.0]):
        super(OctConv, self).__init__()

        # Get layer parameters 
        self.alpha_in, self.alpha_out = alphas
        assert 0 <= self.alpha_in <= 1 and 0 <= self.alpha_in <= 1, \
                    "Alphas must be in interval [0, 1]"
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        # self.padding = (kernel_size - stride ) // 2
        
        # Calculate the exact number of high/low frequency channels 
        self.ch_in_lf = int(self.alpha_in*ch_in)
        self.ch_in_hf = ch_in - self.ch_in_lf
        self.ch_out_lf = int(self.alpha_out*ch_out) 
        self.ch_out_hf = ch_out - self.ch_out_lf

        # Create convolutional and other modules necessary. Not all paths 
        # will be created in call cases. So we check number of high/low freq 
        # channels in input/output to determine which paths are present.
        # Example: First layer has alpha_in = 0, so hasLtoL and hasLtoH (bottom
        # two paths) will be false in this case. 
        self.hasLtoL = self.hasLtoH = self.hasHtoL = self.hasHtoH = False
        if (self.ch_in_lf and self.ch_out_lf):    
            # Green path at bottom. 
            self.hasLtoL = True
            self.conv_LtoL = nn.Conv2d(self.ch_in_lf, self.ch_out_lf, \
                                       self.kernel_size, stride= self.stride, padding=self.padding)
        if (self.ch_in_lf and self.ch_out_hf): 
            # Red path at bottom. 
            self.hasLtoH = True
            self.conv_LtoH = nn.Conv2d(self.ch_in_lf, self.ch_out_hf, \
                                       self.kernel_size, stride= self.stride, padding=self.padding)
        if (self.ch_in_hf and self.ch_out_lf):
            # Red path at top
            self.hasHtoL = True
            self.conv_HtoL = nn.Conv2d(self.ch_in_hf, self.ch_out_lf, \
                                       self.kernel_size, stride= self.stride, padding=self.padding)
        if (self.ch_in_hf and self.ch_out_hf):
            # Green path at top
            self.hasHtoH = True
            self.conv_HtoH = nn.Conv2d(self.ch_in_hf, self.ch_out_hf, \
                                       self.kernel_size, stride= self.stride, padding=self.padding)
        self.avg_pool  = nn.AvgPool2d(2,2)
        
    def forward(self, input):         
        # Split input into high frequency and low frequency components
        fmap_w = input.shape[-1]
        fmap_h = input.shape[-2]
        # We resize the high freqency components to the same size as the low 
        # frequency component when sending out as output. So when bringing in as 
        # input, we want to reshape it to have the original size as the intended 
        # high frequnecy channel (if any high frequency component is available). 
        input_hf = input
        if (self.ch_in_lf):
            input_hf = input[:,:self.ch_in_hf,:,:]
            input_lf = self.avg_pool(input[:,self.ch_in_hf:,:,:])

        
            # input_hf = input[:,:self.ch_in_hf*4,:,:].reshape(-1, self.ch_in_hf,fmap_h*2,fmap_w*2)
            # input_lf = input[:,self.ch_in_hf*4:,:,:]    
        
        # Create all conditional branches 
        LtoH = HtoH = LtoL = HtoL = 0.
        if (self.hasLtoL):
            # Since, there is no change in spatial dimensions between input and 
            # output, we use vanilla convolution
            LtoL = self.conv_LtoL(input_lf)
            # LtoL = F.interpolate(LtoL, scale_factor=2, mode='bilinear')

        if (self.hasHtoH):
            # Since, there is no change in spatial dimensions between input and 
            # output, we use vanilla convolution
            HtoH = self.conv_HtoH(input_hf)
            # We want the high freq channels and low freq channels to be 
            # packed together such that the output has one dimension. This 
            # enables octave convolution to be used as is with other layers 
            # like Relu, elementwise etc. So, we fold the high-freq channels 
            # to make its height and width same as the low-freq channels. So, 
            # h = h/2 and w = w/2 since we are making h and w smaller by a 
            # factor of 2, the number of channels increases by 4. 
            
            # op_h, op_w = HtoH.shape[-2]//2, HtoH.shape[-1]//2
            # HtoH = HtoH.reshape(-1, self.ch_out_hf*4, op_h, op_w)
        if (self.hasLtoH):
            # Since, the spatial dimension has to go up, we do 
            # bilinear interpolation to increase the size of output 
            # feature maps 
            LtoH = F.interpolate(self.conv_LtoH(input_lf), \
                                 scale_factor=2, mode='bilinear')
            # We want the high freq channels and low freq channels to be 
            # packed together such that the output has one dimension. This 
            # enables octave convolution to be used as is with other layers 
            # like Relu, elementwise etc. So, we fold the high-freq channels 
            # to make its height and width same as the low-freq channels. So, 
            # h = h/2 and w = w/2 since we are making h and w smaller by a 
            # factor of 2, the number of channels increases by 4. 
            
            # op_h, op_w = LtoH.shape[-2]//2, LtoH.shape[-1]//2
            # LtoH = LtoH.reshape(-1, self.ch_out_hf*4, op_h, op_w)
        if (self.hasHtoL):
            # Since, the spatial dimension has to go down here, we do 
            # average pooling to reduce the height and width of output
            # feature maps by a factor of 2
            # HtoL = self.avg_pool(self.conv_HtoL(input_hf))
            # HtoL = self.conv_HtoL(input_hf)
            HtoL = self.conv_HtoL(self.avg_pool(input_hf))
        
        # Elementwise addition of high and low freq branches to get the output
        out_hf = LtoH + HtoH
        out_lf = LtoL + HtoL
        out_lf = F.interpolate(out_lf, scale_factor=2, mode='bilinear')
        
        # Since, not all paths are always present, we need to put a check 
        # on how the output is generated. Example: the final convolution layer
        # will have alpha_out == 0, so no low freq. output channels, 
        # so the layers returns just the high freq. components. If there are no 
        # high freq component then we send out the low freq channels (we have it 
        # just to have a general module even though this scenerio has not been
        # used by the authors). If both low and high freq components are present, 
        # we concat them (we have already resized them to be of the same dimension) 
        # and send them out.  
        if (self.ch_out_lf == 0):
            return out_hf
        if (self.ch_out_hf == 0):
            return out_lf
        op = torch.cat([out_hf,out_lf],dim=1)
        return op

def replace_func(model, key):
    c_in     = model._modules[key].in_channels
    c_out    = model._modules[key].out_channels
    kernel_w = model._modules[key].kernel_size[0]
    kernel_h = model._modules[key].kernel_size[1]
    padding  = model._modules[key].padding[0]
    stride   = model._modules[key].stride[0]

    model._modules[key] = OctConv(c_in, c_out, kernel_size=kernel_w, stride=stride, padding=padding)

def make_octconv_net(model):
    for key, val in model._modules.items():
        if isinstance(val, nn.Conv2d):
            # print("conv layer is ", val)
            if(val.in_channels > 4):
                replace_func(model,key)
        elif (isinstance(val, nn.Sequential)) or (isinstance(val,torchvision.models.resnet.BasicBlock)):
              make_octconv_net(val)

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1  = nn.Conv2d(3, 32, kernel_size=(3,3), padding=(1,1), stride=(1,1))
        self.bn1    = nn.BatchNorm2d(32)
        self.conv2  = nn.Conv2d(32, 32, kernel_size=(3,3), padding=(1,1), stride=(1,1))
        self.bn2    = nn.BatchNorm2d(32)
        
        self.conv3  = nn.Conv2d(32, 64, kernel_size=(3,3), padding=(1,1), stride=(1,1))
        self.bn3    = nn.BatchNorm2d(64)
        self.conv4  = nn.Conv2d(64, 64, kernel_size=(3,3), padding=(1,1), stride=(1,1))
        self.bn4    = nn.BatchNorm2d(64)
        
        self.pool   = nn.MaxPool2d(2, 2) 

        self.conv5  = nn.Conv2d(64, 128, kernel_size=(3,3), padding=(1,1), stride=(1,1))               
        self.bn5    = nn.BatchNorm2d(128)

        self.conv6  = nn.Conv2d(128, 10, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.avgpool=nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.flatten=Flatten()
        self.relu   = nn.ReLU()


    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x    )
        x = self.bn2(x)
        x = self.relu(x)
        
        x = self.pool(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)

        x = self.pool(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)

        x = self.conv6(x)                
        x = self.relu(x)
        out = self.flatten(x)
        return out

def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar', snapshot=None):
    # preds = to_numpy(preds)
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))
        # scipy.io.savemat(os.path.join(checkpoint, 'preds_best.mat'), mdict={'preds' : preds})


def adjust_learning_rate(optimizer, epoch, lr, schedule, gamma):
    """Sets the learning rate to the initial LR decayed by schedule"""
    if epoch in schedule:
        lr *= gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    return lr

def mkdir_p(dir_path):
    try:
        os.makedirs(dir_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

def main(args):
    global best_acc

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])

    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform_train)
    train_loader  = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                              shuffle=True, num_workers=4)

    val_dataset   = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform_test)
    val_loader    = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size,
                                             shuffle=False, num_workers=4)

    print("number of batches are ", len(train_loader))
    

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print(device)


    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)


    model = torchvision.models.resnet18(pretrained = True)
    # model.fc.out_features = 10   
    model.fc = nn.Linear(in_features=512, out_features=10, bias=True)                                                                        
    flops, params = get_model_complexity_info(model, (3, 32, 32), as_strings=True, print_per_layer_stat=False)
    print("FLOPs in original resnet18 model are ", flops)
    print("Number of Params in original resnet18 model are", params)

    if args.enable_octave:
        make_octconv_net(model)        
        flops, params = get_model_complexity_info(model, (3, 32, 32), as_strings=True, print_per_layer_stat=False)
        print("FLOPs in OctConv resnet18 model are ", flops)
        print("Number of Params in OctConv resnet18 model are", params)
        print(model)

    

    # model = model.cuda()
    model = torch.nn.DataParallel(model).cuda()
    # checkpoint = torch.load("checkpoint/model_best.pth.tar")
    # model.load_state_dict(checkpoint['state_dict'])
    # summary(model, (3,32,32))

    # criterion = FocalLoss()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            args.lr = checkpoint['optimizer']['param_groups'][0]['lr']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            # logger = Logger(join(args.checkpoint, 'log.txt'), title=title, resume=True)
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))


    # validation
    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    gamma    = args.gamma
    lr       = args.lr
    schedule = args.schedule
    
    for epoch in range(args.start_epoch, args.epochs):
        
        lr = adjust_learning_rate(optimizer, epoch, lr, schedule, gamma)
        print('\nEpoch: %d | LR: %.8f' % (epoch + 1, lr))

        train_loss, train_acc = train(train_loader, model, optimizer, criterion)
        valid_loss, valid_acc = validate(val_loader, model, criterion)

        print(" val loss     ", valid_loss)
        print(" val Accuracy ", valid_acc )
 
        is_best = valid_acc > best_acc
        best_acc = max(valid_acc, best_acc)

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_acc'  : best_acc,
            'optimizer' : optimizer.state_dict(),
        }, is_best, checkpoint=args.checkpoint)

        if (args.enable_octave):
            f =open("log_cifar10_resnet_octave_conv_.txt","a")
        else:
            f =open("log_cifar10_resnet_vanilla_conv_.txt","a")                        
        f.write('Train FP epoch: [{0}]\t'
                'Train loss {train_loss:.3f} \t'
                'Train Accuracy {train_acc:.3f} \t'
                'Val loss {valid_loss:.3f} \t'
                'Val Accuracy {valid_acc:.3f} \t'
                'LR {lr} \n'.format(
                epoch,  train_loss=train_loss, train_acc=train_acc, valid_loss=valid_loss, valid_acc=valid_acc, lr=lr))
        f.close()

def train(train_loader, model, optimizer, criterion):
    correct = 0
    total = 0

    losses = AverageMeter()
    bar = Bar('Train', max=len(train_loader))

    model.train()
    for i, data in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs, labels = inputs.cuda(), labels.cuda()
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total   += labels.size(0)
        correct += (predicted == labels).sum().item()


        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), inputs.size(0))

        # print statistics
        bar.suffix  = '({batch}/{size}) | Loss: {loss:.4f} | Acc: {acc:.4f}'.format(
                    batch=i + 1,
                    size=len(train_loader),
                    loss=losses.avg,
                    acc=(100*correct/total)
                    )
        bar.next()        
        
    bar.finish()

    return losses.avg, (100*correct/total)


def validate(val_loader, model, criterion):
    correct = 0
    total = 0
    losses = AverageMeter()

    model.eval()
    with torch.no_grad():
        for data in val_loader:
            images, labels = data
            
            images, labels = images.cuda(), labels.cuda()

            outputs = model(images)
            loss    = criterion(outputs, labels)
            losses.update(loss.item(), images.size(0))

            _, predicted = torch.max(outputs.data, 1)
            total   += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))


    # class_correct = list(0. for i in range(10))
    # class_total   = list(0. for i in range(10))
    
    # with torch.no_grad():
    #     for data in val_loader:
    #         images, labels = data
    #         images, labels = images.cuda(), labels.cuda()
    #         outputs = model(images)
    #         loss = criterion(outputs, labels)
    #         losses.update(loss.item(), images.size(0))

    #         _, predicted = torch.max(outputs, 1)
    #         c = (predicted == labels).squeeze()
    #         for i in range(4):
    #             label = labels[i]
    #             class_correct[label] += c[i].item()
    #             class_total[label] += 1


    # for i in range(10):
    #     print('Accuracy of %5s : %2d %%' % (
    #         classes[i], 100 * class_correct[i] / class_total[i]))


    return losses.avg, (100*correct/total)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    # Dataset setting
    parser.add_argument('--dataset', metavar='DATASET', default='CIFAR10')

    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')

    parser.add_argument('--epochs', default=150, type=int, metavar='N',
                        help='number of total epochs to run')

    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')

    parser.add_argument('--batch_size', default=64, type=int, metavar='N',
                        help='train batchsize')

    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='initial learning rate')

    parser.add_argument('--momentum', default=0, type=float, metavar='M',
                        help='momentum')

    parser.add_argument('--weight-decay', '--wd', default=0, type=float,
                        metavar='W', help='weight decay (default: 0)')

    parser.add_argument('--schedule', type=int, nargs='+', default=[70, 100, 130],
                        help='Decrease learning rate at these epochs.')

    parser.add_argument('--gamma', type=float, default=0.1,
                        help='LR is multiplied by gamma on schedule.')

    # Miscs
    parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH', help='path to save checkpoint (default: checkpoint)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')

    parser.add_argument('--enable_octave', action='store_true',
                        help='evaluate model on validation set')

    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')

    main(parser.parse_args())