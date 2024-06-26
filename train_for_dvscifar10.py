import argparse
import torch
from parallel_nets.ResNet_for_dvscifar10 import *
from parallel_nets.VGGnet_for_dvscifar10 import *
import torch.optim as optim
import os
import torch.nn.functional as F
import time
from tensorboardX import SummaryWriter
from DVS_dataload.DVS_CIFAR10_dataset import *
import torch.utils.data
# 超参数
same_tau = False
train_tau = False
use_TET = False
#
def data_model_load(args, model, kwargs):
    path = os.getcwd() + os.sep + 'data' + os.sep + 'DVS_CIFAR10'
    train_dataset = DVSCIFAR10Dataset(path, train=True, transform=Compose([Normalize_ToTensor(Vth)]))
    test_dataset = DVSCIFAR10Dataset(path, train=False, transform=Compose([Normalize_ToTensor(Vth)]))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=True, **kwargs)
    if args.pretrained:
        checkpoint = torch.load(args.checkpoint_path)
        model.load_state_dict(checkpoint['model'])
        start_epoch = checkpoint['epoch']
        print('Pretrained model loaded.')
    else:
        start_epoch = 0
        print('Model loaded.')
    return train_loader, test_loader, start_epoch


def train(args, model, train_loader, optimizer, device, epoch, writer):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data_temp, target = data.to(device), target.to(device)
        bs = data_temp.shape[0]
        data = torch.zeros((TimeStep*bs,) + data_temp.shape[2:], device=data_temp.device)
        for t in range(TimeStep):
            data[t*bs:(t+1)*bs, ...] = data_temp[:, t, :, :, :]

        output = model(data)
        #
        # print(model.layer1.execute[0].spike_func.tau2)
        # print(model.layer2.execute[0].spike_func.tau2)
        # print(model.layer1.execute[1].spike_func.tau2)
        #
        target = target.long().to(target.device)
        if not use_TET:
            loss = F.cross_entropy(output, target)
        else:
            loss = .0
            for t in range(TimeStep):
                loss += F.cross_entropy(output[t*bs:(t+1)*bs, ...], target)
            loss /= TimeStep
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * args.batch_size, len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
            if args.tensorboard:
                writer.add_scalar('Train Loss / batch_idx', loss.item(), batch_idx + len(train_loader) * epoch)


def _test(args, model, test_loader, device, writer):
    model.eval()
    total_loss = 0.
    correct = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data_temp, target = data.to(device), target.to(device)
            bs = data_temp.shape[0]
            data = torch.zeros((TimeStep * bs,) + data_temp.shape[2:], device=data_temp.device)
            for t in range(TimeStep):
                data[t * bs:(t + 1) * bs, ...] = data_temp[:, t, :, :, :]

            output = model(data)
            target = target.long().to(target.device)
            #
            if use_TET:
                bs = int(output.shape[0] / TimeStep)
                o = torch.zeros((bs,) + output.shape[1:], device=output.device)
                for t in range(TimeStep):
                    o += output[t * bs:(t + 1) * bs, ...]
                o /= TimeStep
                output = o
            #
            total_loss += F.cross_entropy(output, target, reduction='sum').item()
            pre_result = output.argmax(dim=1, keepdim=True)
            correct += pre_result.eq(target.view_as(pre_result)).sum().item()

    total_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        total_loss, correct, len(test_loader.dataset),
        accuracy))

    if args.tensorboard:
        writer.add_scalar('Test Loss / epoch', total_loss, epoch)
        writer.add_scalar('Test Accuracy / epoch', accuracy, epoch)
    # 仅针对ResNet14， 提取tau参数
    with torch.no_grad():
        if args.tensorboard & train_tau:
            if not same_tau:
                MLF0 = model.spike_func
                MLF10 = model.layer1.execute[0].spike_func
                MLF11 = model.layer1.execute[1].spike_func
                MLF20 = model.layer2.execute[0].spike_func
                MLF21 = model.layer2.execute[1].spike_func
                MLF30 = model.layer3.execute[0].spike_func
                MLF31 = model.layer3.execute[1].spike_func
                writer.add_scalar('MLF0.tau / epoch', MLF0.tau.sigmoid(), epoch)
                writer.add_scalar('MLF0.tau2 / epoch', MLF0.tau2.sigmoid(), epoch)
                writer.add_scalar('MLF0.tau3 / epoch', MLF0.tau3.sigmoid(), epoch)

                writer.add_scalar('MLF10.tau / epoch', MLF10.tau.sigmoid(), epoch)
                writer.add_scalar('MLF10.tau2 / epoch', MLF10.tau2.sigmoid(), epoch)
                writer.add_scalar('MLF10.tau3 / epoch', MLF10.tau3.sigmoid(), epoch)

                writer.add_scalar('MLF11.tau / epoch', MLF11.tau.sigmoid(), epoch)
                writer.add_scalar('MLF11.tau2 / epoch', MLF11.tau2.sigmoid(), epoch)
                writer.add_scalar('MLF11.tau3 / epoch', MLF11.tau3.sigmoid(), epoch)

                writer.add_scalar('MLF20.tau / epoch', MLF20.tau.sigmoid(), epoch)
                writer.add_scalar('MLF20.tau2 / epoch', MLF20.tau2.sigmoid(), epoch)
                writer.add_scalar('MLF20.tau3 / epoch', MLF20.tau3.sigmoid(), epoch)

                writer.add_scalar('MLF21.tau / epoch', MLF21.tau.sigmoid(), epoch)
                writer.add_scalar('MLF21.tau2 / epoch', MLF21.tau2.sigmoid(), epoch)
                writer.add_scalar('MLF21.tau3 / epoch', MLF21.tau3.sigmoid(), epoch)

                writer.add_scalar('MLF30.tau / epoch', MLF30.tau.sigmoid(), epoch)
                writer.add_scalar('MLF30.tau2 / epoch', MLF30.tau2.sigmoid(), epoch)
                writer.add_scalar('MLF30.tau3 / epoch', MLF30.tau3.sigmoid(), epoch)

                writer.add_scalar('MLF31.tau / epoch', MLF31.tau.sigmoid(), epoch)
                writer.add_scalar('MLF31.tau2 / epoch', MLF31.tau2.sigmoid(), epoch)
                writer.add_scalar('MLF31.tau3 / epoch', MLF31.tau3.sigmoid(), epoch)
            else:
                MLF0 = model.spike_func
                MLF10 = model.layer1.execute[0].spike_func
                MLF11 = model.layer1.execute[1].spike_func
                MLF20 = model.layer2.execute[0].spike_func
                MLF21 = model.layer2.execute[1].spike_func
                MLF30 = model.layer3.execute[0].spike_func
                MLF31 = model.layer3.execute[1].spike_func
                writer.add_scalar('MLF0.tau / epoch', MLF0.tau.sigmoid(), epoch)

                writer.add_scalar('MLF10.tau / epoch', MLF10.tau.sigmoid(), epoch)

                writer.add_scalar('MLF11.tau / epoch', MLF11.tau.sigmoid(), epoch)

                writer.add_scalar('MLF20.tau / epoch', MLF20.tau.sigmoid(), epoch)

                writer.add_scalar('MLF21.tau / epoch', MLF21.tau.sigmoid(), epoch)

                writer.add_scalar('MLF30.tau / epoch', MLF30.tau.sigmoid(), epoch)

                writer.add_scalar('MLF31.tau / epoch', MLF31.tau.sigmoid(), epoch)
#     这里专门用来存储每层MLF的tau变化



if __name__ == '__main__':
    # 超参数设置  4
    batch_size = 5
    useAdam = False
    #
    parser = argparse.ArgumentParser(description='trian')
    # 32
    parser.add_argument('--batch-size', type=int, default=batch_size, help='input batch size for training')
    parser.add_argument('--test-batch-size', type=int, default=batch_size, help='input batch size for testing')
    # 100
    parser.add_argument('--total-epochs', type=int, default=30, help='number of epochs to train')
    # 0.1
    if useAdam:
        parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    else:
        # 0.01
        parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--use-cuda', action='store_true', default=True, help='use CUDA training')
    # True
    parser.add_argument('--save', action='store_true', default=False, help='save model')
    parser.add_argument('--tensorboard', action='store_true', default=True, help='write tensorboard')
    parser.add_argument('--pretrained', action='store_true', default=False, help='use pre-trained model') ######   needs to be modified  #####
    parser.add_argument('--log-interval', type=int, default=25,
                        help='how many batches to wait before logging training status')
    # 5
    parser.add_argument('--save-model-interval', type=int, default=5,
                        help='save model every save_model_interval')
    parser.add_argument('--checkpoint-path', type=str, default='./checkpoint/dvs_cifar10/result_dvscifar10.pth',
                        help='use CUDA training')
    args = parser.parse_args()
    use_cuda = args.use_cuda and torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.manual_seed(1)
    # 1 True
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    writer = None
    writer_path = './summaries/dvs_cifar10/result_dvscifar10' + '_' + str(len(os.listdir('./summaries/dvs_cifar10')))
    if args.tensorboard:
        writer = SummaryWriter(writer_path)

    model = resnet14().to(device)
    #test
    # print(model.layer1.execute[0].spike_func.tau2)
    # print(model.layer2.execute[0].spike_func.tau2)
    # print(model.layer1.execute[1].spike_func.tau2)

    train_loader1, test_loader1, start_epoch = data_model_load(args, model, kwargs)
    # SGD
    if not useAdam:
        optimizer = optim.SGD(model.parameters(), args.lr, momentum=momentum_SGD, weight_decay=1e-3)
    else:
        optimizer = optim.Adam(model.parameters(), args.lr, weight_decay=0)
    #     40   0.1
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10, 0.1)
    for _ in range(start_epoch):
        scheduler.step()

    for epoch in range(start_epoch + 1, args.total_epochs + 1):
        start_time = time.time()
        train(args, model, train_loader1, optimizer, device, epoch, writer)
        _test(args, model, test_loader1, device, writer)
        waste_time = time.time() - start_time
        print('One epoch wasting time:{:.0f}s, learning rate:{:.8f}\n'.format(
            waste_time, optimizer.state_dict()['param_groups'][0]['lr']))
        if epoch % args.save_model_interval == 0:
            if args.save:
                state = {'model': model.state_dict(), 'epoch': epoch}
                torch.save(state, args.checkpoint_path)
        scheduler.step()

    if args.tensorboard:
        writer.close()
