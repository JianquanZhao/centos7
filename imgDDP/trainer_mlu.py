from torchmetrics import Metric
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import logging
import os
# from torch.utils.tensorboard import SummaryWriter

from mymodel.MyModel import MyConvNet
from dataset.MyDataLoader import *

import torch_mlu.core.mlu_quantize as qt
import torch_mlu.core.mlu_model as ct


logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logging_formatter = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
ch = logging.StreamHandler()
ch.setFormatter(logging_formatter)
logger.addHandler(ch)


# writer = SummaryWriter(f'./logs')

class MyLoss(Metric):
    # def __init__(self, dist_sync_on_step=False):
    def __init__(self, dist_sync_on_step=True):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("loss_sum", default=torch.tensor(0, dtype=torch.float), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, loss: torch.tensor):
        self.loss_sum += loss
        self.total += 1

    def compute(self):
        return self.loss_sum.float() / self.total


class MyAccuracy(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    # def update(self, preds: torch.Tensor, target: torch.Tensor):
    def update(self, preds: torch.Tensor, target: torch.Tensor):
        # preds, target = self._input_format(preds, target)
        # assert preds.shape == target.shape

        self.correct += torch.sum(preds == target)
        self.total += target.numel()

    def compute(self):
        return self.correct.float() / self.total


def train_model(rank, world_size):
    '''
    :param rank: GPU 数量，the number of GPU
    :param world_size: 一个进程组的进程数量,the number of process
    :return: none
    '''

    '''
    单机多卡中设置进程通讯的系统环境变量，包括IP：port
    set environment paras in DDP so that the process of process group can communicate 
    '''
    ct.set_device(rank)
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '34534'  # 57834
    os.environ['CNCL_SEND_RECV_ENABLE'] = '1'
    '''
    进程组初始化，包括通讯后端、进程数、卡数
    通讯后端不支持nccl
    '''
    dist.init_process_group(backend='cncl', rank=rank, world_size=world_size)
    model = MyConvNet()
    model_quantize = qt.adaptive_quantize(model = model,)
    model_mlu = model.to(ct.mlu_device())
    ddp_model = DDP(model_mlu,device_ids=[rank])
    # ddp_model = DDP_mlu(model,process_group = dist_mlu.get_mlu_default_group())
    # dist_mlu.init_process_group('cncl', rank=rank, world_size=world_size)

    '''
    模型初始化，包括优化器、损失函数、评价标准
    '''
    logger.info(f'')
    logger.info(f'initlize model  ...')
    '''
    先进行
    '''

    # optimizer = torch.optim.Adam(model.parameters(), lr=0.0003, weight_decay=0.01)
    optimizer = torch.optim.Adam(ddp_model.parameters(), lr=0.0003, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    myloss_cpu = MyLoss()
    myacc_cpu = MyAccuracy()

    # myloss = myloss_cpu.to(rank, non_blocking=True)
    # myacc = myacc_cpu.to(rank, non_blocking=True)
    logger.info(f"running on the rank {rank} and the device is {ct.mlu_device()}")
    myloss = myloss_cpu.to(ct.mlu_device(), non_blocking=True)
    myacc = myacc_cpu.to(ct.mlu_device(), non_blocking=True)
    optimizer = ct.to(optimizer,ct.mlu_device())
    criterion = criterion.to(ct.mlu_device())

    '''
    first complete the cpu version 
    then the mlu version that set model,optimizer,and loss funciton to mlu
    
    '''

    '''
    引入loader，并按照进程进程划分，每个进程分得一部分的数据同时进行计算
    '''
    logger.info(f'load dataloader')
    traindataset, testdataset = DataProcess()
    traindataset_rank = split_dataset(traindataset, rank)
    testdataset_rank = split_dataset(testdataset, rank)
    traindata_loader, testdata_loader = make_dataloader(traindataset_rank, testdataset_rank)
    num_epochs = 10
    '''
    训练过程
    '''

    def train_step():
        for step, (b_x, b_y) in enumerate(traindata_loader):
            ddp_model.train()
            b_x = b_x.to(ct.mlu_device())
            b_y = b_y.to(ct.mlu_device())
            output = ddp_model(b_x)
            pre_lab = torch.argmax(output, 1)
            loss = criterion(output, b_y)

            myacc(pre_lab, b_y)
            myloss(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss_mean = myloss.compute()
        acc_mean = myacc.compute()

        if (rank == 0):
            # logger.info(f'train epoch {epoch} step {step}')
            # writer.add_scalar('Loss/train', loss_mean, step)
            # writer.add_scalar('Acc/train', acc_mean, step)
            logger.info(f'rank {rank} epoch {epoch} train Loss: {loss_mean} train acc: {acc_mean}')
        myacc.reset()
        myloss.reset()

    def validate():
        for step, (v_x, v_y) in enumerate(testdata_loader):
            model.eval()
            v_x = v_x.to(ct.mlu_device())
            v_y = v_y.to(ct.mlu_device())
            output = ddp_model(v_x)
            pre_lab = torch.argmax(output, 1)
            loss = criterion(output, v_y)
            myacc(pre_lab, v_y)
            myloss(loss.item())

        loss_mean = myloss.compute()
        acc_mean = myacc.compute()

        if (rank == 0):
            # logger.info(f'train epoch {epoch} step {step}')

            # writer.add_scalar('Loss/validate', loss_mean, step)
            # writer.add_scalar('Acc/validate', acc_mean, step)
            logger.info(f'rank {rank} epoch {epoch} Validate Loss: {loss_mean} Validate acc: {acc_mean}')

    for epoch in range(num_epochs):
        train_step()
        if epoch+1 % 10 == 0:
            validate()

    # writer.close()
    dist.destroy_process_group()
    # dist_mlu.destroy_process_group()


if __name__ == "__main__":
    world_size = 2
    mp.spawn(train_model
             , args=(world_size,)
             , nprocs=world_size
             , join=True
             )

