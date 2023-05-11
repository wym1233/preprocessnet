import os, time, logging
import torch
from torch.utils.tensorboard import SummaryWriter
import time
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Trainer():
    def __init__(self, config, model,criterion):

        self.config = config
        self.logger = self.getlogger(config.logdir)

        self.criterion = criterion
        self.model = model.to(device)
        self.logger.info(model)
        self.load_state_dict()
        self.epoch = 0
        self.record_set = {'msenum':[0,0,0,0,0],'msesum':sum}
        self.record_set['msesum']=.0

        if not os.path.exists("/output/logs"):
            os.makedirs("/output/logs")

    def getlogger(self, logdir):
        logger = logging.getLogger(__name__)
        logger.setLevel(level = logging.INFO)
        handler = logging.FileHandler(os.path.join(logdir, 'log.txt'))
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s: %(message)s', datefmt='%m/%d %H:%M:%S')
        handler.setFormatter(formatter)
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        console.setFormatter(formatter)
        logger.addHandler(handler)
        logger.addHandler(console)
        return logger

    def load_state_dict(self):
        """selectively load model
        """
        if self.config.init_ckpt=='':
            self.logger.info('No ckpt load')
        else:
            ckpt = torch.load(self.config.init_ckpt)
            self.model.load_state_dict(ckpt['model'])
            self.logger.info('Load checkpoint from ' + self.config.init_ckpt)
        return

    def save_model(self):
        self.logger.info('Saving model...')
        filename=os.path.join(self.config.ckptdir, 'epoch_' + str(self.epoch) + '.pth')
        torch.save({'model': self.model.state_dict()}, filename)
        self.logger.info('Saved as '+str(filename)+ '.pth')
        return

    def set_optimizer(self):
        params_lr_list = []
        for module_name in self.model._modules.keys():
            params_lr_list.append({"params":self.model._modules[module_name].parameters(), 'lr':self.config.lr})
        optimizer = torch.optim.Adam(params_lr_list, betas=(0.9, 0.999), weight_decay=1e-4)
        optimizer = torch.optim.SGD(params_lr_list,lr=self.config.lr)

        return optimizer

    @torch.no_grad()
    def test(self, dataloader):
        self.logger.info('-------------------------------------------')
        self.logger.info('Start Testing Epoch ' + str(self.epoch))
        self.logger.info('Testing Files length:' + str(len(dataloader)))
        self.record_set['msesum']=0
        for batch_step, (images, bpp) in enumerate(dataloader):
            bpp = bpp.to(device)/1000
            images = images.to(device)
            out = self.model(images)
            pred=torch.argmax(input=out,dim=1)
            pred=pred/1000
            # print(pred,bpp)
            self.record_set['msesum']+=torch.sum(input=torch.abs(bpp-pred)).item()
            torch.cuda.empty_cache()

        self.logger.info('Testing done')
        self.logger.info('sumloss:' + str(self.record_set['msesum']))
        self.logger.info('Batch step: ' + str(batch_step))

        return


    def train(self, dataloader):
        self.logger.info('-------------------------------------------')
        self.logger.info('Start Training Epoch: '+str(self.epoch))
        self.logger.info('Training Files length:' + str(len(dataloader)))
        self.optimizer = self.set_optimizer()
        logdir = os.path.join('/output', 'logs', str(self.epoch))

        writer=SummaryWriter(logdir)
        t0=time.time()
        self.record_set['msesum']=0
        for batch_step, (images,bpp) in enumerate(dataloader):
            bpp=bpp.to(device)
            bpp = bpp.view(-1).long()
            images = images.to(device)
            self.optimizer.zero_grad()
            out = self.model(images)
            mse = self.criterion(out, bpp)
            mse.backward()
            self.optimizer.step()
            torch.cuda.empty_cache()
            writer.add_scalar('scalar/train_loss', mse, (batch_step + 1 + self.epoch * len(dataloader)))
            self.record_set['msesum']+=mse.item()
            if (batch_step % 3000 == 0):
                self.logger.info(str(batch_step+1) + ' images have been trained in this epoch   '
                                 + str(time.time() - t0) + 's consumed')
                t0=time.time()


        self.logger.info('.....Training Done')
        self.logger.info('Batch step: ' + str(batch_step))
        self.logger.info('sumloss:'+str(self.record_set['msesum']))
        self.save_model()
        return
