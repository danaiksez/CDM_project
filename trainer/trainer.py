import numpy as np
import torch, json
import torch.nn.functional as F
import torch.nn as nn

from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, MetricTracker
from sklearn.metrics import precision_recall_fscore_support

torch.autograd.set_detect_anomaly(True)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def compute_prec_rec_f1(output, target):
    with torch.no_grad():
        pred = F.softmax(output, dim=1)
        pred = torch.argmax(pred, dim=1)
        pred = pred.clone().cpu().numpy()
        labs = target.clone().cpu().numpy()
        prec, rec, f1, _ = precision_recall_fscore_support(labs, pred, labels=[0,1,2,3,4,5,6], zero_division=0)
    return prec, rec, f1


class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, device,
                 train_data_loader, valid_data_loader=None, test_data_loader=None, 
                 lr_scheduler=None, len_epoch=None, threeEncoders='none'):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.device = device
        self.train_data_loader = train_data_loader
        self.valid_data_loader = valid_data_loader
        self.test_data_loader = test_data_loader
        self.threeEncoders = threeEncoders
        self.model = model.to(DEVICE)
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.train_data_loader)
        else:
            # iteration-based training
            self.train_data_loader = inf_loop(train_data_loader)
            self.len_epoch = len_epoch
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(train_data_loader.batch_size))

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.to(DEVICE)
        self.model.train()
        self.train_metrics.reset()
        self.model.pos_tags = []

        #for batch_idx, (data, lengths, target, sentences_number) in enumerate(self.train_data_loader):
        for batch_idx, (data, target, actions, speakers) in enumerate(self.train_data_loader):
            # data, spkrs = data.to(DEVICE), spkrs.to(DEVICE)
            target = target.to(DEVICE)
            self.optimizer.zero_grad()
            self.model._init_hidden_state()

            if 'ThreeEncoders' in self.threeEncoders:
                #import pdb; pdb.set_trace()
                output = self.model(data, speakers, heatmap=False, postags=True)
            elif 'GRU' in self.threeEncoders:
                output = self.model(data, heatmap=False, postags=False)
            else:
                output = self.model(data)

            if output.size(0) != target.size(1):
                idx = []
                for i in range(len(data)):
                    if data[i].size(1) == 0:
                        idx.append(i)
                target_numpy = target.clone().cpu()
                target_numpy = np.delete(target_numpy, idx)
                target = torch.tensor(target_numpy).to(DEVICE)

            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())
            for met in self.metric_ftns:
                if met.__name__ == 'accuracy':
                    self.train_metrics.update(met.__name__, met(output, target.squeeze(0)))
                else:
                    classid = int(met.__name__[-1])  # id of the class (number)
                    pred = torch.argmax(F.softmax(output, dim=1), dim=1)
                    if classid in pred:     # update metric score only if class is present in the targets
                        scores = compute_prec_rec_f1(output, target.squeeze(0))
                        self.train_metrics.update(met.__name__, met(scores))


            #if batch_idx % self.log_step == 0:
            if batch_idx % 100 == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))

            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        #import pdb; pdb.set_trace()
        try:
            with open('pos_tags.json', 'w') as _file: 
                json.dump(self.model.pos_tags, _file, indent=6)
        except:
            import pdb; pdb.set_trace()
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, (data, target, acts, spkrs) in enumerate(self.valid_data_loader):
                #data, target = data.to(self.device), target.to(self.device)
                #self.model._init_hidden_state()
                target = target.to(DEVICE)
                if 'ThreeEncoders' in self.threeEncoders:
                    output = self.model(data, spkrs)
                else:
                    output = self.model(data)
                
                if output.size(0) != target.size(1):
                    idx = []
                    for i in range(len(data)):
                        if data[i].size(1) == 0:
                            idx.append(i)
                    target_numpy = target.clone().cpu()
                    target_numpy = np.delete(target_numpy, idx)
                    target = torch.tensor(target_numpy).to(DEVICE)

                loss = self.criterion(output, target)

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    if met.__name__ == 'accuracy':
                        self.valid_metrics.update(met.__name__, met(output, target.squeeze(0)))
                    else:
                        classid = int(met.__name__[-1])  # id of the class (number)
                        pred = torch.argmax(F.softmax(output, dim=1), dim=1)
                        if classid in pred:     # update metric score only if class is present in the targets
                            scores = compute_prec_rec_f1(output, target.squeeze(0))
                            self.valid_metrics.update(met.__name__, met(scores))

            
        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.train_data_loader, 'n_samples'):
            current = batch_idx * self.train_data_loader.batch_size
            total = self.train_data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

