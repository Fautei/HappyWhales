import torch
from torch import nn, Tensor
from pytorch_lightning import Callback, LightningModule, Trainer
import timm
import torchmetrics
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt


EMBEDDING_SIZE = 512
N_CLASSES = 15587
#N_CLASSES = 96
CENTERS_PER_CLASS = 3
S = 20
MIXED_PRECISION = True


class Swish(torch.autograd.Function):

    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class Swish_module(nn.Module):
    def forward(self, x):
        return Swish.apply(x)
    
class DenseCrossEntropy(nn.Module):
    def forward(self, x, target):
        x = x.float()
        target = target.float()
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        loss = -logprobs * target
        loss = loss.sum(-1)
        return loss.mean()

class ArcMarginProduct_subcenter(nn.Module):
    def __init__(self, in_features=EMBEDDING_SIZE, out_features=N_CLASSES, k=CENTERS_PER_CLASS):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(out_features*k, in_features))
        self.reset_parameters()
        self.k = k
        self.out_features = out_features
        
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        
    def forward(self, features):
        cosine_all = F.linear(F.normalize(features), F.normalize(self.weight))
        cosine_all = cosine_all.view(-1, self.out_features, self.k)
        cosine, _ = torch.max(cosine_all, dim=2)
        return cosine   
    
class ArcFaceLossAdaptiveMargin(nn.modules.Module):
    def __init__(self, margins, out_dim=N_CLASSES, s=S):
        super().__init__()
        self.crit = DenseCrossEntropy()
        self.s = s
        self.register_buffer('margins', torch.tensor(margins))
        self.out_dim = out_dim
            
    def forward(self, logits, labels):
        #ms = []
        #ms = self.margins[labels.cpu().numpy()]
        ms = self.margins[labels]
        cos_m = torch.cos(ms) #torch.from_numpy(np.cos(ms)).float().cuda()
        sin_m = torch.sin(ms) #torch.from_numpy(np.sin(ms)).float().cuda()
        th = torch.cos(math.pi - ms)#torch.from_numpy(np.cos(math.pi - ms)).float().cuda()
        mm = torch.sin(math.pi - ms) * ms#torch.from_numpy(np.sin(math.pi - ms) * ms).float().cuda()
        labels = F.one_hot(labels, self.out_dim)
        labels = labels.half() if MIXED_PRECISION else labels.float()
        cosine = logits
        sine = torch.sqrt(1.0 - cosine * cosine)
        phi = cosine * cos_m.view(-1,1) - sine * sin_m.view(-1,1)
        phi = torch.where(cosine > th.view(-1,1), phi, cosine - mm.view(-1,1))
        output = (labels * phi) + ((1.0 - labels) * cosine)
        output *= self.s
        loss = self.crit(output, labels)
        return loss

class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1)*p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)
        
    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)


class PeakScheduler(torch.optim.lr_scheduler._LRScheduler):
        def __init__(
                self, optimizer,
                epoch_size=-1,
                lr_start   = 0.00001,
                lr_max     = 0.2,
                lr_min     = 0.00001,
                lr_ramp_ep = 4,
                lr_sus_ep  = 0,
                lr_decay   = 0.8,
                verbose = True
            ):
            self.epoch_size = epoch_size
            self.optimizer= optimizer
            self.lr_start = lr_start
            self.lr_max = lr_max
            self.lr_min = lr_min
            self.lr_ramp_ep = lr_ramp_ep
            self.lr_sus_ep = lr_sus_ep
            self.lr_decay = lr_decay
            #self.is_plotting = True
            #epochs = list(range(CFG.EPOCHS))
            #learning_rates = []
            #for i in epochs:
            #    self.epoch = i
            #    learning_rates.append(self.get_lr())
            #self.is_plotting = False
            #self.epoch = 0
            #plt.scatter(epochs,learning_rates)
            #plt.show()
            super(PeakScheduler, self).__init__(optimizer, verbose=verbose)

        def get_lr(self):
            
            if self.epoch_size == -1:
                self.epoch = self._step_count - 1
            else:
                self.epoch = (self._step_count - 1) / self.epoch_size
                    
            if self.epoch < self.lr_ramp_ep:
                lr = (self.lr_max - self.lr_start) / self.lr_ramp_ep * self.epoch + self.lr_start

            elif self.epoch < self.lr_ramp_ep + self.lr_sus_ep:
                lr = self.lr_max
            else:
                lr = (self.lr_max - self.lr_min) * self.lr_decay**(self.epoch - self.lr_ramp_ep - self.lr_sus_ep) + self.lr_min
            return [lr for _ in self.optimizer.param_groups]



#using basemodel for pretraining on low resolutions strong augmented images
class BaseModel(LightningModule):
    def __init__(self,feature_extractor, classify = False):
        super().__init__()
        self.classify = classify
        self.conv1 = nn.Conv2d(3,3,kernel_size=(7,7),stride=(2,2),padding = (3,3),bias = False)
        self.bn1 = nn.BatchNorm2d(3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu1 = nn.SiLU(inplace=True)
        self.gem = GeM()
        self.feature_extractor = timm.create_model(feature_extractor, pretrained=True)
        out_features = self.feature_extractor.classifier.out_features
        self.silu2 = nn.SiLU(inplace=True)
        self.dropout1 = nn.Dropout(p = 0.6,inplace=True)
        self.emb = nn.Linear(out_features, EMBEDDING_SIZE, bias = False)
        self.logits = nn.Linear(EMBEDDING_SIZE, N_CLASSES, bias = False)
        self.criterion = nn.CrossEntropyLoss()
        self.train_acc = torchmetrics.Accuracy()
        self.train_top_k_acc = torchmetrics.Accuracy(top_k=5)
        self.val_acc = torchmetrics.Accuracy()
        self.val_top_k_acc = torchmetrics.Accuracy(top_k=5)
        self.learning_rate = 1e-3

    def forward(self, image):
        """
            Return embedding of the images
        """
        x = self.conv1(image)
        x = self.bn1(x)
        x = self.silu1(x)
        x = self.gem(x)
        x = self.feature_extractor(x)
        x = self.silu2(x)
        x = self.dropout1(x)
        x = self.emb(x)
        if self.classify:
            x = self.logits(x)
        else:
            x = F.normalize(x, p=2, dim=1)
        return x

    def training_step(self, batch, batch_idx):
        """
            Return the loss to do a step on
        """
        img, labels = batch
        preds = self(img)
        loss  = self.criterion(preds, labels)
        # Log metrics
        self.train_acc(preds, labels)
        self.train_top_k_acc(preds, labels)
        self.log("train_loss", loss)
        self.log("train_acc", self.train_acc)
        self.log("train_top_k_acc", self.train_top_k_acc)
        # Return loss, labels and preds
        return {"loss": loss, "preds": preds.detach(), "targets": labels.detach()}
    
    def validation_step(self, batch, batch_idx):
        img, labels = batch
        preds = self(img)
        loss  = self.criterion(preds, labels)
        # Log metrics
        self.train_acc(preds, labels)
        self.train_top_k_acc(preds, labels)
        self.log("val_loss", loss)
        self.log("val_acc", self.val_acc)
        self.log("val_top_k_acc", self.val_top_k_acc)
        # Return loss, labels and preds
        return {"loss": loss, "preds": preds.detach(), "targets": labels.detach()}


    def configure_optimizers(self):
        """
            Build optimizer(s) and lr scheduler(s)
        """
        optimizer = torch.optim.AdamW(self.parameters(), lr = self.learning_rate)
        
        sched = {
            "scheduler": torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9),
            "interval": "step",
        }
        return {
            "optimizer": optimizer,
            "lr_scheduler": sched
        }


class WandDIDNet(LightningModule):
    def __init__(self,model_path,model_name,margins):
        super().__init__()

        
        self.model = BaseModel.load_from_checkpoint(model_path,feature_extractor= model_name)
        BaseModel.classify = False
        
        self.swish = Swish_module()
        self.metric_classify = ArcMarginProduct_subcenter(EMBEDDING_SIZE)
        # Loss
        self.criterion = ArcFaceLossAdaptiveMargin(margins=margins)
        # Metrics
        self.train_acc = torchmetrics.Accuracy()
        self.train_top_k_acc = torchmetrics.Accuracy(top_k=5)
        self.val_acc = torchmetrics.Accuracy()
        self.val_top_k_acc = torchmetrics.Accuracy(top_k=5)
        self.learning_rate = 1e-3

    def forward(self, image):
        """
            Return embedding of the images
        """
        return self.model(image)
    
    def training_step(self, batch, batch_idx):
        """
            Return the loss to do a step on
        """
        img, label = batch
        embedding = self(img)
        logits = self.metric_classify(embedding)
        loss  = self.criterion(logits, label)
        # Log metrics
        self.train_acc(logits, label)
        self.train_top_k_acc(logits, label)
        self.log("train_loss", loss)
        self.log("train_acc", self.train_acc)
        self.log("train_top_k_acc", self.train_top_k_acc)
        # Return loss, labels and preds
        return {"loss": loss, "preds": logits.detach(), "targets": label.detach()}

    def validation_step(self, batch, batch_idx):
        img, label = batch
        embedding = self(img)
        logits = self.metric_classify(embedding)
        loss  = self.criterion(logits, label)
        # Log metrics
        self.val_acc(logits, label)
        self.val_top_k_acc(logits, label)
        self.log("val_loss", loss)
        self.log("val_acc", self.val_acc)
        self.log("val_top_k_acc", self.val_top_k_acc)
        # Return loss, labels and preds
        return {"loss": loss, "preds": preds.detach(), "targets": labels.detach()}

    def configure_optimizers(self):
        """
            Build optimizer(s) and lr scheduler(s)
        """
        optimizer = torch.optim.AdamW(self.parameters())

        sched = {
            "scheduler": PeakScheduler(optimizer),
            "interval": "epoch",
        }
        return {
            "optimizer": optimizer,
            "lr_scheduler": sched
        }
    
    def save_class_weights(self):
        """
            Save the class centers as a tensor
        """
        torch.save(self.metric_classify.weight, 'class_weights.pt')