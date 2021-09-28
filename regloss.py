import torch.nn as nn
import torch.nn.functional as F

def lincomb(a,b,e):
  return (a*e) + ((1-e)*b)

def reduce_loss(loss,reduction="mean"):
  
  if reduction == "mean": return loss.mean()
  elif reduction == "sum" : return loss.sum()
  else : return loss

class LabelSmoothingCrossEntropy(nn.Module):

  def __init__(self,epsilon=0.1,reduction='mean'):
    super().__init__()
    self.epsilon = epsilon
    self.reduction = reduction

  def forward(self,output,target):

    n = output.shape[-1]
    log_preds = F.log_softmax(output,dim=-1)
    log_loss = reduce_loss(-log_preds.sum(dim=-1),self.reduction)
    nll = F.nll_loss(log_preds,target,reduction=self.reduction)

    return lincomb(log_loss/n,nll,self.epsilon)


