import torch

def accuracy(out,targets):
  acc = ( torch.max(out,dim=1)[1] == targets ).float().mean().cpu()
  return acc