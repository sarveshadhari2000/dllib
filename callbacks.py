import time
from fastprogress import master_bar,progress_bar
from fastprogress.fastprogress import format_time
from functools import partial
import torch
import math
from torch.distributions import Beta
import matplotlib.pyplot as plt

class Learner():

  def __init__(self,model,opt,criterion,dls,cbs=None):
    
    self.model = model
    self.opt = opt
    self.criterion = criterion
    self.dls = dls
    self.cbs = cbs

    for cb in self.cbs : 
      cb.set_runner(self)
      setattr(self,cb._name,cb)
  


  def __call__(self,k):

    for cb in sorted(self.cbs,key=lambda x:x._order,reverse=False):
      cb(k)


  
  def one_batch(self,i,x,y):

    self.bs = x.shape[0]

    self.x,self.labels = x,y
    self.bi = i
    
    if self('before_batch'): return 
    
    
    self.preds = self.model(self.x)


    self.loss = self.criterion(self.preds,self.labels)

    if self.in_train:

      self.opt.zero_grad()
      self.loss.backward()
      self.opt.step()
    
    
    if self('after_batch'): return
    
  def all_batches(self,dl):

    
    for i,(x,y) in enumerate(self.c_dl):
      self.one_batch(i,x,y)
  
  def fit(self,epochs):

    self.epochs = epochs

    for cb in self.cbs : cb.set_runner(self)

    self.total_iters = len(self.dls['train']) * self.epochs

    if self('before_fit'): return 

    for epoch in range(epochs):

      self.c_dl = self.dls['train']
      
      if self('before_epoch'):return

      self.all_batches(self.dls['train'])


      self.c_dl = self.dls['validate']
      if self('before_validate'):return 

      with torch.no_grad():
        self.all_batches(self.dls['validate'])
      
      if self('after_validate'): return
      
      if self('after_epoch'):return
    
    if self('after_fit'): return 


class Callbacks():

  def set_runner(self,runner): self.run = runner
  def __call__(self,k):

    f = getattr(self,k,None)

    if f and f(): return True

    else : return False




class TrainEvalCallback(Callbacks):

  _order = 0
  _name = 'train_eval'

  

  def before_epoch(self):
    self.run.model.train()
    self.run.in_train = True
    

  
  def before_validate(self):
    self.run.model.eval()
    self.run.in_train = False
  
  def after_batch(self):
    if self.run.in_train:
      self.run.c_iter += 1
    

  
  def before_fit(self):
    self.run.c_epoch = 0
    self.run.c_iter = 0
    
    
  
  def after_epoch(self):
    self.run.c_epoch += 1

class Stats():

  def __init__(self,metrics,metric_names,train):
    self.metrics = metrics
    self.metrics_values = ([0.0]*len(self.metrics))
    self.metric_names = metric_names
    self.loss = 0
    self.count = 0
    self.train = train
  
  def reset(self):
    self.loss = 0
    self.count = 0
    self.metrics_values.clear()
    self.metrics_values = ([0.0]*len(self.metrics))

  def all_stats(self):
    return  [self.loss] + self.metrics_values 
  
  def get_avg_stats(self):

    return [ round( o/self.count,5 ) for o in self.all_stats()]
  
  def accumalate(self,run):
    
    bs = run.bs

    self.loss = run.loss.item()*bs
    self.count += bs

    for i,m in enumerate(self.metrics):
      self.metrics_values[i] += ( m(run.preds,run.labels)*bs ).item()

  
  def __repr__(self):
    if not self.count: return ""

    return ( f"{'train' if self.train else 'valid'}: {self.get_avg_stats()}" )


class ProgressBarCallback(Callbacks):

  _order = 0
  _name = 'progress_bar'

  def before_fit(self):
    print(f'Device Type : {self.run.cuda.device}')
    self.mb = master_bar( range(self.run.epochs) )
    self.mb.on_iter_begin()
    self.run.logger = partial(self.mb.write,table=True)
  
  def before_epoch(self):
    self.set_progress()
  
  def before_validate(self):
    self.set_progress()
  
  def after_batch(self):
    self.pb.update( self.run.bi )
  
  def after_validate(self):
    self.mb.update( self.run.c_epoch )
  
  def after_fit(self):
    self.mb.on_iter_end()
  


  def set_progress(self):

    self.pb = progress_bar(range(len(self.run.c_dl)),parent=self.mb,auto_update=False)
    




class StatsCallback(Callbacks):

  _order = 0
  _name = 'stats'


  def __init__(self,metrics,metric_names):
    self.train_stats = Stats(metrics,metric_names,True)
    self.validate_stats = Stats(metrics,metric_names,False)
  
  def before_fit(self):
    
    metrics = ['loss'] + [m for m in self.train_stats.metric_names]
    names = [f'train_{n}' for n in metrics] + [f'validate_{n}' for n in metrics] + ['time']
    
    self.run.logger(names)

  def before_epoch(self):
    self.train_stats.reset()
    self.validate_stats.reset()
    self.start_time  = time.time()
  
  def after_batch(self):

    if self.run.in_train :
      self.train_stats.accumalate(self.run)

    else :
      self.validate_stats.accumalate(self.run)
  

  def after_epoch(self):
    
    stats = self.train_stats.get_avg_stats() + self.validate_stats.get_avg_stats() + [format_time(time.time()-self.start_time)]
    self.run.logger(stats)

class ParamScheduler(Callbacks):

  _order = 1
  _name = 'param_scheduler'

  def __init__(self,param_name,sched_func,index=None):
    self.param_name = param_name
    self.sched_func = sched_func
    self.index = index
  
  def before_fit(self):

    if not isinstance(self.sched_func,(list,tuple)):
      self.sched_func = [self.sched_func]*len(self.run.opt.param_groups)
  
  def before_batch(self):
    if self.run.in_train:
      for pg,f in zip(self.run.opt.param_groups,self.sched_func):

        if self.index is not None :

          v = f(self.run.c_iter/self.run.total_iters)
          
          if isinstance(pg[self.param_name],tuple):
            p = list(pg[self.param_name])
            p[self.index] = v
            pg[self.param_name] = tuple(p)

          else : pg[self.param_name][self.index] = v

        else : pg[self.param_name] = f(self.run.c_iter/self.run.total_iters)

class pytorchSchedulers(Callbacks):

  _order = 1
  _name = 'pytorch_scheduler'

  def __init__(self,sched):
    self.sched = sched

  def before_batch(self):

    if self.run.in_train:
      self.sched.step()





class Recorder(Callbacks):
  _order = 1
  _name = 'recorder'

  def before_fit(self):
    self.lrs,self.losses = [ [] for i in range(len(self.run.opt.param_groups))],[]
  
  def after_batch(self):
    if self.run.in_train:
      self.losses.append(self.run.loss.item())

      for pg,lr in zip(self.run.opt.param_groups,self.lrs):

        lr.append(pg['lr'])
      
  
  def plot_loss(self):
    plt.plot(self.losses)
  
  def plot_lr(self,pgid=-1):
    plt.plot(self.lrs[pgid])

class CudaCallback(Callbacks):
  
  _order = 2
  _name = 'cuda'

  def __init__(self,device):
    self.device = device

  def before_fit(self):
    
    self.run.model = self.run.model.to(self.device)

  def before_batch(self):
    self.run.x = self.run.x.to(self.device)
    self.run.labels = self.run.labels.to(self.device)

def annealer(f):
  def _inner(start,end): return partial(f,start,end)
  return _inner

@annealer
def sched_lin(start,end,pos):
  return start + pos*(end-start)


#export
@annealer
def sched_cos(start, end, pos): 
    return start + (1 + math.cos(math.pi*(1-pos))) * (end-start) / 2

@annealer
def sched_exp(start, end, pos):
    return start * (end/start) ** pos

@annealer
def sched_none(start, end, pos): return start

def combine_scheds(pcts,scheds):
  assert sum(pcts)==1
  pcts = [0.0]+pcts
  pcts = torch.Tensor(pcts)
  assert torch.all(pcts>=0)
  pcts = torch.cumsum(pcts,0)

  def _inner(pos):

    idx = (pos>=pcts).nonzero().max()
    actual_pos = (pos-pcts[idx])/(pcts[idx+1]-pcts[idx])

    return scheds[idx](actual_pos)
  
  return _inner

class NoneReduction():

  def __init__(self,loss_fn):
    self.loss_fn = loss_fn
    self.old_reduction = None
  
  def __enter__(self):

    if hasattr(self.loss_fn,'reduction'):

      self.old_reduction = getattr(self.loss_fn,'reduction')
      setattr(self.loss_fn,'reduction','none')
      return self.loss_fn
    
    else : return partial(self.loss_fn,reduction='none')
  
  def __exit__(self, type, value, traceback):

    if self.old_reduction is not None: 
      setattr(self.loss_fn,'reduction',self.old_reduction)
    
def lincomb(a,b,e):
  return (a*e) + ((1-e)*b)

def reduce_loss(loss,reduction="mean"):
  
  if reduction == "mean": return loss.mean()
  elif reduction == "sum" : return loss.sum()
  else : return loss

def unsqueeze(x,dims):

  for i in range(dims):
    x = x.unsqueeze(-1)
  
  return x

class MixUp(Callbacks):

  _order = 4
  _name = 'mixup'

  def __init__(self,a=0.4,dims_expand=1): 
    self.distrib = Beta( torch.Tensor([a]),torch.Tensor([a]) )
    self.dims_expand = dims_expand

  def before_fit(self):
    self.old_loss_func = self.run.criterion
    self.run.criterion = self.loss_func
  
  def after_fit(self):
    self.run.criterion = self.old_loss_func
  
  def before_batch(self):

    if not self.run.in_train: return

    t = self.distrib.sample((self.run.bs,)).squeeze().to(self.run.cuda.device)
    t = torch.stack([t,1-t],1)
    t = torch.max(t,dim=1)[0]

    self.t = unsqueeze(t,self.dims_expand)

    shuffle = torch.randperm(self.run.bs).to(self.run.cuda.device)

    xs = self.run.x[shuffle]
    mlabels = self.run.labels[shuffle]

    self.run.x = lincomb(self.run.x,xs,self.t)
    self.run.mlabels = mlabels

    


  
  def loss_func(self,preds,labels):

    if not self.run.in_train : return self.old_loss_func(preds,labels)

    with NoneReduction(self.old_loss_func) as criterion:

      loss1 = criterion(self.run.preds,self.run.labels)
      loss2 = criterion(self.run.preds,self.run.mlabels)
    
    loss = lincomb(loss1,loss2,self.t)
    return reduce_loss(loss,reduction=getattr(self.old_loss_func,'reduction','mean'))




def cycle1_lr(lrs,pct,mom_start=0.95,mom_mid=0.85,mom_end=0.95):
  
  sched_lrs = []

  for lr in lrs:
    sched_lrs.append( combine_scheds([pct,1-(pct)],cos_1cycle(lr/10.0,lr,lr/1e5)) )
  
  sched_moms = combine_scheds([pct,1-(pct)],cos_1cycle(0.95,0.85,0.95))

  return sched_lrs,sched_moms


def cos_1cycle(start,high,end):
  return [sched_cos(start,high),sched_cos(high,end)]

    
  






  




  

    



     
