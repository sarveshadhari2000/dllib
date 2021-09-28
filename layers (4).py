import torch.nn as nn
import torch

class RunningBatchNorm(nn.Module):
    def __init__(self, n_out, mom=0.1, eps=1e-5):
        super().__init__()
        self.mom, self.eps = mom, eps
        self.mults = nn.Parameter(torch.ones (n_out,1,1))
        self.adds  = nn.Parameter(torch.zeros(n_out,1,1))
        self.register_buffer('sums', torch.zeros(1,n_out,1,1))
        self.register_buffer('sqrs', torch.zeros(1,n_out,1,1))
        self.register_buffer('count', tensor(0.))
        self.register_buffer('factor',tensor(0.))
        self.register_buffer('offset',tensor(0.))
        self.batch = 0
        
    def update_stats(self, x):
        bs, nc, *_ = x.shape
        self.sums.detach_()
        self.sqrs.detach_()
        dims = (0,2,3)
        s    = x    .sum(dims, keepdim=True)
        ss   = (x*x).sum(dims, keepdim=True)
        c    = s.new_tensor(x.numel()/nc)
        mom1 = s.new_tensor(1 - (1 - self.mom) / math.sqrt(bs - 1))
        self.sums .lerp_(s , mom1)
        self.sqrs .lerp_(ss, mom1)
        self.count.lerp_(c , mom1)
        self.batch += bs
        means = self.sums / self.count
        variances = (self.sqrs / self.count).sub_(means * means)
        if bool(self.batch < 20): variances.clamp_min_(0.01)
        self.factor = self.mults / (variances + self.eps).sqrt()
        self.offset = self.adds - means * self.factor
        
    def forward(self, x):
        if self.training: self.update_stats(x)
        return x * self.factor + self.offset

class AdaptiveConcatPool2d(nn.Module):
    def __init__(self, size=1):
        super().__init__()
        self.output_size = size
        self.avg_pool = nn.AdaptiveAvgPool2d(size)
        self.max_pool = nn.AdaptiveMaxPool2d(size)
        
    def forward(self, x): return torch.cat([self.max_pool(x), self.avg_pool(x)], dim=1)

def adapt_model(learner, dl, cut_layer,c_out):
    cut = next(i for i, o in enumerate(learner.model.children()) if isinstance(o, cut_layer))
    bottleneck_model = learner.model[:cut]
    for xb,yb in dl: break
    pred = bottleneck_model(xb)
    n_in = pred.shape[1]
    model_new = nn.Sequential(bottleneck_model,
                              AdaptiveConcatPool2d(),
                              nn.Flatten(),
                              nn.Linear(n_in*2, c_out))
    learner.model = model_new

def set_grad(model,state):
  if isinstance(model,(nn.Linear,nn.BatchNorm2d)): return
  if hasattr(model,'weight'):
    for param in model.parameters(): param.requires_grad = state

def param_group_splitter(model,bot_idx,linr_idx):
  
  low_lr_grp,med_lr_grp,high_lr_grp = [],[],[]
  bottleneck = model[bot_idx]
  hf_bottleneck_sz = len(bottleneck)//2
  
  upper_bottleneck = bottleneck[:hf_bottleneck_sz]
  lower_bottleneck = bottleneck[hf_bottleneck_sz:]

  linr_grp = model[linr_idx]

  def _batchnorm_splitter(layer,target_grp,high_grp):
    if isinstance(layer,nn.BatchNorm2d): high_grp += ( list(layer.parameters()) )
    elif hasattr(layer,'weight'): target_grp += ( list(layer.parameters()) )
    for child in layer.children(): _batchnorm_splitter(child,target_grp,high_grp)
  
  for layer in lower_bottleneck:
    _batchnorm_splitter(layer,med_lr_grp,high_lr_grp)
  
  for layer in upper_bottleneck:
    _batchnorm_splitter(layer,low_lr_grp,high_lr_grp)
  
  high_lr_grp += list( linr_grp.parameters() )

  return low_lr_grp,med_lr_grp,high_lr_grp
  

  


  