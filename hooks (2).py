from functools import partial
import torch

class ListContainer():

  def __init__(self,items): self.items = list(items)

  def __getitem__(self,idx):
    
    try : 
      return self.items[idx]

    except Exception:

      if isinstance(idx[0],bool):

        if len(idx) != len(self.items):
          raise Exception('Internal List Length Not Equal to Index Length')

        return [self.items[i] for i,m in enumerate(idx) if m]
      
      else : return [self.items[i] for i in idx]
    
  
  def __len__(self):
    return len(self.items)
  
  def __delitem__(self,idx):
    del (self.items[i])
  
  def __iter__(self): return iter(self.items)

  def __setitem__(self,idx,val):
    self.items[idx] = val
  
  def __repr__(self):
        res = f'{self.__class__.__name__} ({len(self)} items)\n{self.items[:10]}'
        if len(self) > 10: res = res[:-1] + '...]'
        return res

class ForwardHooks(ListContainer):

  def __init__(self,modules,f): 
    super().__init__([ForwardHook(m,f) for m in modules])
  
  def remove(self):
    for hook in self:
      hook.remove()
  
  def __del__(self):
    self.remove()
  
  def __enter__(self) : return self

  def __exit__(self): self.remove()


class ForwardHook():
  def __init__(self,l,f):
    self.hook = l.register_forward_hook(partial(f,self))
  def remove(self): self.hook.remove()
  def __del__(self): self.remove()

def append_stats(hook,module,inp,outp):

  if not hasattr(hook,'stats'): hook.stats = ([],[])

  means,std = hook.stats

  if module.training:
    means.append(outp.data.mean().item())
    std.append(outp.data.std().item())

  
