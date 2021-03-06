
import torch.nn as nn

def noop(x): return x

class Flatten(nn.Module):
    def forward(self, x): return x.view(x.size(0), -1)
    
def conv(n_in, n_out, ks=3, stride=1, bias=False):
    return nn.Conv2d(n_in, n_out, kernel_size=ks, stride=stride, padding=ks//2, bias=bias)

act_fn = nn.ReLU(inplace=True)

def init_cnn(m):
    if getattr(m, 'bias', None) is not None: nn.init.constant_(m.bias, 0)
    if isinstance(m, (nn.Conv2d, nn.Linear)): nn.init.kaiming_normal_(m.weight)
    for l in m.children(): init_cnn(l)
        
def conv_layer(n_in, n_out, ks=3, stride=1, zero_bn=False, act=True):
    bn = nn.BatchNorm2d(n_out)
    nn.init.constant_(bn.weight, 0. if zero_bn else 1.)
    layers = [conv(n_in, n_out, ks, stride=stride), bn]
    if act: layers.append(act_fn)
    return nn.Sequential(*layers)

class ResBlock(nn.Module):
    def __init__(self, expansion, n_in, n_hidden, stride=1):
        super().__init__()
        n_out, n_in = n_hidden*expansion, n_in*expansion
        layers = [conv_layer(n_in, n_hidden, 1)]
        layers += [
            conv_layer(n_hidden, n_out, 3, stride=stride, zero_bn=True, act=False)
        ] if expansion == 1 else [
            conv_layer(n_hidden, n_hidden, 3, stride=stride),
            conv_layer(n_hidden, n_out, 1, zero_bn=True, act=False)
        ]
        self.convs = nn.Sequential(*layers)
        self.idconv = noop if n_in == n_out else conv_layer(n_in, n_out, 1, act=False)
        self.pool = noop if stride == 1 else nn.AvgPool2d(2, ceil_mode=True)
        
    def forward(self, x): return act_fn(self.convs(x) + self.idconv(self.pool(x)))

class XResNet(nn.Sequential):
    @classmethod
    def create(cls, expansion, layers, channels_in=3, channels_out=1000):
        n_outs = [channels_in, (channels_in+1)*8, 64, 64]
        stem = [conv_layer(n_outs[i], n_outs[i+1], stride=2 if i==0 else 1)
               for i in range(3)]
        
        n_outs = [64//expansion, 64, 128, 256, 512]
        res_layers = [cls._make_layer(expansion, n_outs[i], n_outs[i+1],
                                      n_blocks=1, stride=1 if i==0 else 2)
                     for i, l in enumerate(layers)]
        res = cls(*stem,
                  nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                  *res_layers,
                  nn.AdaptiveAvgPool2d(1),
                  Flatten(),
                  nn.Linear(n_outs[-1]*expansion, channels_out)
        )
        init_cnn(res)
        return res
    
    @staticmethod
    def _make_layer(expansion, n_in, n_out, n_blocks, stride):
        return nn.Sequential(
            *[ResBlock(expansion, n_in if i==0 else n_out, n_out, stride if i==0 else 1)
             for i in range(n_blocks)])


def XResNet18 (**kwargs): return XResNet.create(1, [2, 2,  2, 2], **kwargs)
def XResNet34 (**kwargs): return XResNet.create(1, [3, 4,  6, 3], **kwargs)
def XResNet50 (**kwargs): return XResNet.create(4, [3, 4,  6, 3], **kwargs)
def XResNet101(**kwargs): return XResNet.create(4, [3, 4, 23, 3], **kwargs)
def XResNet152(**kwargs): return XResNet.create(4, [3, 8, 36, 3], **kwargs)
