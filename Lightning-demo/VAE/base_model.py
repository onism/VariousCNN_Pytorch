from torch import nn 
from abc import abstractclassmethod 

class BaseVAE(nn.Module):
    def __init__(self):
        super(BaseVAE, self).__init__()
    
    def encoder(self, input):
        raise NotImplementedError
    
    def decoder(self, input):
        raise NotImplementedError

    def sample(self, batch_size, device, **kwargs):
        raise RuntimeWarning()

    def generate(self, x, **kwargs):
        raise NotImplementedError

    @abstractclassmethod
    def forward(self, *inputs):
        pass 

    @abstractclassmethod
    def loss_function(self, *inputs, **kwargs):
        pass