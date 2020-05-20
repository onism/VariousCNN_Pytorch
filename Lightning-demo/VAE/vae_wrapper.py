'''
from https://github.com/AntixK/PyTorch-VAE

'''

import torch
from torch import optim
import pytorch_lightning as pl 
from torchvision import transforms
import torchvision.utils as vutils
from torchvision.datasets import CelebA
from torch.utils.data import DataLoader

class VAEWrapper(pl.LightningModule):

    def __init__(self, vae_model, params:dict) -> None:
        super(VAEWrapper, self).__init__()
        self.model = vae_model 
        self.params = params 
        self.device = None 
    
    def forward(self, input, **kwargs):
        return self.model(input, **kwargs)
    
    def training_step(self, batch, batch_idx, optimizer_idx=0):
        real_img, labels = batch 
        self.device = real_img.device
        results = self.forward(real_img, labels=labels)
        train_loss = self.model.loss_function(*results, M_N=self.params['batch_size']/self.num_train_imgs,
                                                        optimizer_idx=optimizer_idx, batch_idx=batch_idx)
        self.logger.experiment.log({key:val.item() for key, val in train_loss.items()})
        return train_loss 
    
    def validation_step(self, batch_idx, optimizer_idx=0):
        real_img, labels = batch 
        self.device = real_img.device
        results = self.forward(real_img, labels=labels)
        val_loss = self.model.loss_function(*results, M_N=self.params['batch_size'].self.num_val_imgs,
                                            optimizer_idx=optimizer_idx, batch_idx=batch_idx)
        return val_loss 
    
    def validation_step(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        tensorboard_logs = {'avg_val_loss':avg_loss}
        self.sample_images()
        return {'val_loss':avg_loss, 'log':tensorboard_logs}
    
    def sample_images(self):
        test_input, test_label = next(iter(self.sample_dataloader))
        test_input = test_input.to(self.device)
        test_label = test_label.to(self.device)
        recons = self.model.generate(test_input, labels=test_label)
        vutils.save_image(recons.data, f"{self.logger.save_dir}{self.logger.name}/version_{self.logger.version}/"
                          f"recons_{self.logger.name}_{self.current_epoch}.png",
                          normalize=True,
                          nrow=12)
        samples = self.model.sample(144,
                                        self.curr_device,
                                        labels = test_label)
        vutils.save_image(samples.cpu().data,
                              f"{self.logger.save_dir}{self.logger.name}/version_{self.logger.version}/"
                              f"{self.logger.name}_{self.current_epoch}.png",
                              normalize=True,
                              nrow=12)
    
    def configure_optimizers(self):
        optims = [] 
         
        optimizer = optim.Adam(self.model.parameters(), lr=self.params['LR'], weight_decay=self.params['weight_decay'])
        optims.append(optimizer)
        if self.params['LR_2'] is not None:
            optimizer2 = optim.Adam(getattr(self.model, self.params['submodel']).parameters(),lr=self.params['LR_2'])
            optims.append(optimizer2)
        return optims 
    
    def data_transforms(self):
        transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                       transforms.CenterCrop(148),
                                       transforms.Resize(self.params['img_size']),
                                       transforms.ToTensor(),
                                       transforms.Lambda(lambda X: 2 * X - 1)])
        return transform
    
    def train_dataloader(self):
        transform = self.data_transforms()
        dataset = CelebA(root=self.params['data_path'], split='train', transform=transform, download=True)
        self.num_train_imgs = len(dataset)
        return DataLoader(dataset, batch_size=self.params['batch_size'], shuffle=True, drop_last=True)
    
    def val_dataloader(self):
        transform = self.data_transforms()
        dataset = CelebA(root=self.params['data_path'], split='test', transform=transform, downalod=True)
        self.sample_dataloader = DataLoader(dataset, batch_size=144, shuffle=True, drop_last=True)
        self.num_val_imgs = len(self.sample_dataloader)
        return self.sample_dataloader