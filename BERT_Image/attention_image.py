import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import random
from bert import BertEncoder, BertConfig
from torch.autograd import Variable
from enum import Enum
from collections import OrderedDict
import numpy as np

class BERTImage(nn.Module):
    def __init__(self, config, num_classes, output_attentions=False):
        super(BERTImage, self).__init__()
        self.output_attentions = output_attentions
        bert_config = BertConfig.from_dict(config)
        num_channels_in = config['num_channels_in']
        self.hidden_size = config['hidden_size']
        self.features_upscale = nn.Linear(num_channels_in, self.hidden_size)
        # use the BERT encoder 
        self.encoder = BertEncoder(bert_config, output_attentions=output_attentions)
        self.register_buffer('attention_mask', torch.tensor(1.0))
        self.classifier = nn.Linear(self.hidden_size, num_classes)

    def forward(self, batch_images):

        batch_features  = batch_images
        # reshape from NCHW to NHWC
        batch_features = batch_features.permute(0,2,3,1)
         
        # upscale to BERT dimension  batch * H * W * C -> batch * H * W * hidden_size
        batch_features = self.features_upscale(batch_features)
        b,w,h,_ = batch_features.shape
        all_attentions, all_representations = self.encoder(
            batch_features,
            attention_mask=self.attention_mask,
            output_all_encoded_layers = True,
        )
        representations = all_representations[0]

        # reshape representations to batch * hidden
        cls_representations = representations.view(b, -1, representations.shape[-1]).mean(1)
        cls_predictions = self.classifier(cls_representations)
        return cls_predictions

# fmt: off
config = OrderedDict(
    # === From BERT ===
    vocab_size_or_config_json_file=-1,
    hidden_size=128,  # 768,
    position_encoding_size=-1,              # dimension of the position embedding for relative attention, if -1 will default to  hidden_size
    num_hidden_layers=2,
    num_attention_heads=8,
    intermediate_size=512,
    hidden_act="gelu",
    hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1,
    max_position_embeddings=16,
    type_vocab_size=2,
    initializer_range=0.02,
    layer_norm_eps=1e-12,
    num_channels_in=3,

    # === BERT IMAGE===
    add_positional_encoding_to_input=False,
    use_learned_2d_encoding=False,
    share_position_encoding=False,           # share learned relative position encoding for all layers
    use_attention_data=False,                # use attention between pixel values instead of only positional (q.k attention)
    query_positional_score=False,            # use q.r attention (see Ramachandran, 2019)
    use_gaussian_attention=True,
    attention_isotropic_gaussian=False,
    prune_degenerated_heads=False,           # remove heads with Sigma^{-1} close to 0 or very singular (kappa > 1000) at epoch 0
    reset_degenerated_heads=False,           # reinitialize randomly the heads mentioned above
    fix_original_heads_position=False,       # original heads (not pruned/reinit) position are fixed to their original value
    fix_original_heads_weights=False,        # original heads (not pruned/reinit) value matrix are fixed to their original value
    gaussian_spread_regularizer=0.,          # penalize singular covariance gaussian attention

    gaussian_init_sigma_std=0.01,
    gaussian_init_mu_std=2.,
    attention_gaussian_blur_trick=False,     # use a computational trick for gaussian attention to avoid computing the attention probas
    pooling_concatenate_size=2,              # concatenate the pixels value by patch of pooling_concatenate_size x pooling_concatenate_size to redude dimension
    pooling_use_resnet=False,
)

bert_image_model = BERTImage(config, num_classes=10)
x = np.random.randn(16,3,28,28).astype(np.float32)
out = bert_image_model(torch.from_numpy(x))
print(out.size())

 