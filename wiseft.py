import torch
from torchvision.transforms import transforms
from argparse import Namespace

import sys
sys.path.insert(0, '../wise-ft_clipsitu')
sys.path.insert(1, '../wise-ft_clipsitu/src')
from src.models.modeling import ImageClassifier, ImageEncoder, ClassificationHead
from src.models.zeroshot import get_zeroshot_classifier
from src.datasets import ImSituVerbsDataset

class WiseFT:
    def __init__(self, wiseft_args=None):
        if wiseft_args is None:
            wiseft_args = {
                'train_dataset': 'ImSituVerbsDataset',
                'cache_dir':'cache',
                'model':'ViT-B/32',
                'batch_size':512,
                'eval_dataset':'ImSituVerbsDataset',
                'template':'openai_imagenet_template',
                'data_location':'~/data',
                'classnames':'openai',
                'device':'cuda'
            }

        self.wiseft_args = Namespace(**wiseft_args)
        self.mlp_verb_model = self._create_mlp_verb_model()

    def _create_mlp_verb_model(self):
        image_encoder = ImageEncoder(self.wiseft_args, keep_lang=True)
        delattr(image_encoder.model, 'transformer')

        zeroshot_weights = torch.load('data/processed/zeroshot_weights.pt')
        classification_head = ClassificationHead(normalize=True, weights=zeroshot_weights)

        mlp_verb_model = ImageClassifier(image_encoder, classification_head, process_images=True)
        finetuned_checkpoint = '../wise-ft_clipsitu/models/wiseft/ViTB32/finetuned/wise_ft_alpha=0.900_sd.pt'
        mlp_verb_model.load_state_dict(torch.load(finetuned_checkpoint))

        return mlp_verb_model