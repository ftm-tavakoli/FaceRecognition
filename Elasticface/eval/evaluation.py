import logging
import os
import sys
import torch

# Importing local modules
sys.path.append('/media/statlab/SeagateHDD/Fateme Tavakoli/ElasticFace')
from utils.utils_callbacks import CallBackVerification
from utils.utils_logging import init_logging
from config.config import config as cfg
from backbones.iresnet import iresnet100, iresnet50

if __name__ == "__main__":
    gpu_id = 0

    # Initialize logging
    log_root = logging.getLogger()
    init_logging(log_root, 0, cfg.output, logfile="test1.log")

    # Callback for verification during evaluation
    callback_verification = CallBackVerification(1, 0, cfg.val_targets, cfg.rec)
    
    output_folder = cfg.output
    weights = os.listdir(output_folder)
    
    # Loop through the weights files
    for w in weights:
        # Check if the file contains "backbone" in its name
        if "backbone" in w:
            # Load the backbone model based on the configuration
            if cfg.network == "iresnet100":
                backbone = iresnet100(num_features=cfg.embedding_size).to(f"cuda:{gpu_id}")
            elif cfg.network == "iresnet50":
                backbone = iresnet50(num_classes=cfg.embedding_size).to(f"cuda:{gpu_id}")
            else:
                backbone = None
                exit()  # Exit if the network type is not supported
            
            # Load the weights for the backbone
            backbone.load_state_dict(torch.load(os.path.join(output_folder, w)))
            
            # Create a data parallel model
            model = torch.nn.DataParallel(backbone, device_ids=[gpu_id])
            
            # Perform verification with the loaded model and weight index
            callback_verification(int(w.split("backbone")[0]), model)
