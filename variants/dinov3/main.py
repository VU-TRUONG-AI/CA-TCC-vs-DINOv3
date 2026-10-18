"""Separate CA-TCC entry point using DINOv3; the baseline main.py is unchanged."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from variants.dinov3.common import (ROOT, require_baseline_complete, validate_experiment,
                                    experiment_data_root, sha256, snapshot_sources)
import argparse
import os
import sys
import random
import json
import importlib
import hashlib
import math
from datetime import datetime

import numpy as np
import torch

from dataloader.dataloader import data_generator
from models.TC import TC
from variants.dinov3.model import DinoAudioModel
from variants.dinov3.trainer import Trainer, model_evaluate, gen_pseudo_labels
from utils import _calc_metrics
from utils import _logger

start_time = datetime.now()

parser = argparse.ArgumentParser()

######################## Model parameters ########################
home_dir = os.getcwd()
parser.add_argument('--experiment_description',     required=True,  type=str,   help='Experiment Description')
parser.add_argument('--run_description',            default='test1',            type=str,   help='Experiment Description')
parser.add_argument('--seed',                       default=0,                  type=int,   help='seed value')
parser.add_argument('--training_mode',              default='self_supervised',  type=str,
                    help='Modes of choice: random_init, supervised, self_supervised, SupCon, ft_1p, gen_pseudo_labels')

parser.add_argument('--selected_dataset',           default='SpeechCommands',              type=str,   choices=['SpeechCommands', 'RAVDESS'], help='Dataset configuration')
parser.add_argument('--data_path',                  default='datasets/tensors/full',           type=str,   help='Path containing dataset')

parser.add_argument('--logs_save_dir',              default='experiments_logs', type=str,   help='saving directory')
parser.add_argument('--device',                     default='auto',           type=str,   help='cpu or cuda')
parser.add_argument('--home_path',                  default=home_dir,           type=str,   help='Project home directory')
parser.add_argument('--epochs', type=int, default=None)
parser.add_argument('--batch_size', type=int, default=None)
parser.add_argument('--label_percent', type=int, default=1, choices=[1, 5, 10, 50, 75])
parser.add_argument('--threads', type=int, default=4)
parser.add_argument('--resume', action='store_true')
parser.add_argument('--dinov3-repo', required=True, help='Local official DINOv3 checkout; never cloned automatically.')
parser.add_argument('--dinov3-weights', default=None, help='Local official ViT-S/16 checkpoint, required for pretrained initialization.')
parser.add_argument('--initialization', choices=['pretrained', 'random'], default='pretrained')
parser.add_argument('--backbone-policy', choices=['trainable', 'frozen'], default='trainable')
parser.add_argument('--image-height', type=int, default=64)
parser.add_argument('--clip-z', type=float, default=4.0)
parser.add_argument('--learning-rate', type=float, default=None)
parser.add_argument('--weight-decay', type=float, default=3e-4)
parser.add_argument('--validation-only', action='store_true', help='Do not evaluate test during parameter search.')
args = parser.parse_args()
require_baseline_complete()
validate_experiment(args.experiment_description)
if args.run_description != args.selected_dataset:
    parser.error('--run_description must match --selected_dataset for isolated reports.')
if Path(args.logs_save_dir).resolve() != ROOT / 'experiments_logs':
    parser.error('Keep --logs_save_dir=experiments_logs so each dinov3_ experiment has a separate run tree.')
if Path(args.data_path).resolve() != experiment_data_root(args.experiment_description):
    parser.error('Use this experiment private data root; prepare it with variants/dinov3/pipeline.py.')
valid_modes = {'self_supervised', 'SupCon', 'gen_pseudo_labels', 'supervised',
               f'train_linear_{args.label_percent}p', f'ft_{args.label_percent}p',
               f'train_linear_SupCon_{args.label_percent}p', f'ft_SupCon_{args.label_percent}p'}
if args.training_mode not in valid_modes:
    parser.error('Mode does not match the selected label budget.')
repo = Path(args.dinov3_repo).expanduser().resolve()
if not (repo / 'hubconf.py').is_file():
    parser.error('--dinov3-repo must contain the official hubconf.py.')
if args.initialization == 'pretrained' and (not args.dinov3_weights or not Path(args.dinov3_weights).expanduser().is_file()):
    parser.error('Pretrained initialization requires a local --dinov3-weights file.')
if args.initialization == 'random' and args.dinov3_weights:
    parser.error('Random initialization must not receive pretrained weights.')
provenance = {'dinov3_weights_sha256': sha256(Path(args.dinov3_weights).expanduser()) if args.dinov3_weights else None,
              'dinov3_repo_source_sha256': {str(p.relative_to(repo)): sha256(p) for p in sorted((repo/'dinov3').rglob('*.py'))}}

if args.device == 'auto':
    args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
if args.device.startswith('cuda') and not torch.cuda.is_available():
    parser.error('CUDA unavailable in this process. Check driver/access or pass --device cpu.')
torch.set_num_threads(args.threads)


device = torch.device(args.device)
experiment_description = args.experiment_description
data_type = args.selected_dataset
training_mode = args.training_mode
run_description = args.run_description

logs_save_dir = args.logs_save_dir
os.makedirs(logs_save_dir, exist_ok=True)

Configs = importlib.import_module(f'config_files.{data_type}_Configs').Config
configs = Configs()
if args.epochs is not None:
    configs.num_epoch = args.epochs
if args.batch_size is not None:
    configs.batch_size = args.batch_size
if args.learning_rate is not None:
    configs.lr = args.learning_rate
if not math.isfinite(configs.lr) or configs.lr <= 0 or not math.isfinite(args.weight_decay) or args.weight_decay < 0:
    parser.error('learning-rate must be finite and positive; weight-decay finite and nonnegative.')
if configs.num_epoch < 1 or configs.batch_size < 2:
    parser.error('epochs must be >=1 and batch_size >=2')


# ##### fix random seeds for reproducibility ########
SEED = args.seed
random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)
#####################################################


experiment_log_dir = os.path.join(logs_save_dir, experiment_description, run_description,
                                  training_mode + f"_seed_{SEED}")
os.makedirs(experiment_log_dir, exist_ok=True)
config_path = os.path.join(experiment_log_dir, 'run_config.json')
if os.path.isfile(config_path):
    previous = json.load(open(config_path))
    if not args.resume:
        parser.error(f'Run already exists: {experiment_log_dir}. Use a new experiment or --resume.')
    for key in ('selected_dataset', 'data_path', 'training_mode', 'seed', 'label_percent', 'epochs', 'batch_size',
                'dinov3_repo', 'dinov3_weights', 'initialization', 'backbone_policy', 'image_height', 'clip_z',
                'learning_rate', 'weight_decay', 'validation_only'):
        if previous.get(key) != vars(args).get(key):
            parser.error(f'Cannot resume with changed {key}.')
    for key, value in provenance.items():
        if previous.get(key) != value:
            parser.error(f'Cannot resume with changed source/weights: {key}.')

# loop through domains
counter = 0
src_counter = 0

# Logging
log_file_name = os.path.join(experiment_log_dir, f"logs_{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.log")
logger = _logger(log_file_name)
logger.debug("=" * 45)
logger.debug(f'Dataset: {data_type}')
logger.debug(f'Mode:    {training_mode}')
logger.debug("=" * 45)

# Load datasets
data_path = os.path.join(args.data_path, data_type)
train_dl, valid_dl, test_dl = data_generator(data_path, configs, training_mode)
logger.debug(f'Data loaded. Device: {device}; train/val/test: '
             f'{len(train_dl.dataset)}/{len(valid_dl.dataset)}/{len(test_dl.dataset)}')
def config_values(value):
    if hasattr(value, '__dict__'):
        return {k: config_values(v) for k, v in vars(value).items()}
    return value

manifest_path = os.path.join(data_path, 'manifest.json')
metadata = {**vars(args), **provenance, 'backbone': 'dinov3_vits16', 'torch_version': str(torch.__version__),
            'resolved_config': config_values(configs),
            'manifest_sha256': hashlib.sha256(open(manifest_path, 'rb').read()).hexdigest() if os.path.exists(manifest_path) else None,
            'num_epoch': configs.num_epoch, 'batch_size': configs.batch_size}
with open(os.path.join(experiment_log_dir, 'run_config.json'), 'w') as f:
    json.dump(metadata, f, indent=2)


# Load Model
model = DinoAudioModel(configs, repo=args.dinov3_repo, weights=args.dinov3_weights,
                       initialization=args.initialization, image_height=args.image_height,
                       clip_z=args.clip_z, backbone_policy=args.backbone_policy).to(device)
temporal_contr_model = TC(configs, device).to(device)

if "fine_tune" in training_mode or "ft_" in training_mode:
    # load saved model of this experiment
    if 'SupCon' not in training_mode:
        load_from = os.path.join(
            os.path.join(logs_save_dir, experiment_description, run_description, f"self_supervised_seed_{SEED}",
                         "saved_models"))
    else:
        load_from = os.path.join(
            os.path.join(logs_save_dir, experiment_description, run_description, f"SupCon_seed_{SEED}", "saved_models"))
    selected_path = os.path.join(load_from, "ckp_best.pt")
    if not os.path.isfile(selected_path):
        selected_path = os.path.join(load_from, "ckp_last.pt")
    chkpoint = torch.load(selected_path, map_location=device, weights_only=True)
    pretrained_dict = chkpoint["model_state_dict"]
    model_dict = model.state_dict()
    del_list = ['logits']
    pretrained_dict_copy = pretrained_dict.copy()
    for i in pretrained_dict_copy.keys():
        for j in del_list:
            if j in i:
                del pretrained_dict[i]
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

if training_mode == "gen_pseudo_labels":
    ft_perc = f"{args.label_percent}p"
    load_from = os.path.join(
        os.path.join(logs_save_dir, experiment_description, run_description, f"ft_{ft_perc}_seed_{SEED}", "saved_models"))
    selected_path = os.path.join(load_from, "ckp_best.pt")
    if not os.path.isfile(selected_path):
        selected_path = os.path.join(load_from, "ckp_last.pt")
    chkpoint = torch.load(selected_path, map_location=device, weights_only=True)
    pretrained_dict = chkpoint["model_state_dict"]
    model.load_state_dict(pretrained_dict)
    gen_pseudo_labels(model, train_dl, device, data_path)
    sys.exit(0)

if "train_linear" in training_mode or "tl" in training_mode:
    if 'SupCon' not in training_mode:
        load_from = os.path.join(
            os.path.join(logs_save_dir, experiment_description, run_description, f"self_supervised_seed_{SEED}",
                         "saved_models"))
    else:
        load_from = os.path.join(
            os.path.join(logs_save_dir, experiment_description, run_description, f"SupCon_seed_{SEED}", "saved_models"))
    selected_path = os.path.join(load_from, "ckp_best.pt")
    if not os.path.isfile(selected_path):
        selected_path = os.path.join(load_from, "ckp_last.pt")
    chkpoint = torch.load(selected_path, map_location=device, weights_only=True)
    pretrained_dict = chkpoint["model_state_dict"]
    model_dict = model.state_dict()

    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

    # delete these parameters (Ex: the linear layer at the end)
    del_list = ['logits']
    pretrained_dict_copy = pretrained_dict.copy()
    for i in pretrained_dict_copy.keys():
        for j in del_list:
            if j in i:
                del pretrained_dict[i]

    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    # configure_stage below freezes all encoder parameters for linear evaluation.

if training_mode == "SupCon":
    load_from = os.path.join(       
        os.path.join(logs_save_dir, experiment_description, run_description, f"ft_{args.label_percent}p_seed_{SEED}", "saved_models"))      
    selected_path = os.path.join(load_from, "ckp_best.pt")
    if not os.path.isfile(selected_path):
        selected_path = os.path.join(load_from, "ckp_last.pt")
    chkpoint = torch.load(selected_path, map_location=device, weights_only=True)      
    pretrained_dict = chkpoint["model_state_dict"]      
    model.load_state_dict(pretrained_dict)     
    
    
model.configure_stage(training_mode)

model_optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=configs.lr, betas=(configs.beta1, configs.beta2),
                                   weight_decay=args.weight_decay)

temporal_contr_optimizer = torch.optim.Adam(temporal_contr_model.parameters(), lr=configs.lr,
                                            betas=(configs.beta1, configs.beta2), weight_decay=args.weight_decay)

if training_mode == "self_supervised" or training_mode == "SupCon":  # to do it only once
    snapshot_sources(experiment_log_dir)

# Trainer
Trainer(model, temporal_contr_model, model_optimizer, temporal_contr_optimizer, train_dl, valid_dl, test_dl, device,
        logger, configs, experiment_log_dir, training_mode, resume=args.resume, evaluate_test=not args.validation_only)

if not args.validation_only and training_mode not in ("self_supervised", "SupCon", "SupCon_pseudo"):
    # Testing
    outs = model_evaluate(model, temporal_contr_model, test_dl, device, training_mode)
    total_loss, total_acc, pred_labels, true_labels = outs
    _calc_metrics(pred_labels, true_labels, experiment_log_dir, args.home_path)

logger.debug(f"Training time is : {datetime.now() - start_time}")
