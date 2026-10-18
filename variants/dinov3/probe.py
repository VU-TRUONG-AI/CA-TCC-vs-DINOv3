"""Technical GPU check with random weights and synthetic inputs, never an experiment score."""
import argparse
import gc
import importlib
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
from torch.utils.data import DataLoader, TensorDataset
from models.TC import TC
from variants.dinov3.model import DinoAudioModel
from variants.dinov3.trainer import model_train
from variants.dinov3.common import ROOT, DATASETS, atomic_json, require_baseline_complete, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dinov3-repo', default='external/dinov3')
    parser.add_argument('--dinov3-weights')
    parser.add_argument('--batch-sizes', nargs='+', type=int, default=[2, 8, 32])
    parser.add_argument('--output', default='artifacts/dinov3_preflight_20261017/probe.json')
    args = parser.parse_args()
    require_baseline_complete()
    torch.set_num_threads(4)
    torch.manual_seed(0)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; run with access to the GPU.')
    initialization = 'pretrained' if args.dinov3_weights else 'random'
    report = dict(purpose='synthetic technical check; not training results', initialization=initialization,
                  weights_sha256=sha256(args.dinov3_weights) if args.dinov3_weights else None,
                  torch=torch.__version__, gpu=torch.cuda.get_device_name(0), checks=[])
    output = ROOT / args.output
    for dataset in DATASETS:
        configs = importlib.import_module(f'config_files.{dataset}_Configs').Config()
        for batch in args.batch_sizes:
            row = dict(dataset=dataset, batch_size=batch, initialization=initialization, state='running')
            report['checks'].append(row)
            atomic_json(output, report)
            start = time.monotonic()
            model = temporal = optimizer = temporal_optimizer = loader = None
            try:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                configs.batch_size = batch
                model = DinoAudioModel(configs, repo=args.dinov3_repo, initialization=initialization,
                                       weights=args.dinov3_weights).cuda()
                temporal = TC(configs, torch.device('cuda:0')).cuda()
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-4)
                temporal_optimizer = torch.optim.Adam(temporal.parameters(), lr=1e-5)
                frames = 601 if dataset == 'RAVDESS' else 101
                samples = torch.randn(batch, 40, frames)
                labels = torch.arange(batch) % configs.num_classes
                loader = DataLoader(TensorDataset(samples, labels, samples + 0.01, samples - 0.01), batch_size=batch)
                stages = {}
                for mode in ('self_supervised', 'SupCon', 'ft_10p', 'train_linear_10p'):
                    model.configure_stage(mode)
                    loss, accuracy = model_train(model, temporal, optimizer, temporal_optimizer,
                                                  torch.nn.CrossEntropyLoss(), loader, configs, torch.device('cuda:0'), mode)
                    if mode == 'train_linear_10p':
                        assert all(not p.requires_grad for p in model.encoder.parameters())
                    stages[mode] = dict(loss=loss, accuracy=accuracy)
                torch.cuda.synchronize()
                row.update(state='passed', stages=stages,
                           peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,
                           peak_reserved_mib=torch.cuda.max_memory_reserved()/1024**2)
            except torch.cuda.OutOfMemoryError as error:
                row.update(state='out_of_memory', error=str(error))
            except Exception as error:
                row.update(state='failed', error=f'{type(error).__name__}: {error}')
            finally:
                row['seconds'] = time.monotonic()-start
                print(row, flush=True)
                model = temporal = optimizer = temporal_optimizer = loader = None
                gc.collect()
                torch.cuda.empty_cache()
                atomic_json(output, report)
            if row['state'] != 'passed':
                break
    report['state'] = 'completed'
    atomic_json(output, report)


if __name__ == '__main__':
    main()
