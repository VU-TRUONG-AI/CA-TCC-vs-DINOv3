"""Compare completed CNN and DINOv3 runs only; never launches training."""
import argparse
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from variants.dinov3.common import ROOT, atomic_json, require_baseline_complete, validate_experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-experiment', required=True)
    parser.add_argument('--variant-experiment', required=True)
    args = parser.parse_args()
    require_baseline_complete()
    validate_experiment(args.variant_experiment)
    if Path(args.baseline_experiment).name != args.baseline_experiment or args.baseline_experiment == args.variant_experiment:
        parser.error('Choose a different, single-folder CNN baseline experiment.')
    runs = {}
    for kind, experiment in [('baseline', args.baseline_experiment), ('variant', args.variant_experiment)]:
        log = ROOT / 'experiments_logs' / experiment
        pipeline = json.loads((log / 'pipeline_status.json').read_text())
        if pipeline['state'] != 'completed':
            raise RuntimeError(f'Run is incomplete: {experiment}')
        metrics = json.loads((ROOT / 'artifacts' / experiment / 'metrics_summary.json').read_text())
        runs[kind] = {'experiment': experiment, 'settings': pipeline['settings'],
                      'metrics': {(r['dataset'], r['stage']): r for r in metrics}}
    baseline, variant = runs['baseline'], runs['variant']
    for key in ('label_percent', 'seed', 'epochs'):
        if baseline['settings'][key] != variant['settings'][key]:
            raise ValueError(f'Use matching {key} for a paired backbone comparison.')
    for kind, run in runs.items():
        if kind == 'variant' and run['settings'].get('backbone') != 'dinov3_vits16':
            raise ValueError('Variant does not identify DINOv3 ViT-S/16.')
        if kind == 'baseline' and run['settings'].get('backbone', 'cnn') != 'cnn':
            raise ValueError('Reference run must be the CNN baseline.')
    percent = variant['settings']['label_percent']
    stages = [f'train_linear_{percent}p', f'ft_{percent}p',
              f'train_linear_SupCon_{percent}p', f'ft_SupCon_{percent}p']
    datasets = sorted(set(baseline['settings']['datasets']) & set(variant['settings']['datasets']))
    if not datasets:
        raise ValueError('No common datasets.')
    rows = []
    for dataset in datasets:
        for stage in stages:
            configs = []
            for run in (baseline, variant):
                path = ROOT / 'experiments_logs' / run['experiment'] / dataset
                configs.append(json.loads((path / f'{stage}_seed_{variant["settings"]["seed"]}' / 'run_config.json').read_text()))
            # Manifest copies are byte-identical; normalization is derived from these training splits.
            if not configs[0].get('manifest_sha256') or configs[0]['manifest_sha256'] != configs[1].get('manifest_sha256'):
                raise ValueError('Manifests differ. Review split/path relocation before comparing; do not silently equate runs.')
            a, b = baseline['metrics'][(dataset, stage)], variant['metrics'][(dataset, stage)]
            if a['test_count'] != b['test_count']:
                raise ValueError('Test counts differ.')
            row = {'dataset': dataset, 'stage': stage, 'label_percent': percent,
                   'seed': variant['settings']['seed'], 'test_count': a['test_count']}
            for metric in ('accuracy', 'macro_f1', 'balanced_accuracy'):
                row['cnn_' + metric] = a[metric]
                row['dinov3_' + metric] = b[metric]
                row['delta_' + metric + '_pp'] = 100 * (b[metric] - a[metric])
            rows.append(row)
    output = ROOT / 'artifacts' / args.variant_experiment / 'comparison'
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / 'metrics.json', {'runs': {k: v['experiment'] for k, v in runs.items()}, 'rows': rows})
    with (output / 'metrics.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ['# CNN và DINOv3: so sánh kết quả đã hoàn thành', '',
             f'CNN: `{args.baseline_experiment}`; DINOv3: `{args.variant_experiment}`.', '',
             '| Bộ dữ liệu | Pha | CNN Acc (%) | DINOv3 Acc (%) | Chênh lệch (điểm %) | CNN macro-F1 (%) | DINOv3 macro-F1 (%) |',
             '|---|---|---:|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f'| {row["dataset"]} | {row["stage"]} | {100*row["cnn_accuracy"]:.2f} | '
                     f'{100*row["dinov3_accuracy"]:.2f} | {row["delta_accuracy_pp"]:+.2f} | '
                     f'{100*row["cnn_macro_f1"]:.2f} | {100*row["dinov3_macro_f1"]:.2f} |')
    lines += ['', '## Điều kiện diễn giải', '',
              f'- DINOv3 initialization: `{variant["settings"]["initialization"]}`; backbone policy: `{variant["settings"]["backbone_policy"]}`.',
              f'- Batch CNN/DINOv3: {baseline["settings"]["batch_size"]}/{variant["settings"]["batch_size"]}. Batch khác nhau ảnh hưởng contrastive negatives.',
              '- Nếu dùng pretrained DINOv3, có kiến thức từ dữ liệu ảnh ngoài; không quy toàn bộ khác biệt cho kiến trúc.',
              '- Resize/clipping, số tham số và self-attention toàn cục cũng thay đổi; không phải tái lập benchmark tác giả.',
              '- Một seed/split chưa đủ kết luận thống kê. Chỉ chọn checkpoint bằng validation; test dùng đánh giá cuối.']
    (output / 'SO_SANH_BACKBONE.md').write_text('\n'.join(lines))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig, axes = plt.subplots(len(datasets), 2, figsize=(13, 4 * len(datasets)), squeeze=False)
    for i, dataset in enumerate(datasets):
        selected = [r for r in rows if r['dataset'] == dataset]
        x = np.arange(len(selected))
        for j, metric in enumerate(('accuracy', 'macro_f1')):
            ax = axes[i, j]
            ax.bar(x - .2, [100*r['cnn_' + metric] for r in selected], .4, label='CNN')
            ax.bar(x + .2, [100*r['dinov3_' + metric] for r in selected], .4, label='DINOv3')
            ax.set(xticks=x, xticklabels=['TS linear', 'TS fine-tune', 'CA linear', 'CA fine-tune'],
                   ylim=(0, 100), ylabel='Percent', title=f'{dataset}: {metric}')
            ax.tick_params(axis='x', rotation=20)
            ax.legend()
    fig.tight_layout()
    fig.savefig(output / 'backbone_comparison.png', dpi=180)
    plt.close(fig)
    print(f'Comparison saved: {output}')


if __name__ == '__main__':
    main()
