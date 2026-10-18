"""Generate the experiment README from saved configs, histories and search results."""
import argparse
from datetime import datetime
import fcntl
import itertools
import json
import math
import os
from pathlib import Path
import time
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text()) if path.is_file() else None


def number(value, percent=False):
    if value is None:
        return '—'
    value = float(value)
    if not math.isfinite(value):
        return str(value)
    return f'{100 * value:.2f}%' if percent else f'{value:.6g}'


def cell(value):
    return str(value).replace('|', '\\|').replace('\n', ' ')


def render(root, experiment, runner_alive=None):
    output = root / 'artifacts' / experiment
    status = read(output / 'search_status.json')
    if not status:
        raise FileNotFoundError('Missing search_status.json')
    settings = status['settings']
    configs = sorted((root / 'experiments_logs').glob(f'{experiment}_*/*/*/run_config.json'))
    first_config = read(configs[0]) if configs else {}
    resolved = first_config.get('resolved_config', {})
    state = status['state']
    updated = datetime.now(ZoneInfo('Asia/Bangkok')).isoformat(timespec='seconds')
    lines = [
        '# CA-TCC với DINOv3 ViT-S/16 — thực nghiệm 10% nhãn', '',
        f'Experiment: `{experiment}`. Trạng thái ghi nhận: **{state}**. README cập nhật: {updated}.', '',
        'README được sinh từ cấu hình và kết quả đã lưu; dấu **—** nghĩa là chưa có số liệu, không phải 0. '
        'Kết quả từng lượt đang chạy còn tạm thời. Chỉ kết luận cấu hình thắng khi có đủ kết quả lựa chọn.', '',
        '## 1. Mục tiêu và phạm vi', '',
        'Thay encoder CNN của CA-TCC bằng backbone thị giác DINOv3 ViT-S/16, '
        'chuyển log-mel thành đầu vào ảnh trong model và giữ đầu ra chuỗi cho temporal/contextual contrastive learning. '
        'Thực nghiệm trên RAVDESS và SpeechCommands với 10% nhãn thật; chọn hyperparameter bằng validation.', '',
        '## 2. Thay encoder/backbone ở đâu và như thế nào?', '',
        '| File | Thay đổi và vai trò |',
        '|---|---|',
        '| [variants/dinov3/model.py](../../variants/dinov3/model.py) | `DinoSequenceEncoder` thay 3 conv block bằng factory chính thức `dinov3.hub.backbones.dinov3_vits16`; `DinoAudioModel` ghép encoder, projection và classifier. |',
        '| [variants/dinov3/main.py](../../variants/dinov3/main.py) | Khởi tạo model biến thể; nạp checkpoint strict; truyền policy, kích thước adapter, learning rate/weight decay; cấu hình đóng băng theo pha; lưu resolved config/hash. |',
        '| [variants/dinov3/trainer.py](../../variants/dinov3/trainer.py) | Trainer riêng dựa trên baseline; linear probe đóng băng toàn encoder, chọn checkpoint bằng validation; hỗ trợ tắt đánh giá test trong search. |',
        '| [variants/dinov3/pipeline.py](../../variants/dinov3/pipeline.py) | Điều phối các pha và dữ liệu/nhãn giả riêng từng experiment. |',
        '| [variants/dinov3/search.py](../../variants/dinov3/search.py) | Thử grid tuần tự, lưu CSV/JSON, chọn winner và huấn luyện lại; đánh giá test cuối. |',
        '| [scripts/report_dinov3_search.py](../../scripts/report_dinov3_search.py) | Sinh/cập nhật README này từ dữ liệu thực tế đã ghi. |', '',
        'Đường chạy CNN gốc ở `main.py`, `models/model.py` và `trainer/trainer.py` vẫn tồn tại. '
        'Biến thể dùng chung `models/TC.py`, `models/loss.py`, dataloader, augmentation và cấu hình dataset.', '',
        '```text',
        'Log-mel đã chuẩn hóa train [B,40,T]',
        ' → clip [-4,4], ánh xạ [0,1]',
        ' → resize, lặp 3 kênh, chuẩn hóa RGB cố định',
        ' → DINOv3 ViT-S/16 → x_norm_patchtokens (không dùng CLS/register tokens)',
        ' → khôi phục lưới tần số × thời gian, trung bình theo tần số',
        ' → projection Conv1d 1×1: 384 → 128 → chuỗi [B,128,L]',
        '    ├─ temporal/contextual contrastive losses của CA-TCC',
        '    └─ flatten → Linear classifier → logits',
        '```', '',
        '| Dataset | Log-mel | Đầu vào DINOv3 | Chuỗi sau projection | Số lớp |',
        '|---|---|---|---|---:|',
        '| RAVDESS | `[B,40,601]` | `[B,3,64,1232]` | `[B,128,77]` | 8 |',
        '| SpeechCommands | `[B,40,101]` | `[B,3,64,240]` | `[B,128,15]` | 35 |', '',
        'Đây là adapter âm thanh của dự án; không phải recipe âm thanh chính thức của DINOv3. '
        'RGB mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]. Không tạo PNG hoặc colormap.', '',
        '## 3. Khởi tạo ban đầu', '',
        '| Thành phần | Giá trị |', '|---|---|',
        f'| Backbone / initialization | DINOv3 ViT-S/16 / `{settings["initialization"]}` |',
        '| Pretraining | LVD-1689M; kiến thức học từ ảnh ngoài dataset âm thanh |',
        '| Checkpoint chính thức | `dinov3_vits16_pretrain_lvd1689m-08c60483.pth` |',
        f'| File weights | `{cell(settings["dinov3_weights"])}` |',
        f'| SHA-256 | `{status.get("provenance", {}).get("weights_sha256", first_config.get("dinov3_weights_sha256", "chưa ghi"))}` |',
        '| Cách nạp | State dict chính thức, `strict=True`; thiếu/sai weights báo lỗi |',
        '| Projection và classifier | Khởi tạo mới theo mặc định PyTorch, dưới seed của lượt chạy |',
        '| Temporal model TC | Khởi tạo mới; không lấy weights CNN cũ |',
        f'| Backbone policy | `{settings["backbone_policy"]}`; khi linear probe, toàn encoder gồm projection đóng băng |',
        f'| Seed / label budget | {settings["seed"]} / {settings["label_percent"]}% train có nhãn |',
        f'| Batch / trial epochs / final epochs | {settings["batch_size"]} / {settings["trial_epochs"]} / {settings["final_epochs"]} mỗi pha học |',
        f'| Optimizer | Adam; betas=({resolved.get("beta1", "chưa ghi")}, {resolved.get("beta2", "chưa ghi")}) |',
        '| Scheduler | ReduceLROnPlateau theo validation loss ở pha phân loại; không step ở self-supervised/SupCon |',
        '| Adapter / loss temperature | image_height=64; clip_z=4; temperature=' + str(resolved.get('Context_Cont', {}).get('temperature', 'chưa ghi')) + ' |', '',
        'Cấu hình gốc dataset kế thừa lr=3e-4, weight decay baseline=3e-4, batch=32. '
        'Đợt DINOv3 này override LR/weight decay bằng từng tổ hợp dưới đây và batch=8 theo phép đo VRAM. '
        'Lượt đầu khởi tạo optimizer với lr=1e-5, weight decay=1e-4. '
        'LR trong bảng là LR khởi tạo optimizer cho cả model và TC; scheduler có thể giảm LR về sau. '
        'Không suy đoán lịch LR từng epoch khi history không ghi trường đó; trạng thái optimizer nằm trong `resume.pt`.', '',
        'Self-supervised giữ hệ số temporal/contextual 1/0.7; SupCon 0.01/0.1. '
        'Các tensor trọng số học được lưu trong checkpoint, không liệt kê hàng triệu giá trị trong README. '
        'Resolved config đầy đủ nằm ở `run_config.json` của từng pha.', '',
        '## 4. Các tham số đã thay đổi và kết quả từng lượt', '',
        'Grid: learning rate = ' + ', '.join(map(str, settings['learning_rates'])) +
        '; weight decay = ' + ', '.join(map(str, settings['weight_decays'])) + '. '
        'Mỗi tổ hợp chạy lại đầy đủ pipeline; không nối weights từ trial trước. '
        '10% nhãn dùng ở linear/fine-tune; self-supervised dùng toàn train không nhãn, SupCon dùng toàn train với pseudo labels riêng.', '',
        '| Dataset | Trial | LR ban đầu | Weight decay | Batch | Epoch/pha | Seed | Trạng thái | Val loss chọn | Val accuracy | Epoch chọn |',
        '|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|',
    ]
    trials = {r['experiment']: r for r in status['trials']}
    for dataset in settings['datasets']:
        for index, (lr, wd) in enumerate(itertools.product(settings['learning_rates'], settings['weight_decays'])):
            name = f'{experiment}_{dataset.lower()}_t{index:02d}'
            row = trials.get(name, {})
            lines.append(f'| {dataset} | t{index:02d} | {lr:g} | {wd:g} | {settings["batch_size"]} | '
                         f'{settings["trial_epochs"]} | {settings["seed"]} | {row.get("state", "chưa chạy")} | '
                         f'{number(row.get("validation_loss"))} | {number(row.get("validation_accuracy"), True)} | {row.get("selected_epoch", "—")} |')
    lines += ['', 'Các cột validation ở bảng trên lấy từ checkpoint được chọn của pha `ft_SupCon_10p`. '
              'Trial chưa kết thúc sẽ chưa có điểm xếp hạng. Chi tiết: [trials.csv](trials.csv), [search_status.json](search_status.json).', '',
              '## 5. Tiến độ và kết quả từng pha đã ghi', '',
              '| Experiment / dataset | Pha | Epoch tham chiếu | Train loss | Val loss | Val accuracy | Loại mốc |',
              '|---|---|---:|---:|---:|---:|---|']
    for config_path in configs:
        folder = config_path.parent
        history = read(folder / 'history.json') or []
        if not history:
            continue
        selection = read(folder / 'selection.json')
        row = next((r for r in history if selection and r['epoch'] == selection['epoch']), history[-1])
        run = read(config_path)
        lines.append(f'| {cell(run["experiment_description"])} / {run["selected_dataset"]} | {run["training_mode"]} | '
                     f'{row["epoch"]} | {number(row.get("train_loss"))} | {number(row.get("validation_loss"))} | '
                     f'{number(row.get("validation_accuracy"), True)} | {"checkpoint được chọn" if selection else "epoch cuối đã ghi"} |')
    lines += ['', 'Self-supervised/SupCon không có validation accuracy phân loại; giá trị placeholder 0 trong console '
              'không được coi là kết quả đánh giá. Lịch sử đầy đủ từng epoch trong `history.json`.', '',
              '## 6. Cấu hình thắng và đánh giá test cuối', '',
              'Tiêu chí: validation cross entropy nhỏ nhất tại checkpoint được chọn của pha cuối; '
              'không dùng test để chọn tham số. Winner được huấn luyện lại từ pretrained đủ '
              f'{settings["final_epochs"]} epoch mỗi pha, rồi chọn checkpoint theo validation và đánh giá test.', '',
              '| Dataset | Trial được chọn | LR / weight decay | Val loss lượt cuối | Test accuracy | Macro-F1 | Balanced accuracy |',
              '|---|---|---|---:|---:|---:|---:|']
    for dataset in settings['datasets']:
        final = status.get('finalists', {}).get(dataset, {})
        best = read(output / f'best_parameters_{dataset}.json') or {}
        test = final.get('test', {})
        params = f'{best["learning_rate"]:g} / {best["weight_decay"]:g}' if best else '—'
        lines.append(f'| {dataset} | {best.get("experiment", "chưa chọn")} | {params} | '
                     f'{number(final.get("validation_loss"))} | {number(test.get("accuracy"), True)} | '
                     f'{number(test.get("macro_f1"), True)} | {number(test.get("balanced_accuracy"), True)} |')
    lines += ['', '“Tốt nhất” chỉ trong grid, seed và ngân sách đã thử; chưa phải tối ưu toàn cục. '
              'Xếp hạng 10 epoch có thể khác 40 epoch. Một seed chưa đủ kết luận thống kê.', '',
              '## 7. Các lượt lỗi và lần thử lại', '']
    failures = []
    for trial in status['trials'] + list(status.get('finalists', {}).values()):
        for attempt in trial.get('previous_attempts', []) + [trial]:
            if attempt.get('error') or attempt.get('state') == 'failed':
                failures.append(f'- `{trial["experiment"]}`: {cell(attempt.get("error", "failed"))}; exit code={attempt.get("exit_code", "—")}.')
    lines.extend(failures or ['Chưa ghi nhận trial lỗi.'])
    if status.get('error'):
        lines += ['', 'Thông báo search: ' + cell(status['error'])]
    if runner_alive is False and state not in ('completed', 'failed', 'interrupted', 'awaiting_weights'):
        lines += ['', '**Tiến trình search đã dừng nhưng chưa ghi trạng thái kết thúc. Báo cáo này chưa hoàn tất.**']
    lines += ['', '## 8. Lưu trữ và tái lập', '',
              '- `experiments_logs/<trial>/<dataset>/<stage>_seed_0/`: config, history, selection, best/last/resume checkpoint.',
              '- `artifacts/<trial>/`: source snapshot/hash, package versions và pipeline log.',
              '- `datasets/tensors/experiments/<trial>/`: liên kết dữ liệu, manifest/normalization và pseudo labels riêng.',
              '- `best_parameters_<dataset>.json`: cấu hình thắng; `winner_test_metrics.json`/`winner_test_predictions.csv` nằm trong pha cuối của finalist.',
              '- Trọng số model và hash nguồn/checkpoint trong config/snapshot cho phép truy vết từng lượt.', '',
              'Chỉ resume khi không có search đang hoạt động:', '', '```bash',
              '.venv-dinov3/bin/python -u variants/dinov3/search.py --resume', '```', '',
              'Sinh lại README từ kết quả đã lưu:', '', '```bash',
              f'.venv-dinov3/bin/python scripts/report_dinov3_search.py --experiment {experiment}', '```', '',
              '## 9. Điều kiện diễn giải', '',
              'DINOv3 có pretraining ngoài dataset; adapter resize/clipping và attention toàn cục cũng thay đổi. '
              'Batch 8 khác CNN baseline batch 32, ảnh hưởng số negatives và số bước optimizer. '
              'Không quy toàn bộ chênh lệch cho encoder hoặc coi hai lượt chỉ khác backbone. '
              'Chưa hỗ trợ export TorchScript DINOv3 trong pipeline này.', '',
              'Nguồn kiến trúc: [DINOv3 chính thức](https://github.com/facebookresearch/dinov3). '
              'Thiết kế chi tiết: [README biến thể](../../variants/dinov3/README.md).', '']
    return '\n'.join(lines)


def generate(root, experiment, runner_alive=None):
    text = render(root, experiment, runner_alive)
    target = root / 'artifacts' / experiment / 'README.md'
    temporary = target.with_suffix('.md.tmp')
    temporary.write_text(text, encoding='utf-8')
    temporary.replace(target)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', default='dinov3_search_10p_20261017')
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--interval', type=float, default=30)
    parser.add_argument('--runner-pid', type=int)
    args = parser.parse_args()
    if Path(args.experiment).name != args.experiment or not args.experiment.startswith('dinov3_') or args.interval < 1:
        parser.error('Invalid experiment or interval.')
    lock_path = ROOT / 'artifacts' / args.experiment / 'readme_report.lock'
    with lock_path.open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            alive = None
            if args.runner_pid:
                try:
                    os.kill(args.runner_pid, 0)
                    alive = True
                except ProcessLookupError:
                    alive = False
            try:
                target = generate(ROOT, args.experiment, alive)
                state = read(target.parent / 'search_status.json')['state']
                if not args.watch or state in ('completed', 'failed', 'interrupted', 'awaiting_weights') or alive is False:
                    print(f'README saved: {target}; state={state}; runner_alive={alive}', flush=True)
                    return
            except json.JSONDecodeError:
                if not args.watch:
                    raise
                print('An epoch file is being updated; retrying next interval.', flush=True)
            time.sleep(args.interval)


if __name__ == '__main__':
    main()
