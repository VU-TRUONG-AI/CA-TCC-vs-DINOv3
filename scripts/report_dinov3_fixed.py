"""README for a fixed-parameter DINOv3 training run."""
from datetime import datetime
import json
from pathlib import Path
from zoneinfo import ZoneInfo


def generate(root, experiment):
    output = root / 'artifacts' / experiment
    status = json.loads((output / 'fixed_run_status.json').read_text())
    settings = status['settings']
    old = root / 'artifacts/dinov3_search_10p_20261017/README.md'
    old_text = old.read_text() if old.exists() else ''
    architecture = old_text.split('## 2. Thay encoder/backbone ở đâu và như thế nào?')[1].split('## 3. Khởi tạo ban đầu')[0] if '## 2. Thay encoder/backbone ở đâu và như thế nào?' in old_text else 'Xem [thiết kế](../../variants/dinov3/README.md).'
    def num(value, percent=False):
        return '—' if value is None else (f'{100*value:.2f}%' if percent else f'{value:.6g}')
    lines = ['# DINOv3 trên SpeechCommands — dùng trực tiếp tham số thắng RAVDESS', '',
             f'Experiment: `{experiment}`. Trạng thái: **{status["state"]}**. Cập nhật: {datetime.now(ZoneInfo("Asia/Bangkok")).isoformat(timespec="seconds")}.', '',
             '## 1. Mục tiêu và thay đổi theo yêu cầu', '',
             'Người dùng yêu cầu lấy trực tiếp bộ hyperparameter tốt nhất của RAVDESS để huấn luyện SpeechCommands. '
             'Đã dừng grid SpeechCommands cũ và giữ lại config, history, checkpoint, nhãn giả, log của lượt dở dang. '
             'Không tiếp tục tìm 5 tổ hợp còn lại hoặc chạy refinement. Lượt này khởi tạo lại từ DINOv3 pretrained chính thức; '
             'chỉ chuyển hyperparameter, không chuyển weights đã học trên RAVDESS hay nối weights từ lượt SpeechCommands LR=1e-5.', '',
             '## 2. Thay encoder/backbone ở đâu và như thế nào?', architecture.strip(), '',
             '## 3. Nguồn lựa chọn và tham số khởi tạo', '',
             f'Nguồn: `{status["selection_source"]["experiment"]}`. '
             f'Validation loss của trial nguồn: **{status["selection_source"]["validation_loss"]:.6f}**. '
             'Đây là cấu hình thắng trên RAVDESS theo validation cross entropy, chưa được xác nhận tối ưu cho SpeechCommands.', '',
             '| Tham số | Giá trị |', '|---|---|',
             f'| Learning rate ban đầu | {settings["learning_rate"]:g} |',
             f'| Weight decay | {settings["weight_decay"]:g} |',
             f'| Batch / seed | {settings["batch_size"]} / {settings["seed"]} |',
             f'| Epoch mỗi pha học | {settings["epochs"]} |',
             '| Initialization | DINOv3 ViT-S/16 pretrained LVD-1689M, strict loading |',
             '| Chính sách | Trainable; linear probe đóng băng encoder và projection |',
             '| Optimizer | Adam, betas=(0.9,0.99); LR/weight decay trên áp dụng cho model và TC |',
             '| Scheduler | ReduceLROnPlateau theo validation ở pha phân loại |',
             '| Nhãn | 10% train có nhãn cho linear/fine-tune; toàn train cho contrastive |',
             '| Adapter | image height=64, clip z=4, RGB normalization cố định |',
             '| Weights đầu vào | `weights/dinov3_vits16.pth`; projection/classifier/TC khởi tạo mới dưới seed 0 |', '',
             'Cùng split/subset SpeechCommands đã dùng trước đó. Lượt cũ có LR=1e-5, weight decay=1e-4, '
             f'10 epoch mỗi pha; lượt mới đổi LR=1e-4, weight decay=3e-4 và ngân sách {settings["epochs"]} epoch mỗi pha. '
             'Không gọi đây là resume của cấu hình cũ. Ngân sách rút gọn theo yêu cầu nộp lúc 19:00 ngày 18/10/2026, '
             'để dành thời gian tích hợp web/API; không coi là lượt đối chứng cùng epoch với RAVDESS. '
             'LR có thể được scheduler giảm; không suy đoán lịch LR từng epoch.', '',
             '## 4. Kết quả từng pha', '',
             '| Pha | Trạng thái | Epoch tham chiếu | Train loss | Val loss | Val accuracy | Mốc |',
             '|---|---|---:|---:|---:|---:|---|']
    pipeline_file = root / 'experiments_logs' / experiment / 'pipeline_status.json'
    pipeline = json.loads(pipeline_file.read_text()) if pipeline_file.exists() else {}
    for stage in pipeline.get('stages', []):
        folder = pipeline_file.parent / 'SpeechCommands' / (stage['stage'] + '_seed_0')
        history_file = folder / 'history.json'
        history = json.loads(history_file.read_text()) if history_file.exists() else []
        selection_file = folder / 'selection.json'
        selected = json.loads(selection_file.read_text()) if selection_file.exists() else {}
        row = next((r for r in history if r['epoch'] == selected.get('epoch')), history[-1] if history else {})
        lines.append(f'| {stage["stage"]} | {stage["state"]} | {row.get("epoch", "—")} | '
                     f'{num(row.get("train_loss"))} | {num(row.get("validation_loss"))} | '
                     f'{num(row.get("validation_accuracy"), True)} | {"checkpoint chọn" if selected else "epoch cuối đã ghi"} |')
    lines += ['', 'Dấu — nghĩa là chưa có hoặc không áp dụng. Self-supervised/SupCon không có accuracy phân loại. '
              'Validation chọn checkpoint; test không được dùng để điều chỉnh cấu hình.', '',
              '## 5. Đánh giá test cuối', '',
              '| Accuracy | Macro-F1 | Balanced accuracy | Số mẫu |', '|---:|---:|---:|---:|']
    test = status.get('test', {})
    lines += [f'| {num(test.get("accuracy"), True)} | {num(test.get("macro_f1"), True)} | '
              f'{num(test.get("balanced_accuracy"), True)} | {test.get("samples", "—")} |', '',
              'Chỉ đánh giá test sau khi hoàn tất pipeline, trên checkpoint pha `ft_SupCon_10p` '
              'được chọn theo validation loss. Kết quả chưa có không được coi là 0.', '',
              '## 6. Tái lập và artifact', '', '```bash',
              '.venv-dinov3/bin/python -u scripts/run_dinov3_fixed.py --resume', '```', '',
              '- `fixed_run_status.json`: cấu hình cố định, nguồn lựa chọn và lịch sử attempt.',
              '- `supervisor.log`, `pipeline.log`: log tiến trình.',
              '- `experiments_logs/<experiment>/SpeechCommands/<stage>_seed_0/`: resolved config, history, selection, best/last/resume checkpoints.',
              '- Pha cuối chứa `winner_test_metrics.json`, `winner_test_predictions.csv` và hash checkpoint được đánh giá.',
              '- `source_snapshot.zip`, `source_sha256.json`, `packages.log`: provenance do pipeline lưu.',
              '- [Kết quả các lượt trước](../dinov3_search_10p_20261017/README.md) được giữ nguyên, có ghi rõ dừng theo yêu cầu.', '',
              '## 7. Lỗi và giới hạn', '', status.get('error', 'Chưa ghi nhận lỗi ở lượt mới.'), '',
              'Cấu hình được chọn trên dataset khác; không bảo đảm tốt nhất trên SpeechCommands. '
              'Một seed và pretraining thị giác ngoài dataset không đủ để kết luận thống kê. '
              'Batch 8 khác baseline CNN batch 32. Exporter TorchScript nằm ở `scripts/export_dinov3_audio.py`; '
              'tiến độ xuất và đóng gói app ở `artifacts/dinov3_app_bundles/package_status.json`.', '']
    target = output / 'README.md'
    temporary = target.with_suffix('.md.tmp')
    temporary.write_text('\n'.join(lines), encoding='utf-8')
    temporary.replace(target)
    return target
