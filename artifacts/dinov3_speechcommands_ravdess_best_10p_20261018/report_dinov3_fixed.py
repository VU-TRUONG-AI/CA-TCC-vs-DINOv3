"""Generate the fixed-run README from the checked-in artifact README.

This keeps the generated report aligned with the author-edited markdown instead of
letting a stale hard-coded template drift away from the source of truth.
"""
from pathlib import Path


def generate(root, experiment):
    """Sync the artifact README with the current markdown in this folder."""
    output = root / 'artifacts' / experiment
    output.mkdir(parents=True, exist_ok=True)

    source = Path(__file__).with_name('README.md')
    target = output / 'README.md'

    if source.exists():
        target.write_text(source.read_text(encoding='utf-8'), encoding='utf-8')
        return target

    fallback = (
        '# DINOv3 trên SpeechCommands — dùng trực tiếp tham số thắng RAVDESS\n\n'
        '## 1. Mục tiêu\n\n'
        'Chỉ dùng DINOv3 làm một phương hướng nghiên cứu không lấy DINOv3 làm backbone mặc định cho CA-TCC trên audio.\n\n'
        '## 2. Thay encoder/backbone ở đâu và như thế nào?\n\n'
        'Xem [thiết kế](../../variants/dinov3/model.py) và [pipeline](../../variants/dinov3/pipeline.py).\n'
    )
    target.write_text(fallback, encoding='utf-8')
    return target
