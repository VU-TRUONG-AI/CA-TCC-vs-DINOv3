# Huấn luyện và xuất CA-TCC trên hai bộ âm thanh

## Một dự án, các nhóm nhiệm vụ

- `datasets/`: WAV gốc, tiền xử lý, kiểm tra và báo cáo. Bắt đầu tại [README dữ liệu](../datasets/README.md).
- `CA-TCC-main`: kiến trúc, augmentation lúc train, học biểu diễn, pseudo labels, đánh giá, export và API sau này.

Speech Commands là 35-class word classification; RAVDESS là 8-class acted emotion classification. Không gộp hai không gian nhãn. Báo cáo tiền xử lý và hướng dẫn lấy hình nằm trong thư mục datasets.

## Hiểu các giai đoạn

Theo [CA-TCC](https://arxiv.org/html/2208.06616v3), pipeline chính là TS-TCC không nhãn → fine-tune ít nhãn → gán nhãn giả trên train → class-aware contrastive learning. Linear evaluation đóng băng encoder; fine-tune cập nhật encoder. Repo có thêm các giai đoạn đánh giá để so sánh.

`run_audio_pipeline.py` chạy tuần tự: self_supervised, train_linear, ft, gen_pseudo_labels, SupCon, train_linear_SupCon, ft_SupCon. Cùng experiment/run/seed được dùng tìm checkpoint đúng giai đoạn. Một bước lỗi sẽ dừng pipeline.

## Cài môi trường và chuẩn bị tensor

Máy đã kiểm tra có GTX 1650 4 GB, driver 580.178.04. GPU không hiện trong sandbox không đồng nghĩa driver máy hỏng; kiểm tra ở terminal VSCode.

```bash
cd '/home/truongvu/Documents/CĐ/CA-TCC-main'
source .venv/bin/activate
python scripts/install_deps_ipv4.py install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements-audio.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
python scripts/package_audio_data.py --source datasets/processed/smoke --output datasets/tensors/smoke
python scripts/package_audio_data.py --source datasets/processed/full --output datasets/tensors/full
```

Môi trường hiện dùng `--system-site-packages` và một `.pth` tới thư viện Python 3.12 của Codex để tái sử dụng pandas/openpyxl. Toàn dự án dùng chung `.venv` ở thư mục gốc. Máy mới nên tạo venv độc lập và cài requirements; không phụ thuộc các đường dẫn runtime của máy này.

## Chạy kiểm tra rồi chạy đầy đủ

```bash
# Bộ nhỏ: kiểm tra các giai đoạn, không dùng kết quả làm benchmark.
python scripts/run_audio_pipeline.py --data-root datasets/tensors/smoke --experiment smoke_audio --epochs 1 --label-percent 10 --device cuda:0
# Dữ liệu đầy đủ, chạy tuần tự hai bộ.
python scripts/run_audio_pipeline.py --data-root datasets/tensors/full --experiment audio_1p --epochs 40 --label-percent 1 --device cuda:0 --include-supervised
```

Dữ liệu NPZ nằm trong `datasets/processed/`; dữ liệu PT dùng huấn luyện nằm trong `datasets/tensors/`. Các liên kết dữ liệu cũ đã được bỏ. Batch mặc định 32; mức sử dụng VRAM và thời gian cần được xác nhận bằng chạy thực tế. Dùng experiment mới cho mỗi thí nghiệm để tránh ghi đè checkpoint. Hỗ trợ `--resume` từ epoch đã lưu và bỏ qua giai đoạn hoàn tất; chưa tiếp tục giữa batch. Dùng cùng cấu hình và experiment khi resume.

RAVDESS 1% chỉ có 10 nhãn thật cho 8 lớp, nên cần thí nghiệm thêm 5%/10%, nhiều seed và nhiều split người nói. Báo cáo Accuracy, macro-F1, balanced accuracy, confusion matrix; chọn cấu hình bằng val, không chọn seed tốt nhất bằng test.

## Những sửa lỗi đã thực hiện

- SupCon dùng `--label_percent` thay cho biến chưa khai báo.
- Xử lý batch nhỏ, NT-Xent dùng batch thực; kiểm tra ít nhất hai mẫu khi train với BatchNorm.
- Giữ toàn bộ mẫu val/test/pseudo-label; pseudo data lưu trên CPU.
- Channel dimension theo config, tránh suy từ chiều nhỏ nhất khi có ít mẫu.
- Linear evaluation giữ encoder ở eval, không cập nhật BatchNorm hoặc bật dropout.
- Temporal LogSoftmax có `dim=1`; permutation gốc giữ đúng các kênh và tương thích NumPy mới.
- Đánh giá được cân theo số mẫu, xuất metrics.json với macro-F1/balanced accuracy.

Adapter âm thanh dùng nhiễu nhỏ và time/frequency masking thay xáo trộn đoạn lời nói. Đây là lựa chọn khởi đầu cần ablation, không khẳng định tối ưu. Augmentation thuộc huấn luyện nên nằm trong `dataloader/`, đã chạy được trong smoke test; chưa có ablation trên dữ liệu đầy đủ.

## Export và suy luận sau khi huấn luyện thành công

```bash
python scripts/export_audio_model.py --dataset RAVDESS --checkpoint experiments_logs/audio_1p/RAVDESS/ft_SupCon_1p_seed_0/saved_models/ckp_best.pt --data-root datasets/tensors/full --output artifacts/RAVDESS
python scripts/predict_audio.py --bundle artifacts/RAVDESS --wav Data/Audio_Speech_Actors_01-24/Actor_01/03-01-02-01-01-01-01.wav
```

Bundle gồm TorchScript classifier, lớp, frontend metadata và normalization. Export kiểm tra logits trước/sau; bundle từ bộ nhỏ được đánh dấu `smoke_test_only`. Frontend suy luận được import từ thư mục datasets, không duy trì một bản sao xử lý khác. Softmax score chưa được calibration.

## Cải tiến và web/API

Sau baseline ổn định: so augmentation, số mel/sample rate/pooling, kiểm tra pseudo-label collapse và confidence threshold; giữ split/budget cố định, thay một yếu tố mỗi lần. API sẽ load model một lần, nhận WAV, dùng frontend/normalization đúng phiên bản, trả top-k. Đã có mô hình smoke và export thành công; chưa có benchmark dữ liệu đầy đủ. Web/API đã dừng theo yêu cầu để ưu tiên mô hình.

Ngày 17/10/2026 đã chạy thành công toàn bộ pipeline, export và suy luận WAV cho cả hai bộ trên tập nhỏ. Xem `artifacts/RUN_STATUS.md`, `docs/NHAT_KY_CHUYEN_DOI_DU_LIEU.md` và `artifacts/smoke_audio_20261017/` để xem bằng chứng và số liệu thực tế.
