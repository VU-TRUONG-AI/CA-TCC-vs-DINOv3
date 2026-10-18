# CA-TCC trên Speech Commands và RAVDESS

Trong chuyên đề này, tôi kế thừa mã nguồn CA-TCC để triển khai và đánh giá phương pháp học biểu diễn trên hai bộ dữ liệu âm thanh: **Speech Commands v0.02** cho nhận diện từ khóa và **RAVDESS Speech** cho nhận diện cảm xúc giọng nói. Tôi thực hiện tổ chức dữ liệu, tiền xử lý, điều chỉnh repo cho đầu vào âm thanh, huấn luyện, đánh giá và lưu kết quả phục vụ báo cáo.

Theo định hướng của giảng viên, tôi triển khai thêm biến thể thay encoder CNN bằng **DINOv3 ViT-S/16** để khảo sát và so sánh. CA-TCC, DINOv3 và hai bộ dữ liệu là tài nguyên tôi kế thừa từ tác giả nguồn; phần đóng góp trong repo này là việc chuyển đổi, tích hợp và thực nghiệm trên bài toán âm thanh.

Tôi tổ chức dữ liệu, mã nguồn, môi trường chạy, kết quả và tài liệu trong **một thư mục `CA-TCC-main`**. Đây là thư mục gốc làm việc khi mở dự án trong VSCode hoặc thực hiện các lệnh bên dưới.

## Tài liệu và kết quả

| Nội dung | Vị trí |
|---|---|
| Cấu trúc thư mục và nhật ký sắp xếp | [docs/SAP_XEP_THU_MUC.md](docs/SAP_XEP_THU_MUC.md) |
| WAV, dữ liệu đã xử lý và mã tiền xử lý | [datasets/README.md](datasets/README.md) |
| Báo cáo và hình tiền xử lý | [datasets/reports/BAO_CAO_TIEN_XU_LY.md](datasets/reports/BAO_CAO_TIEN_XU_LY.md), `datasets/figures/` |
| Kết quả CNN 10% nhãn, bảng và hình dùng trong báo cáo | [báo cáo huấn luyện](artifacts/full_audio_10p_seed0_20261017/KET_QUA_HUAN_LUYEN.md) |
| Checkpoint và log từng giai đoạn | `experiments_logs/full_audio_10p_seed0_20261017/` |
| Thay đổi encoder DINOv3, tham số từng lượt và kết quả 10% nhãn | [README thực nghiệm DINOv3](artifacts/dinov3_search_10p_20261017/README.md) — tự cập nhật từ lượt chạy |
| Kết quả DINOv3 Speech Commands dùng tham số chọn từ RAVDESS | [Tham số RAVDESS chuyển sang SpeechCommands](artifacts/dinov3_speechcommands_ravdess_best_10p_20261018/README.md) — đã hoàn tất, ngân sách 4 epoch mỗi pha |
| Các thay đổi để chạy với dữ liệu mới | [nhật ký chuyển đổi](docs/NHAT_KY_CHUYEN_DOI_DU_LIEU.md) |
| Hướng dẫn chạy lại huấn luyện, đánh giá và suy luận | [hướng dẫn tái lập](docs/REPRODUCE_AUDIO_VI.md) |
| Bài báo, slide và tài liệu tham khảo của chuyên đề | [documents/README.md](documents/README.md) |

## Cấu trúc dự án

```text
CA-TCC-main/
├── README.md                 Tổng quan chuyên đề và liên kết tài liệu
├── datasets/                 Tất cả dữ liệu và quy trình tiền xử lý
│   ├── raw/                  WAV và gói dữ liệu gốc
│   ├── processed/            Đặc trưng NPZ: full và smoke
│   ├── tensors/              Tensor PT và nhãn giả: full và smoke
│   ├── pipeline/             Mã tiền xử lý, kiểm tra, dựng hình
│   ├── results/              Kết quả kiểm tra tiền xử lý
│   ├── reports/              Báo cáo tiền xử lý
│   └── figures/              Hình tiền xử lý đưa vào báo cáo
├── documents/                Bài báo, slide và tài liệu tham khảo
├── docs/                     Hướng dẫn, nhật ký thay đổi, tài liệu tham chiếu
├── artifacts/                Báo cáo, CSV, biểu đồ và mô hình đã xuất
├── experiments_logs/         Checkpoint, lịch sử epoch, kết quả từng pha
├── config_files/             Cấu hình mô hình và hai bộ dữ liệu
├── dataloader/               Nạp dữ liệu và augmentation
├── models/                   Encoder, temporal transformer, loss
├── trainer/                  Huấn luyện, đánh giá, sinh nhãn giả
├── variants/dinov3/           Biến thể thay encoder CNN bằng DINOv3
├── scripts/                  Điều phối chạy, export, suy luận, báo cáo
├── tests/                    Kiểm tra mã nguồn
├── web/                      App/API minh họa suy luận âm thanh
├── main.py                   Chạy một giai đoạn huấn luyện
├── .venv/                    Môi trường Python cho baseline CNN
└── .venv-dinov3/              Môi trường Python cho biến thể DINOv3
```

Tôi tập trung dữ liệu dùng cho huấn luyện trong **`datasets/`** và bỏ các liên kết cũ `Data`, `data`, `data_smoke`, `runtime_data` ở gốc để tránh nhầm lẫn. Các lệnh hiện hành dùng trực tiếp đường dẫn mới; các gói đóng gói phục vụ chia sẻ và suy luận được lưu trong `artifacts/`.

`config_files/HAR_Configs.py` vẫn cần thiết vì hai cấu hình âm thanh kế thừa tham số chung từ lớp này. Frontend âm thanh dùng chung nằm ở `datasets/pipeline/audio_features.py`.

## Quy trình thực nghiệm

Tôi dùng dữ liệu đã tiền xử lý để tạo đầu vào log-mel cho mô hình. Trong pipeline CA-TCC, self-supervised sử dụng toàn bộ tập train không nhãn; các pha linear và fine-tune sử dụng tập con có nhãn thật theo tỷ lệ của thí nghiệm. Sau bước sinh nhãn giả, pha SupCon sử dụng dữ liệu train với nhãn giả rồi tiếp tục đánh giá linear và fine-tune.

Tỷ lệ **1% hoặc 10%** được tính trên nhãn thật của tập train, không phải tỷ lệ dữ liệu được sử dụng trong mọi pha. Tôi dùng validation để lựa chọn checkpoint phân loại và giữ test cho đánh giá.

## Kết quả baseline CNN với 10% nhãn thật

Tôi đã hoàn tất tiền xử lý và kiểm chứng 107.269 WAV thuộc hai bài toán phân loại: 105.829 bản ghi từ vựng Speech Commands và 1.440 bản ghi speech của RAVDESS. Sáu WAV nhiễu nền của Speech Commands không được tính trong số mẫu phân loại này.

Tôi đã huấn luyện trên dữ liệu đầy đủ của cả hai bộ bằng GPU GTX 1650, với ngân sách 40 epoch/giai đoạn, 10% nhãn thật cho các bước phân loại ít nhãn và seed 0. Tôi sử dụng thêm supervised với 100% nhãn làm mốc tham khảo, đồng thời lưu checkpoint, metrics, mô hình TorchScript và kết quả kiểm chứng suy luận WAV.

Ở pha fine-tune sau SupCon, CA-TCC đạt Accuracy **85,48% / macro-F1 84,38% trên Speech Commands**, **23,75% / macro-F1 20,05% trên RAVDESS**. Đây là kết quả trên một seed và một cách chia dữ liệu; tôi chưa đo độ biến thiên qua nhiều lượt chạy.

- [Kết quả huấn luyện đầy đủ](artifacts/full_audio_10p_seed0_20261017/KET_QUA_HUAN_LUYEN.md).
- [Đánh giá Speech Commands](artifacts/full_audio_10p_seed0_20261017/DANH_GIA_SPEECHCOMMANDS.md).
- [Nhật ký thay đổi để repo chạy với dữ liệu mới](docs/NHAT_KY_CHUYEN_DOI_DU_LIEU.md).
- [Kết quả chạy thử và giới hạn diễn giải](artifacts/smoke_audio_20261017/KET_QUA_CHAY_THU.md).
- [Trạng thái và bằng chứng chạy](artifacts/RUN_STATUS.md).

```bash
# Chạy từ thư mục gốc repo; dùng experiment riêng cho mỗi lượt mới.
.venv/bin/python scripts/verify_audio_repo.py --experiment smoke_audio_new --device cuda:0
```

Lệnh trên kiểm chứng repo trên tập nhỏ. Có thể dùng `--device cpu` nếu không có GPU; `--resume` được dùng với cùng experiment/cấu hình khi tiếp tục lượt đã lưu. Kết quả smoke chỉ phục vụ kiểm tra pipeline, không thay thế kết quả huấn luyện đầy đủ. App/API minh họa được mô tả trong [README web](web/README.md).

Tôi lưu bản mã cũ tại `docs/reference/legacy_20261013.tar.gz`. Nhật ký sắp xếp và kiểm chứng nằm trong [SẮP XẾP THƯ MỤC](docs/SAP_XEP_THU_MUC.md).

Tôi thực nghiệm chuyển CA-TCC sang âm thanh; các kết quả này không được đồng nhất trực tiếp với benchmark HAR/EEG trong bài báo gốc.

## Lượt bổ sung 1% nhãn thật đã hoàn tất

Tôi đã hoàn tất lượt 1% trên cả hai bộ lúc 15:33 ngày 17/10/2026: đủ 40 epoch/pha, seed 0, toàn bộ test. Ở pha fine-tune sau SupCon, CA-TCC đạt Accuracy/macro-F1 **53,00%/48,07% trên Speech Commands**, **14,58%/8,53% trên RAVDESS**. Tôi đã kiểm tra checkpoint, metrics và đối chiếu suy luận WAV với tensor trên 43 mẫu. [Bảng so sánh đầy đủ 1%–10%](artifacts/full_audio_1p_seed0_20261017/SO_SANH_1P_10P.md), [CSV](artifacts/full_audio_1p_seed0_20261017/SO_SANH_1P_10P.csv), [biểu đồ](artifacts/full_audio_1p_seed0_20261017/SO_SANH_1P_10P.png). Lượt 1% có gián đoạn sau epoch 39 của pha self-supervised Speech Commands rồi được khôi phục từ checkpoint; tôi lưu nhật ký khôi phục trong artifacts của lượt này.

## Biến thể DINOv3

Theo định hướng của giảng viên, tôi tách biến thể tại `variants/dinov3/` để thay encoder CNN bằng DINOv3 ViT-S/16 và giữ đường chạy CNN làm mốc đối chiếu. Adapter chuyển log-mel thành đầu vào ba kênh, lấy patch token, tổng hợp theo trục tần số rồi chiếu thành chuỗi đặc trưng cho phần temporal/contextual contrastive learning của CA-TCC.

Tôi đã triển khai và chạy biến thể này. Các lượt tìm tham số được lưu tại [báo cáo thực nghiệm DINOv3 10% nhãn](artifacts/dinov3_search_10p_20261017/README.md). Sau đó, tôi dùng cấu hình được chọn từ RAVDESS để chạy Speech Commands với ngân sách rút gọn **4 epoch mỗi pha**; lượt này đã hoàn tất theo [báo cáo ngày 18/10/2026](artifacts/dinov3_speechcommands_ravdess_best_10p_20261018/README.md).

Khi phân tích so sánh, tôi ghi rõ DINOv3 sử dụng trọng số pretrained thị giác và batch 8, trong khi CNN dùng batch 32. Lượt Speech Commands DINOv3 có ngân sách 4 epoch mỗi pha, khác CNN 40 epoch; vì vậy chênh lệch kết quả không thể được quy hoàn toàn cho việc thay encoder.

[README DINOv3 riêng](variants/dinov3/README.md) trình bày vị trí thay đổi, adapter và cách cấu hình. Các mục kế hoạch ban đầu trong tài liệu này cần được đọc cùng các cập nhật thực nghiệm ở đầu tài liệu.

Tôi cũng đã bổ sung app/API minh họa suy luận cho hai mô hình DINOv3. Gói xuất được lưu trong `artifacts/dinov3_app_bundles/`; hướng dẫn chạy và giới hạn đầu vào nằm trong [README app/API](web/README.md).
