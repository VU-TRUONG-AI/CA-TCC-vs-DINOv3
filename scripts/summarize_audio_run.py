"""Create report-ready tables and figures exclusively from completed run artifacts."""
import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--experiment',required=True)
args=p.parse_args()
folder=ROOT/'artifacts'/args.experiment
status=json.loads((folder/'verification_status.json').read_text())
if status['state']!='completed':
    raise SystemExit('Verification has not completed; no final report generated.')
rows=json.loads((folder/'metrics_summary.json').read_text())
figures=folder/'figures';figures.mkdir(exist_ok=True)
lines=['# Kết quả chạy thử CA-TCC trên hai bộ âm thanh', '',
       'Ngày chạy: 17/10/2026. Đây là kiểm chứng repo trên tập nhỏ, 1 epoch mỗi giai đoạn, seed 0, batch tối đa 32; không phải benchmark trên dữ liệu đầy đủ.', '',
       'Các giai đoạn phân loại ít nhãn dùng 10% nhãn thật của tập train nhỏ. Supervised dùng 100% nhãn thật của cùng tập train nhỏ. Chọn checkpoint bằng validation loss, sau đó đánh giá test.', '',
       '| Bộ dữ liệu | Giai đoạn | Accuracy (%) | Macro-F1 (%) | Balanced accuracy (%) | Số mẫu test |',
       '|---|---|---:|---:|---:|---:|']
for row in rows:
    lines.append(f"| {row['dataset']} | {row['stage']} | {100*row['accuracy']:.2f} | {100*row['macro_f1']:.2f} | {100*row['balanced_accuracy']:.2f} | {row['test_count']} |")
lines += ['', '## Kiểm chứng kỹ thuật', '',
          '- Toàn bộ pipeline chạy riêng cho Speech Commands và RAVDESS.',
          '- Xuất TorchScript và đối chiếu logits trước/sau export đạt.',
          '- Nạp lại trạng thái resume của fine-tune cuối đạt; không chạy thêm epoch.',
          '- Đối chiếu một WAV test mỗi lớp với đặc trưng đã lưu và logits của bundle đạt.', '',
          '## Giới hạn diễn giải', '',
          'Số epoch và tập huấn luyện nhỏ chỉ nhằm phát hiện lỗi tích hợp. Không dùng các chỉ số này để kết luận CA-TCC vượt supervised hoặc phản ánh chất lượng triển khai thực tế. Việc nạp resume được kiểm tra; chưa có thí nghiệm đối chiếu hai lượt nhiều epoch để khẳng định tái lập bitwise sau gián đoạn.', '',
          '## Hình và dữ liệu nguồn', '']
for dataset in ('SpeechCommands','RAVDESS'):
    source=ROOT/'experiments_logs'/args.experiment/dataset/'ft_SupCon_10p_seed_0'
    true=np.load(source/'labels/true_labels.npy').astype(int)
    pred=np.load(source/'labels/predicted_labels.npy').astype(int)
    meta=json.loads((ROOT/'datasets/tensors/smoke'/dataset/'manifest.json').read_text())
    classes=meta['classes']
    cm=np.zeros((len(classes),len(classes)),dtype=int)
    np.add.at(cm,(true,pred),1)
    fig,ax=plt.subplots(figsize=(13,11) if len(classes)>10 else (8,7))
    image=ax.imshow(cm,cmap='Greens',vmin=0)
    ax.set(xticks=np.arange(len(classes)),yticks=np.arange(len(classes)),xticklabels=classes,yticklabels=classes,
           xlabel='Predicted label',ylabel='True label',title=f'{dataset} · CA-TCC smoke test · 1 epoch/stage')
    plt.setp(ax.get_xticklabels(),rotation=90 if len(classes)>10 else 45,ha='right',fontsize=9)
    plt.setp(ax.get_yticklabels(),fontsize=9)
    fig.colorbar(image,ax=ax,label='Number of test samples')
    if len(classes)<=8:
        for i in range(len(classes)):
            for j in range(len(classes)):ax.text(j,i,str(cm[i,j]),ha='center',va='center',color='white' if cm[i,j]>cm.max()/2 else 'black')
    fig.tight_layout()
    fig.savefig(figures/f'{dataset}_confusion_matrix.png',dpi=180)
    plt.close(fig)
    np.savetxt(figures/f'{dataset}_confusion_matrix.csv',cm,fmt='%d',delimiter=',',header=','.join(classes),comments='')
    lines.append(f'- `figures/{dataset}_confusion_matrix.png`: confusion matrix của fine-tune CA-TCC cuối; hàng là nhãn thật, cột là dự đoán.')
lines += ['', 'Bảng nguồn: `metrics_summary.csv`; kết quả đối chiếu WAV: `wav_parity.json`; cấu hình và lịch sử: `experiments_logs/'+args.experiment+'/`; phiên bản thư viện: `packages.log`.', '']
(folder/'KET_QUA_CHAY_THU.md').write_text('\n'.join(lines),encoding='utf-8')
print(folder/'KET_QUA_CHAY_THU.md')
