"""Export tables, per-file predictions and figures from completed full-data stages."""
import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def build(experiment, datasets):
    output = ROOT / 'artifacts' / experiment
    output.mkdir(parents=True, exist_ok=True)
    pipeline = json.loads((ROOT/'experiments_logs'/experiment/'pipeline_status.json').read_text())
    settings = pipeline['settings']
    data_root = Path(settings['data_root'])
    if not data_root.is_absolute():
        data_root = ROOT/data_root
    # Historical runs recorded the old alias before data consolidation.
    legacy = ROOT/'runtime_data'
    if data_root.is_relative_to(legacy):
        data_root = ROOT/'datasets/tensors'/data_root.relative_to(legacy)
    rows = []
    for dataset in datasets:
        manifest = json.loads((data_root/dataset/'manifest.json').read_text())
        if manifest['smoke_subset']:
            raise ValueError('Refusing to label smoke data as full-data evaluation.')
        classes = manifest['classes']
        folder = output/dataset
        folder.mkdir(exist_ok=True)
        completed = {s['stage'] for s in pipeline['stages'] if s['dataset']==dataset and s['state']=='completed'}
        for stage in sorted(completed):
            run = ROOT/'experiments_logs'/experiment/dataset/f'{stage}_seed_{settings["seed"]}'
            metric_path = run/'metrics.json'
            if not metric_path.exists():
                continue
            metrics = json.loads(metric_path.read_text())
            selected = json.loads((run/'selection.json').read_text())
            history = json.loads((run/'history.json').read_text())
            true = np.load(run/'labels/true_labels.npy').astype(int)
            pred = np.load(run/'labels/predicted_labels.npy').astype(int)
            if not len(true)==len(pred)==manifest['splits']['test']['count']:
                raise ValueError(f'Incomplete test predictions: {dataset}/{stage}')
            rows.append(dict(dataset=dataset,stage=stage,epochs=len(history),selected_epoch=selected['epoch'],
                             accuracy=metrics['accuracy'],macro_f1=metrics['macro_f1'],
                             balanced_accuracy=metrics['balanced_accuracy'],test_count=len(true),
                             label_percent=100 if stage=='supervised' else settings['label_percent'],seed=settings['seed']))
            with (folder/f'{stage}_predictions.csv').open('w',newline='',encoding='utf-8-sig') as stream:
                writer=csv.writer(stream)
                writer.writerow(['file','true_label','predicted_label','correct'])
                for name,t,p in zip(manifest['splits']['test']['paths'],true,pred):
                    writer.writerow([name,classes[t],classes[p],bool(t==p)])
            if stage != f'ft_SupCon_{settings["label_percent"]}p':
                continue
            cm=np.zeros((len(classes),len(classes)),dtype=np.int64)
            np.add.at(cm,(true,pred),1)
            with (folder/'confusion_matrix.csv').open('w',newline='') as stream:
                writer=csv.writer(stream);writer.writerow(['true / predicted',*classes])
                writer.writerows([[c,*row] for c,row in zip(classes,cm)])
            fig,ax=plt.subplots(figsize=(13,11) if len(classes)>10 else (8,7))
            image=ax.imshow(cm,cmap='Blues',vmin=0)
            ax.set(xticks=np.arange(len(classes)),yticks=np.arange(len(classes)),xticklabels=classes,yticklabels=classes,
                   xlabel='Predicted label',ylabel='True label',title=f'{dataset} · CA-TCC · Full test set')
            plt.setp(ax.get_xticklabels(),rotation=90 if len(classes)>10 else 45,ha='right',fontsize=9)
            plt.setp(ax.get_yticklabels(),fontsize=9)
            fig.colorbar(image,ax=ax,label='Test samples')
            if len(classes)<=8:
                for i in range(len(classes)):
                    for j in range(len(classes)):
                        ax.text(j,i,str(cm[i,j]),ha='center',va='center',color='white' if cm[i,j]>cm.max()/2 else 'black')
            fig.tight_layout();fig.savefig(folder/'confusion_matrix.png',dpi=180);plt.close(fig)
            epochs=[h['epoch'] for h in history]
            fig,axes=plt.subplots(1,2,figsize=(11,4))
            axes[0].plot(epochs,[h['train_loss'] for h in history],label='Train')
            axes[0].plot(epochs,[h['validation_loss'] for h in history],label='Validation')
            axes[0].set(xlabel='Epoch',ylabel='Cross entropy',title='Classification loss')
            axes[1].plot(epochs,[h['train_accuracy'] for h in history],label='Train')
            axes[1].plot(epochs,[h['validation_accuracy'] for h in history],label='Validation')
            axes[1].set(xlabel='Epoch',ylabel='Accuracy',title='Classification accuracy',ylim=(0,1))
            for ax in axes:ax.legend();ax.grid(alpha=.2)
            fig.suptitle(f'{dataset} · Final CA-TCC fine-tuning')
            fig.tight_layout();fig.savefig(folder/'training_curves.png',dpi=180);plt.close(fig)
    # Keep previously reported datasets when the next one finishes.
    previous_path=output/'metrics_summary.json'
    previous=json.loads(previous_path.read_text()) if previous_path.exists() else []
    rows=[r for r in previous if r['dataset'] not in datasets]+rows
    (output/'metrics_summary.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2))
    if rows:
        with (output/'metrics_summary.csv').open('w',newline='',encoding='utf-8-sig') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    timestamp=datetime.now(ZoneInfo('Asia/Bangkok')).strftime('%d/%m/%Y %H:%M:%S')
    lines=['# Kết quả huấn luyện CA-TCC trên dữ liệu đầy đủ','',f'Cập nhật: {timestamp} (UTC+7).','',
           f'Cấu hình: {settings["epochs"]} epoch mỗi giai đoạn, batch {settings["batch_size"]}, {settings["label_percent"]}% nhãn thật cho linear/fine-tune, seed {settings["seed"]}. Self-supervised và SupCon dùng toàn bộ tập train. Supervised dùng 100% nhãn train nên khác ngân sách nhãn.', '',
           'Chọn checkpoint phân loại theo validation loss, sau đó đánh giá toàn bộ test. Bảng chỉ chứa giai đoạn đã hoàn thành. Dữ liệu đang chạy chưa có kết quả cuối cùng.','',
           '| Bộ dữ liệu | Giai đoạn | Nhãn thật (%) | Epoch được chọn | Accuracy (%) | Macro-F1 (%) | Balanced accuracy (%) | Số mẫu test |',
           '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f'| {r["dataset"]} | {r["stage"]} | {r["label_percent"]} | {r["selected_epoch"]} | {100*r["accuracy"]:.2f} | {100*r["macro_f1"]:.2f} | {100*r["balanced_accuracy"]:.2f} | {r["test_count"]} |')
    lines += ['', '## Phạm vi kết luận','',
              'Đây là kết quả của một seed và một cách chia người nói cố định. Chưa đo biến thiên qua nhiều seed hoặc nhiều split. Không chọn cấu hình hoặc epoch dựa trên test.','',
              '## Tệp được lưu','',
              '- `metrics_summary.csv`: bảng chỉ số tổng hợp.',
              '- Mỗi thư mục bộ dữ liệu chứa CSV dự đoán theo file, confusion matrix và đường học của fine-tune CA-TCC cuối.',
              '- `bundles/`: mô hình TorchScript, metadata và thống kê chuẩn hóa sau khi export thành công.',
              '- `full_run_status.json`: trạng thái toàn lượt; `pipeline.log`: log huấn luyện.',
              '- `source_snapshot.zip`, `source_sha256.json`, `environment.log`, `packages.log`: mã nguồn và môi trường của lượt chạy.',
              f'- Checkpoint, lịch sử và cấu hình từng giai đoạn: `experiments_logs/{experiment}/`.','']
    (output/'KET_QUA_HUAN_LUYEN.md').write_text('\n'.join(lines),encoding='utf-8')
    print(f'Report updated: {output}; {len(rows)} completed classification stages.',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment',required=True)
    parser.add_argument('--datasets',nargs='+',required=True,choices=['RAVDESS','SpeechCommands'])
    args=parser.parse_args();build(args.experiment,args.datasets)
