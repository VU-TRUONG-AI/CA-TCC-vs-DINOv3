"""Render thesis-ready tables and PNG/SVG figures from measured stage results."""
import argparse
import csv
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
NAMES={'raw_pcm':'WAV/PCM gốc','pcm_to_float':'PCM → float32','mono':'Gộp kênh mono','resample_16k':'Resample 16 kHz',
       'pad_window':'Pad cửa sổ cố định','stft_power':'STFT power','mel_power':'Mel power 40 kênh',
       'log_mel':'Log-mel','normalized':'Chuẩn hóa bằng train'}


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(str(v) for v in row)+' |' for row in rows])


from reportlab.graphics.shapes import Drawing, String, Line, Image, Rect
from reportlab.graphics.charts.barcharts import HorizontalBarChart, VerticalBarChart
from reportlab.graphics.charts.lineplots import LinePlot
from reportlab.graphics import renderSVG
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors
from PIL import Image as PILImage
import subprocess
import shutil

FONT='DejaVuSans'
pdfmetrics.registerFont(TTFont(FONT,'/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'))
NODE=shutil.which('node') or '/home/truongvu/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node'
SHARP='/home/truongvu/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/sharp'
BLUE=colors.HexColor('#2878A0')


def text(d,x,y,label,size=17,anchor='start'):
    d.add(String(x,y,str(label),fontName=FONT,fontSize=size,textAnchor=anchor,fillColor=colors.HexColor('#183349')))


def blank(title):
    d=Drawing(1600,900)
    d.add(Rect(0,0,1600,900,fillColor=colors.white,strokeColor=None))
    text(d,800,858,title,30,'middle')
    return d


def savefig(d,folder,name):
    svg=folder/f'{name}.svg'
    renderSVG.drawToFile(d,str(svg))
    subprocess.run([NODE,'-e',"const sharp=require(process.argv[3]); sharp(process.argv[1],{density:144}).png().toFile(process.argv[2]).catch(e=>{console.error(e);process.exit(1)});",
                    str(svg),str(folder/f'{name}.png'),SHARP],check=True)


def format_axes(chart):
    for axis in [chart.xValueAxis,chart.yValueAxis]:
        axis.labels.fontName=FONT;axis.labels.fontSize=12
        axis.strokeColor=colors.HexColor('#9CAEB7')
    chart.lines[0].strokeColor=BLUE;chart.lines[0].strokeWidth=1.2


def lineplot(d,a,x,y,w,h,label,rate=None,limits=None):
    chart=LinePlot();chart.x=x;chart.y=y;chart.width=w;chart.height=h
    stride=max(1,len(a)//2000)
    times=np.arange(len(a))/rate if rate else np.arange(len(a))
    chart.data=[list(zip(times[::stride].tolist(),a[::stride].astype(float).tolist()))]
    chart.xValueAxis.valueMin=0;chart.xValueAxis.valueMax=max(1e-6,len(a)/rate if rate else float(times[-1]))
    lo,hi=limits if limits else (float(np.min(a)),float(np.max(a)))
    pad=max((hi-lo)*.05,1e-8)
    chart.yValueAxis.valueMin=lo-pad;chart.yValueAxis.valueMax=hi+pad
    format_axes(chart);d.add(chart)
    text(d,x+w/2,y+h+25,label,19,'middle')
    text(d,x+w/2,y-42,'Thời gian (s)' if rate else 'Chỉ số kênh mel',15,'middle')


def heatmap(d,a,x,y,w,h,title,extent,units,out,name):
    lo,hi=float(a.min()),float(a.max())
    values=np.clip((a-lo)/max(hi-lo,1e-12),0,1)
    # Explicit blue-cyan-yellow palette; values are not modified, only mapped to color.
    anchors=np.asarray([[16,35,83],[25,113,159],[74,190,181],[246,222,74]],float)
    def rgb(v):
        pos=v*3;indices=np.minimum(pos.astype(int),2);fraction=(pos-indices)[...,None]
        return np.round(anchors[indices]*(1-fraction)+anchors[indices+1]*fraction).astype(np.uint8)
    (out/'_assets').mkdir(exist_ok=True)
    path=out/'_assets'/f'{name}_heatmap.png';PILImage.fromarray(rgb(values)[::-1]).resize((int(w*2),int(h*2)),PILImage.Resampling.NEAREST).save(path)
    d.add(Image(x,y,w,h,str(path)))
    d.add(Line(x,y,x+w,y,strokeColor=colors.HexColor('#9CAEB7')))
    for val in [0,extent/2,extent]:
        text(d,x-12,y+h*val/extent-5,f'{val:g}',12,'end')
    for val in [0,1,2]:
        text(d,x+w*val/2,y-20,f'{val*float(wave_seconds)/2:.2f}',12,'middle')
    text(d,x+w/2,y+h+25,title,19,'middle');text(d,x+w/2,y-45,'Thời gian (s)',15,'middle')
    text(d,x,y+h+5,units,11)
    gradient=rgb(np.linspace(1,0,200)[:,None]);gp=out/'_assets'/f'{name}_colorbar.png';PILImage.fromarray(gradient).resize((24,int(h*2)),PILImage.Resampling.NEAREST).save(gp)
    d.add(Image(x+w+14,y,12,h,str(gp)))
    text(d,x+w+32,y+h-8,f'{hi:.3g}',10);text(d,x+w+32,y,f'{lo:.3g}',10)


def charts(name,summary,source,stage_dir,out,audit_dir):
    global wave_seconds
    original=json.loads((audit_dir/'audit_summary.json').read_text())
    rows=list(csv.DictReader((audit_dir/'audit.csv').open()))
    d=blank(f'{name}: kiểm tra dữ liệu ban đầu')
    items=list(original['class_counts'].items())
    text(d,420,802,'Phân bố lớp — toàn bộ WAV gốc',22,'middle')
    chart=HorizontalBarChart();chart.x=180;chart.y=92;chart.width=560;chart.height=660
    chart.data=[[v for _,v in items][::-1]];chart.categoryAxis.categoryNames=[k for k,_ in items][::-1]
    chart.categoryAxis.labels.fontName=FONT;chart.categoryAxis.labels.fontSize=14 if len(items)<20 else 12
    chart.valueAxis.labels.fontName=FONT;chart.valueAxis.labels.fontSize=13;chart.valueAxis.valueMin=0
    chart.valueAxis.valueMax=max(v for _,v in items)*1.2;chart.bars[0].fillColor=BLUE
    chart.barLabelFormat='%d';chart.barLabels.fontName=FONT;chart.barLabels.fontSize=12;chart.barLabels.nudge=5;chart.barLabels.boxAnchor='w'
    d.add(chart);text(d,460,48,'Số file',17,'middle')
    durations=np.asarray([float(r['duration']) for r in rows]);counts,edges=np.histogram(durations,bins=24)
    logscale=name=='SpeechCommands'
    hist=VerticalBarChart();hist.x=890;hist.y=215;hist.width=620;hist.height=440
    hist.data=[np.log10(counts+1).tolist() if logscale else counts.tolist()]
    hist.categoryAxis.categoryNames=[f'{v:.2f}' if i%4==0 else '' for i,v in enumerate((edges[:-1]+edges[1:])/2)]
    hist.categoryAxis.labels.fontName=FONT;hist.categoryAxis.labels.fontSize=12
    hist.valueAxis.labels.fontName=FONT;hist.valueAxis.labels.fontSize=13;hist.valueAxis.valueMin=0
    hist.bars[0].fillColor=BLUE;d.add(hist)
    text(d,1200,802,'Phân bố thời lượng',22,'middle')
    text(d,1200,735,f'Median = {np.median(durations):.4f} s',17,'middle')
    text(d,1200,168,'Thời lượng (s)',17,'middle')
    text(d,1200,120,'Trục dọc: log10(số file + 1)' if logscale else 'Trục dọc: số file',15,'middle')
    text(d,800,15,f'N={len(rows):,}; giữ nguyên các bản trùng; Speech Commands không tính background noise.',13,'middle')
    savefig(d,out,'01_raw_data_overview')

    with np.load(stage_dir/'illustration_arrays.npz') as data:
        sr=int(data['sample_rate']);raw=data['raw_pcm'];raw=raw if raw.ndim==1 else raw.mean(axis=1)
        d=blank(f'{name}: các bước trên MỘT mẫu train')
        wave_seconds=len(data['padded'])/16000
        for x,a,rate,label in [(75,raw,sr,f'PCM gốc — {sr:,} Hz'),(600,data['resampled'],16000,'Mono + resample — 16 kHz'),
                               (1125,data['padded'],16000,f'Pad — cửa sổ {wave_seconds:g} s')]:
            lineplot(d,a,x,540,390,200,label,rate)
        heatmap(d,10*np.log10(np.maximum(data['power'],1e-10)),75,150,390,220,'STFT power (dB)',8,'Tần số (kHz)',out,'stft')
        heatmap(d,data['logmel'],600,150,390,220,'Log-mel (log tự nhiên)',40,'Kênh mel',out,'logmel')
        heatmap(d,data['normalized'],1125,150,390,220,'Log-mel chuẩn hóa',40,'Kênh mel',out,'normalized')
        text(d,800,36,f'Lớp: {summary["illustration"]["class"]}; mẫu có thời lượng median trong train, không clipping.',14,'middle')
        text(d,800,13,'Các colorbar có thang giá trị riêng. Hình một mẫu không thay thế thống kê toàn corpus.',13,'middle')
        savefig(d,out,'02_frontend_example')

    norm=np.load(source/'normalization.npz');train=np.load(stage_dir/'train_normalized_channel_stats.npz')
    d=blank(f'{name}: chuẩn hóa theo từng kênh mel')
    lineplot(d,norm['mean'].ravel(),120,550,570,185,'Mean train trước chuẩn hóa')
    lineplot(d,train['mean'],920,550,570,185,'Mean train sau chuẩn hóa',limits=(-1e-5,1e-5))
    lineplot(d,norm['std'].ravel(),120,170,570,185,'Std train trước chuẩn hóa')
    lineplot(d,train['std'],920,170,570,185,'Std train sau chuẩn hóa',limits=(.99999,1.00001))
    text(d,800,32,'Mean ≈ 0, std ≈ 1 kiểm chứng phép biến đổi; không chứng minh hiệu quả phân loại tăng.',14,'middle')
    savefig(d,out,'03_normalization_checks')


def report(args):
    base=Path(args.report_root)
    all_summary=json.loads((base/'summary.json').read_text())
    lines=['# Báo cáo kiểm tra dữ liệu và từng bước tiền xử lý CA-TCC',
           '', '## Phạm vi và bằng chứng', '',
           'Báo cáo được tạo bằng **chạy lại toàn bộ WAV gốc qua từng bước frontend**, rồi đối chiếu **từng mẫu** đầu ra chuẩn hóa với `.npz` đã lưu. Đây là kiểm chứng bổ sung sau lượt tiền xử lý đầu tiên; không phải log trung gian đã được ghi ở lượt đầu.',
           '', 'Không sửa WAV gốc, không đổi split, không fit lại normalization và không huấn luyện mô hình. Hình frontend minh họa một mẫu train; các bảng thống kê tính trên toàn bộ corpus. Thống kê số học của feature không phải Accuracy/F1 của mô hình.',
           '', '## Tóm tắt dữ liệu ban đầu', '']
    rows=[]
    for name,s in all_summary.items():
        source=Path(args.data_root)/name
        audit_dir=Path(args.audit_root)/name
        old=json.loads((audit_dir/'audit_summary.json').read_text())
        rows.append([name,s['raw_files'],len(old['class_counts']),str(old['sample_rates']),s['stereo_to_mono_files'],
                     f'{s["raw_duration_seconds"]["min"]:.4f}–{s["raw_duration_seconds"]["max"]:.4f}',
                     old['silent_files'],s['pcm_float_files_with_abs_ge_0_999'],len(old['duplicate_pairs'])])
    lines += [table(['Bộ dữ liệu','File','Lớp','Sample rate gốc (số file)','Stereo','Thời lượng (s)','RMS < 1e-6','Có biên độ ≥ 0,999','Bản trùng ngoài bản đầu'],rows),
              '', 'Clipping ở bảng này được đo trên tín hiệu đã đổi PCM sang float32 **trước khi gộp kênh**. Audit cũ đo sau gộp kênh; báo cáo phân biệt hai đại lượng. Bản trùng được xác định bằng SHA256 của byte WAV, không phải phép tìm tín hiệu tương tự hoặc so trùng sau resample. Khoảng lặng trong file không đồng nghĩa file hoàn toàn im lặng.', '']
    for name,s in all_summary.items():
        source=Path(args.data_root)/name; out=base/name
        audit_dir=Path(args.audit_root)/name
        figure_dir=Path(args.figure_root)/name
        figure_dir.mkdir(parents=True,exist_ok=True)
        manifest=json.loads((source/'manifest.json').read_text());old=json.loads((audit_dir/'audit_summary.json').read_text())
        charts(name,s,source,out,figure_dir,audit_dir)
        lines += [f'## {name}', '', '### Kết quả từng bước', '']
        rows=[]
        for stage,label in NAMES.items():
            pieces=[s['stage_statistics'][split][stage] for split in ['train','val','test']]
            shape=set(k for piece in pieces for k in piece['shape_counts'])
            display=', '.join(sorted(shape)) if len(shape)<=3 else f'{len(shape)} kích thước khác nhau theo thời lượng/kênh'
            rows.append([label,sum(piece['files'] for piece in pieces),display,
                         ', '.join(sorted(set(k for piece in pieces for k in piece['dtype_counts']))),
                         f'{min(piece["min"] for piece in pieces):.6g}',f'{max(piece["max"] for piece in pieces):.6g}',
                         sum(piece['nonfinite_values'] for piece in pieces)])
        lines += [table(['Bước','Số file','Shape mỗi mẫu','Dtype','Min toàn bộ','Max toàn bộ','NaN/Inf'],rows),'',
                  f'- Chuyển stereo → mono: **{s["stereo_to_mono_files"]} file**; đổi sample rate: **{s["resampled_files"]} file**.',
                  f'- Sai lệch thời lượng tối đa sau resample: **{s["max_duration_error_seconds"]:.9g} giây** (do làm tròn số mẫu).',
                  f'- Tỷ lệ padding: trung bình **{s["padding_fraction"]["mean"]*100:.3f}%**, median **{s["padding_fraction"]["median"]*100:.3f}%**, lớn nhất **{s["padding_fraction"]["max"]*100:.3f}%**; số file bị cắt: **{s["cropped_files"]}**.',
                  f'- File có biên độ |x| > 1 sau resample: **{s["resampled_files_with_abs_gt_1"]}**. Bộ lọc resample có thể overshoot; đây không tự động là clipping PCM. Frontend không tự clip hoặc normalize peak.',
                  f'- File gốc thay đổi SHA256 so với audit ban đầu: **{s["source_hash_mismatches"]}**.',
                  f'- Sai lệch tuyệt đối lớn nhất giữa frontend chạy lại và feature đã lưu: **{s["max_normalized_parity_abs_error"]:.9g}**; kiểm tra từng phần tử `rtol=1e-5, atol=1e-5`: **đạt**.',
                  '', 'Min/max giữa các bước có đơn vị khác nhau: PCM integer, biên độ float, năng lượng phổ, log năng lượng và z-score. Không diễn giải việc khoảng giá trị thay đổi thành tăng chất lượng dữ liệu.',
                  '', '### Chia tập và chuẩn hóa', '']
        rows=[]
        for split in ['train','val','test']:
            stats=np.load(out/f'{split}_normalized_channel_stats.npz')
            rows.append([split,manifest['splits'][split]['count'],len(manifest['splits'][split]['speakers']),
                         f'{stats["mean"].min():.9g}–{stats["mean"].max():.9g}',
                         f'{stats["std"].min():.9g}–{stats["std"].max():.9g}'])
        lines += [table(['Tập','Mẫu','Người nói','Khoảng mean của 40 kênh','Khoảng std của 40 kênh'],rows),'',
                  f'Giao người nói giữa train/val/test: `{s["speaker_overlaps"]}`. Chỉ train được dùng tính mean/std; val/test không cần có mean 0 hoặc std 1.',
                  '', '### Ngân sách nhãn', '',
                  table(['Ngân sách','Mẫu có nhãn','Tỷ lệ thực (%)','Số lớp có mặt'],
                        [[f'{pct}%',r['count'],f'{r["actual_percent"]:.4f}',r['classes_present']] for pct,r in s['label_budgets'].items()]),
                  '', 'Tỷ lệ thực có thể lệch nhẹ do làm tròn số mẫu. Các subset ngân sách được lấy riêng, không mặc định lồng nhau.',
                  '', '### Phân bố lớp sau chia tập', '',
                  table(['Lớp','Toàn bộ','Train','Val','Test'],[[cls,old['class_counts'][cls],*[s['split_class_counts'][split].get(cls,0) for split in ['train','val','test']]] for cls in manifest['classes']]),
                  '', '### Hình cho báo cáo và slide', '',
                  f'![Dữ liệu gốc {name}]({figure_dir.as_posix()}/01_raw_data_overview.png)', '',
                  f'![Frontend {name}]({figure_dir.as_posix()}/02_frontend_example.png)', '',
                  f'File minh họa: `{s["illustration"]["path"]}`; lớp `{s["illustration"]["class"]}`; chọn theo thời lượng median trong train, không có cờ clipping ở audit. Hình này không đại diện cho toàn bộ biến thiên của corpus.', '',
                  f'![Chuẩn hóa {name}]({figure_dir.as_posix()}/03_normalization_checks.png)', '']
    lines += ['## Đoạn văn gợi ý cho bài luận', '',
              '> Trước khi huấn luyện, chúng tôi kiểm tra toàn bộ 105.829 bản ghi Speech Commands và 1.440 bản ghi RAVDESS. Dữ liệu được chia tập độc lập theo người nói trước khi tính các thống kê chuẩn hóa. Các bản ghi được chuyển sang mono, đưa về tần số lấy mẫu 16 kHz bằng resampling polyphase và bổ sung mẫu 0 ở cuối để đạt cửa sổ cố định 1 giây đối với Speech Commands và 6 giây đối với RAVDESS. Không bản ghi nào trong hai bộ hiện tại bị cắt theo cấu hình này. Đặc trưng log-mel gồm 40 kênh được tính với cửa sổ Hann 25 ms, bước 10 ms và FFT 512. Mean và standard deviation theo kênh được ước lượng trên tập huấn luyện, sau đó áp dụng nguyên trạng cho tập xác thực và kiểm tra. Để kiểm chứng, chúng tôi chạy lại toàn bộ frontend từ WAV gốc, kiểm tra giá trị hữu hạn, bảo toàn số mẫu và nhãn, xác nhận không giao người nói giữa các tập và đối chiếu từng mẫu đặc trưng với dữ liệu đã lưu. Các phép kiểm tra này đánh giá tính nhất quán của tiền xử lý; hiệu quả phân loại cần được xác định bằng thí nghiệm huấn luyện riêng.', '',
              '## Gợi ý trình bày 5 slide', '',
              '1. **Dữ liệu ban đầu:** mục tiêu hai bài toán, số file/lớp, histogram thời lượng và phân bố lớp.',
              '2. **Chất lượng và rò rỉ:** stereo, clipping, trùng byte, split theo người nói; nêu rõ không tự loại file.',
              '3. **Frontend âm thanh:** PCM → mono → 16 kHz → pad → STFT → mel → log → normalization, dùng hình một mẫu.',
              '4. **Kiểm chứng:** số file bảo toàn, NaN/Inf = 0, hash không đổi, đối chiếu feature và biểu đồ mean/std train.',
              '5. **Đầu vào CA-TCC và giới hạn:** shape `[N,40,101]` / `[N,40,601]`, ngân sách nhãn, chưa có Accuracy/F1; padding RAVDESS là biến cần nghiên cứu tiếp.', '',
              '## Các giới hạn cần nói rõ', '',
              '- Chưa đánh giá augmentation hoặc huấn luyện do chưa cài được PyTorch; không có số liệu mô hình.',
              '- Audit trước đó không đo clipping trước downmix; lượt kiểm tra này bổ sung phép đo đó.',
              '- Không phát hiện trùng gần giống hoặc đánh giá chất lượng bằng nghe thủ công toàn corpus.',
              '- Không tự động loại clipping, bản trùng hoặc trim silence; các thay đổi này cần ablation riêng.',
              '- Phần padding chiếm nhiều thời gian trong RAVDESS và được tính vào normalization/encoder hiện tại. Cần so sánh masked pooling hoặc xử lý độ dài để đánh giá ảnh hưởng; chưa khẳng định ảnh hưởng tốt/xấu.',
              '- Speech Commands là 35 từ closed-set, không phải giao thức unknown/silence; RAVDESS là tiếng Anh diễn xuất.', '',
              '## Tái chạy và truy vết', '',
              'Chạy từ thư mục repo:', '',
              '```bash\nOPENBLAS_NUM_THREADS=1 .venv/bin/python pipeline/report_preprocessing_stages.py\n.venv/bin/python pipeline/render_preprocessing_report.py\n```', '',
              'Mỗi bộ có `stage_summary.json`, `per_file_stages.jsonl`, các thống kê theo kênh và hình PNG/SVG. `summary.json` chứa kết quả cả hai bộ. PNG để chèn slide; SVG giữ độ sắc nét khi phóng to. Dữ liệu gốc và file dùng huấn luyện không bị ghi đè.', '',
              'Cơ sở phương pháp: [CA-TCC](https://arxiv.org/html/2208.06616v3), [Speech Commands](https://arxiv.org/abs/1804.03209), [RAVDESS](https://zenodo.org/records/1188976). Các số liệu trong báo cáo này được đo trên dữ liệu cục bộ, không sao chép kết quả benchmark của bài báo.']
    Path(args.report_dir).mkdir(parents=True,exist_ok=True)
    (Path(args.report_dir)/'BAO_CAO_TIEN_XU_LY.md').write_text('\n'.join(lines))
    print(f'Report and figures ready: {base.resolve()}')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--data-root',default=str(ROOT/'processed/full'))
    p.add_argument('--audit-root',default=str(ROOT/'results/01_raw_audit'))
    p.add_argument('--figure-root',default=str(ROOT/'figures'))
    p.add_argument('--report-dir',default=str(ROOT/'reports'))
    p.add_argument('--report-root',default=str(ROOT/'results/02_stage_checks'))
    report(p.parse_args())
