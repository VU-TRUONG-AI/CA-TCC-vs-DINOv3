"""Render the Vietnamese deployment guide with actual evidence and explicit pending work."""
from pathlib import Path
import json
import re
from html import escape
import textwrap
import subprocess
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle, Preformatted, KeepTogether
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_JUSTIFY, TA_CENTER
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.pagesizes import A4

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/pdf/BAO_CAO_CICD_DOCKER_AI.pdf'
OUT.parent.mkdir(parents=True, exist_ok=True)
def matched_font(family, style):
    return subprocess.check_output(['fc-match','-f','%{family}|%{file}',family+':style='+style],text=True).strip().split('|',1)
font_family, regular = matched_font('Times New Roman','Regular')
_, bold = matched_font('Times New Roman','Bold')
for name,file in [('VN',regular),('VN-Bold',bold),('Mono',regular)]:
    pdfmetrics.registerFont(TTFont(name,file))
pdfmetrics.registerFontFamily('VN',normal='VN',bold='VN-Bold',italic='VN',boldItalic='VN-Bold')
exact_font = 'Times New Roman' in font_family
navy=colors.black; teal=colors.black; grey=colors.HexColor('#333333')
styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name='BodyVN',fontName='VN',fontSize=14,leading=21,spaceAfter=7,textColor=navy,alignment=TA_JUSTIFY,firstLineIndent=28.35))
styles.add(ParagraphStyle(name='ListVN',parent=styles['BodyVN'],firstLineIndent=0,leftIndent=12))
styles.add(ParagraphStyle(name='H1VN',fontName='VN-Bold',fontSize=16,leading=23,spaceBefore=14,spaceAfter=12,textColor=navy,keepWithNext=True))
styles.add(ParagraphStyle(name='H2VN',fontName='VN-Bold',fontSize=14,leading=21,spaceBefore=10,spaceAfter=7,textColor=teal,keepWithNext=True))
styles.add(ParagraphStyle(name='SmallVN',fontName='VN',fontSize=12,leading=17,spaceAfter=6,textColor=grey,alignment=TA_JUSTIFY))
styles.add(ParagraphStyle(name='CodeVN',fontName='VN',fontSize=11,leading=15,spaceBefore=6,spaceAfter=10,backColor=colors.HexColor('#f4f4f4'),borderPadding=7))
styles.add(ParagraphStyle(name='TitleVN',fontName='VN-Bold',fontSize=22,leading=32,spaceAfter=22,textColor=navy,alignment=TA_CENTER))

def inline(text):
    text=escape(text)
    text=re.sub(r'\*\*(.*?)\*\*',r'<b>\1</b>',text)
    text=re.sub(r'`([^`]+)`',r'<font name="Mono">\1</font>',text)
    text=re.sub(r'(https?://[^\s<]+)',r'<link href="\1" color="#087f8c">\1</link>',text)
    return text

def p(text,style='BodyVN'):
    return Paragraph(inline(text),styles['ListVN' if style=='BodyVN' and text.startswith('- ') else style])

def table(rows,widths=None):
    data=[[p(str(x),'SmallVN') for x in row] for row in rows]
    t=Table(data,colWidths=widths or [112,327],repeatRows=1,hAlign='LEFT')
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#eeeeee')),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),9),('RIGHTPADDING',(0,0),(-1,-1),9),('TOPPADDING',(0,0),(-1,-1),8),('BOTTOMPADDING',(0,0),(-1,-1),6),('LINEBELOW',(0,0),(-1,0),1,teal),('LINEBELOW',(0,1),(-1,-1),.35,colors.HexColor('#dce4e8'))]))
    return t

def evidence(name):
    path=ROOT/'output/deployment'/name
    return json.loads(path.read_text()) if path.is_file() else None

api=evidence('local-api-verification.json')
onnx=evidence('onnx-verification.json')
container=evidence('container-verification.json')
publication=evidence('publication.json') or {}
complete=bool(container and publication.get('dockerhub_verified') and publication.get('ci_verified') and publication.get('pages_verified'))
state='BÁO CÁO KẾT QUẢ TRIỂN KHAI' if complete else 'BẢN NHÁP - CÒN BƯỚC PHÁT HÀNH CHƯA XÁC NHẬN'
story=[Spacer(1,42),p('CA-TCC + DINOv3','H2VN'),Paragraph('CI/CD cho AI<br/>và Docker image',styles['TitleVN']),p('Hướng dẫn thực hành từng bước bằng tiếng Việt'),p('SpeechCommands: 35 từ khóa | RAVDESS: 8 cảm xúc'),Spacer(1,20),p(state,'H2VN'),p('Ngày lập báo cáo: 31/10/2026. Kết quả được đối chiếu với các tệp kiểm tra đã lưu.','SmallVN'), p('Trình bày: '+('Times New Roman' if exact_font else font_family+' (font tạm, chờ Times New Roman)')+', nội dung 14 pt, giãn dòng 1,5.','SmallVN'),Spacer(1,22)]
story.append(table([['Hạng mục','Trạng thái có bằng chứng'],['API/model trên máy','Đạt với WAV thật' if api and api.get('passed') else 'Chưa xác nhận'],['ONNX dùng cho web','Đạt kiểm tra số học CPU; chưa thay thế kiểm tra trình duyệt' if onnx and onnx.get('passed') else 'Chưa xác nhận'],['Docker container','Đạt kiểm tra container thật' if container and container.get('passed') else 'Chưa có kết quả container đạt'],['CI trên GitHub','Đã xác minh' if publication.get('ci_verified') else 'Có cấu hình; chưa có workflow run thành công được lưu'],['Docker Hub public','Đã push/pull xác minh' if publication.get('dockerhub_verified') else 'Chưa xác nhận public image'],['GitHub Pages CD','Đã triển khai và xác minh' if publication.get('pages_verified') else 'Có cấu hình; chưa xác nhận deploy thành công']]))
story += [Spacer(1,18),p('Không thuê VPS. Web phục vụ tài nguyên tĩnh và chạy AI trên thiết bị người dùng. Docker là nhánh dịch vụ API độc lập.','SmallVN'),PageBreak()]
story += [p('Kiến trúc triển khai và cách đọc báo cáo','H1VN'),p('Hai nhánh phục vụ hai mục đích: image Docker để tích hợp API; web tĩnh để demo khi máy cá nhân đã tắt. Tắt máy sẽ dừng container local nhưng không dừng bản web đã được hosting phân phối.'),table([['Nhánh','Luồng xử lý'],['Docker/API','WAV -> HTTP API -> log-mel + normalization -> TorchScript CPU -> JSON top-k'],['Web tĩnh','Trình duyệt -> tải ONNX chunks + WASM -> xử lý WAV trên thiết bị -> kết quả'],['CI','Push main / pull request -> kiểm tra logic, HTTP và xác minh tài nguyên'],['Phát hành + CD','Tag v* / Run workflow -> build/test -> push/pull Docker Hub -> deploy Pages']]),Spacer(1,12),p('Các kết quả đã kiểm tra','H2VN')]
rows=[['Kiểm tra','Kết quả / giới hạn']]
if api:
    for name,result in api.get('real_wav_parity',{}).items():
        rows.append([f'API {name}',f"Sai khác điểm tối đa: {result['max_probability_error']:.8g}; nhãn cao nhất: {result['response']['top_k'][0]['label']}."])
if onnx:
    for name,result in onnx.get('datasets',{}).items():
        rows.append([f'ONNX {name}',f"Sai khác điểm tối đa: {result['maximum_probability_error']:.8g}; nhãn cao nhất: {result['top_label']}."])
if container:
    for name,result in container.get('real_wav_parity',{}).items():
        rows.append([f'Container {name}',f"Sai khác điểm tối đa: {result['max_probability_error']:.8g}."])
story += [table(rows),Spacer(1,10),p('Các số trên đo tính tương đương suy luận của một số fixture. Đây không phải accuracy/F1 trên tập đánh giá đầy đủ. Các kiểm tra API còn bao gồm WAV lỗi, dataset không hợp lệ, giới hạn thời lượng và Origin không được phép.'),p('Ảnh chụp giao diện và workflow sẽ được bổ sung khi quyền truy cập và công cụ kiểm tra cho phép. Không dùng hình minh họa như bằng chứng đã chạy thực tế.','SmallVN'),PageBreak()]

lines=(ROOT/'deployment/README.md').read_text().splitlines()
paragraph=[]; code=[]; in_code=False; table_rows=[]
def flush_paragraph():
    if paragraph:
        story.append(p(' '.join(paragraph)));paragraph.clear()
def flush_table():
    if table_rows:
        story.append(table(table_rows));story.append(Spacer(1,10));table_rows.clear()
for line in lines:
    if line.startswith('```'):
        flush_paragraph();flush_table()
        if in_code:
            wrapped=[]
            for row in code:
                wrapped.extend(textwrap.wrap(row, width=79, subsequent_indent='  ',replace_whitespace=False,drop_whitespace=False) or [''])
            story.append(Preformatted('\n'.join(wrapped),styles['CodeVN']));code.clear()
        in_code=not in_code;continue
    if in_code:
        code.append(line);continue
    if line.startswith('|'):
        flush_paragraph()
        if not re.match(r'^\|[\s:|-]+\|$',line):
            table_rows.append([c.strip() for c in line.strip('|').split('|')])
        continue
    flush_table()
    if not line.strip():
        flush_paragraph();continue
    if line.startswith('# '):
        continue
    if line.startswith('## '):
        flush_paragraph()
        story.append(p(line[3:],'H1VN'));continue
    if line.startswith('- ') or re.match(r'^\d+\. ',line):
        flush_paragraph();story.append(p(line));continue
    paragraph.append(line)
flush_paragraph();flush_table()
story += [PageBreak(),p('Phụ lục: bằng chứng và bước còn lại','H1VN'),p('Bằng chứng máy đọc được được lưu ở output/deployment/. Các file này ghi kết quả test thực tế; các mục chưa có bằng chứng không được đánh dấu hoàn tất.'),p('Danh sách file bằng chứng','H2VN')]
for name in ['local-api-verification.json','onnx-verification.json','container-verification.json','docker-build.log','image-inspect.json','publication.json']:
    path=ROOT/'output/deployment'/name
    story.append(p(f'{name}: '+('đã có' if path.exists() else 'chưa có'),'SmallVN'))
story.append(p('Trình tự hoàn tất còn lại','H2VN'))
for text in ['Hoàn tất build, chạy container và lưu kết quả regression của hai WAV thật.', 'Kết nối GitHub, lưu model assets thành release và cấu hình variables/secrets.', 'Tạo repository Docker Hub public, chạy workflow phát hành và kiểm tra image đã pull lại.', 'Bật GitHub Pages bằng Actions; kiểm tra URL và dự đoán từ thiết bị khác.', 'Lưu URL workflow/image/site cùng digest thực tế; cập nhật báo cáo với ảnh chụp và bằng chứng cuối cùng.']:
    story.append(p('- '+text))
for key in ('dockerhub_url','ci_url','pages_url'):
    if publication.get(key): story.append(p(f'{key}: {publication[key]}','SmallVN'))

class NumberedCanvas(canvas.Canvas):
    def __init__(self,*a,**k):
        canvas.Canvas.__init__(self,*a,**k);self.states=[]
    def showPage(self):
        self.states.append(dict(self.__dict__));self._startPage()
    def save(self):
        count=len(self.states)
        for state in self.states:
            self.__dict__.update(state)
            self.setStrokeColor(colors.HexColor('#dce4e8'));self.line(44,39,A4[0]-44,39)
            self.setFont('VN',7);self.setFillColor(grey)
            self.drawString(44,26,'CA-TCC + DINOv3 | CI/CD & Docker | 31/10/2026')
            self.drawRightString(A4[0]-44,26,f'{self._pageNumber} / {count}')
            canvas.Canvas.showPage(self)
        canvas.Canvas.save(self)

doc=SimpleDocTemplate(str(OUT),pagesize=A4,leftMargin=99.2,rightMargin=56.7,topMargin=56.7,bottomMargin=56.7,title='CI/CD và Docker cho CA-TCC + DINOv3',author='CA-TCC Audio Project')
doc.build(story,canvasmaker=NumberedCanvas)
print(OUT)
