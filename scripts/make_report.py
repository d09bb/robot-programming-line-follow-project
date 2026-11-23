"""Build the report PDF and explicitly labelled synthetic vision examples."""
from pathlib import Path
import sys
import cv2
import numpy as np
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.pagesizes import A4
from xml.sax.saxutils import escape
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root/'src/line_follower'))
from line_follower.vision import observe
panels=[]
for x, label in [(320, 'CENTER'), (220, 'LEFT'), (420, 'RIGHT')]:
    frame=np.full((480,640,3), (140,180,210), np.uint8)
    cv2.line(frame,(x,479),(x,260),(0,0,0),14)
    obs=observe(frame)
    for px,py in obs.points:
        cv2.circle(frame,(int(px),py),5,(0,0,255),-1)
    cv2.putText(frame,'SYNTHETIC TEST / '+label,(15,45),cv2.FONT_HERSHEY_SIMPLEX,.75,(0,0,160),2)
    panels.append(cv2.resize(frame,(320,240)))
cv2.imwrite(str(root/'docs/vision_examples.png'),np.hstack(panels))
pdfmetrics.registerFont(UnicodeCIDFont('HYSMyeongJo-Medium'))
style=ParagraphStyle('body', fontName='HYSMyeongJo-Medium',fontSize=10,leading=16,wordWrap='CJK')
story=[]
for para in (root/'docs/report.md').read_text().split('\n\n'):
    story += [Paragraph(escape(para).replace('\n','<br/>'),style),Spacer(1,12)]
story += [Paragraph('합성 영상 검증 예시 (실제 주행 사진 아님)',style),Image(str(root/'docs/vision_examples.png'),width=480,height=120)]
for filename, caption in [('runtime_tracking.png', '실제 시뮬레이터 카메라: 선 인식 및 속도 명령'), ('runtime_line_lost.png', '실제 시뮬레이터 카메라: 다른 실행의 선 유실 및 정지 명령 (완주 아님)')]:
    path = root/'docs'/filename
    if path.exists():
        story += [Spacer(1, 12), Paragraph(caption, style), Image(str(path), width=400, height=300)]
SimpleDocTemplate(str(root/'docs/report.pdf'),pagesize=A4,rightMargin=45,leftMargin=45,topMargin=40,bottomMargin=40).build(story)
