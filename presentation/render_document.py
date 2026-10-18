from pathlib import Path
import html
import re
import shutil
import xml.etree.ElementTree as ET
import markdown
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle, Image, Preformatted, KeepTogether

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / 'assets'
FONT_SOURCE = ASSETS if (ASSETS / 'IBMPlexSans-Regular.ttf').exists() else ROOT.parents[1] / 'prd/fonts'
ASSETS.mkdir(exist_ok=True)
for name in ['IBMPlexSans-Regular.ttf','IBMPlexSans-SemiBold.ttf','IBMPlexSans-Italic.ttf','IBMPlexMono-Regular.ttf','FONT-LICENSE.txt']:
    if (FONT_SOURCE / name).resolve() != (ASSETS / name).resolve():
        shutil.copyfile(FONT_SOURCE/name,ASSETS/name)
text=(ROOT/'Veyra-Pitch-Content.txt').read_text()

def linkify(match):
    raw=match.group(0)
    url=raw.rstrip('.,;')
    return '<'+url+'>'+raw[len(url):]
text=re.sub(r'(?<![<(])https?://[^\s<>()|]+',linkify,text)
chunks=re.split(r'(?m)^## ',text)
intro=chunks[0].strip()+'\n\n'+chunks[1]
sections=[]
for piece in chunks[2:]:
    title,body=piece.split('\n',1)
    if title.startswith('Slide '):
        num=int(re.match(r'Slide (\d+)',title).group(1)); identifier=f'slide-{num}'
    else:
        num=None;identifier='appendix-'+re.match(r'Appendix ([A-Z])',title).group(1).lower()
    rendered=markdown.markdown(body,extensions=['tables','fenced_code','sane_lists'])
    sections.append({'title':title,'body':body,'html':rendered,'number':num,'id':identifier})
figures={4:['architecture'],5:['data-split'],7:['runtime-recovery'],8:['stream-outcomes','response-outcomes'],10:['native-proof']}
nav=[];content=[]
for sec in sections:
    short=sec['title'].split(' — ',1)[1]
    label=f'{sec["number"]:02d}' if sec['number'] else sec['title'].split(' — ')[0].replace('Appendix ','')
    nav.append(f'<a href="#{sec["id"]}"><span>{label}</span>{html.escape(short)}</a>')
    body=sec['html']
    if sec['number'] in figures:
        images=''.join(f'<figure><img src="visuals/{name}.png" alt="{html.escape(name.replace("-"," "))}" loading="lazy"/></figure>' for name in figures[sec['number']])
        pos=body.find('<h3>Recommended visual</h3>')
        if pos>=0:body=body[:pos]+images+body[pos:]
        else:body+=images
    if sec['number']==4:
        body+='<details><summary>Copy the Mermaid architecture source</summary><pre><code>'+html.escape((ROOT/'visuals/architecture.mmd').read_text())+'</code></pre></details>'
    content.append(f'<section id="{sec["id"]}"><div class="eyebrow">'+('SLIDE '+label if sec['number'] else 'REFERENCE '+label)+f'</div><h2>{html.escape(short)}</h2>{body}</section>')
css='''
@font-face{font-family:Plex;src:url('assets/IBMPlexSans-Regular.ttf')}@font-face{font-family:Plex;src:url('assets/IBMPlexSans-SemiBold.ttf');font-weight:600}@font-face{font-family:Plex;src:url('assets/IBMPlexSans-Italic.ttf');font-style:italic}@font-face{font-family:PlexMono;src:url('assets/IBMPlexMono-Regular.ttf')}
:root{--blue:#0f62fe;--ink:#161616;--muted:#525252;--line:#e0e0e0;--paper:#fff;--teal:#007d79}*{box-sizing:border-box}html{scroll-behavior:smooth;scroll-padding-top:32px}body{margin:0;background:#f4f4f4;color:var(--ink);font:16px/1.6 Plex,Arial,sans-serif}a{color:#0043ce;text-decoration-thickness:1px;text-underline-offset:3px;overflow-wrap:anywhere}aside{position:fixed;inset:0 auto 0 0;width:255px;padding:30px 20px;background:#161616;color:white;overflow:auto}aside .brand{font-size:23px;font-weight:600;letter-spacing:-.7px}aside .sub{font:11px/1.5 PlexMono,monospace;color:#a6c8ff;margin:8px 0 28px}nav a{display:flex;gap:12px;color:#f4f4f4;text-decoration:none;font-size:12px;line-height:1.45;padding:10px 5px;border-bottom:1px solid #393939}nav a:hover{color:#a6c8ff}nav span{font-family:PlexMono;color:#78a9ff;flex:0 0 19px}main{margin-left:255px;max-width:1430px;padding:0 44px 80px}.cover{padding:72px 50px 52px;background:#001141;color:white;margin:0 -44px 36px}.cover .eyebrow{color:#a6c8ff}.cover h1{font-size:68px;letter-spacing:-2.8px;line-height:1.05;margin:18px 0}.cover .tagline{font-size:27px;color:#d0e2ff;max-width:650px}.cover .scope{color:#c6c6c6;font-size:14px;margin-top:28px;max-width:720px}.cover .actions{display:flex;gap:20px;flex-wrap:wrap;margin-top:30px}.cover a{color:white;font-size:13px}.eyebrow{font:12px/1.5 PlexMono,monospace;letter-spacing:1.4px;color:var(--blue)}.intro{background:white;padding:26px 34px;border-left:4px solid var(--blue);margin-bottom:34px}.intro h1,.intro h2{display:none}.intro p{font-size:14px}.intro hr{display:none}section{background:var(--paper);margin-bottom:28px;padding:44px 42px;border:1px solid #e0e0e0}h2{font-size:35px;line-height:1.17;letter-spacing:-1px;margin:12px 0 28px;font-weight:600}h3{font-size:19px;line-height:1.4;margin:32px 0 11px;padding-top:14px;border-top:1px solid var(--line);font-weight:600}p{margin:10px 0 15px;max-width:950px}ul,ol{padding-left:24px}li{margin:8px 0}strong{font-weight:600}code{font:12px/1.6 PlexMono,monospace;background:#f4f4f4;padding:2px 4px;overflow-wrap:anywhere}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#161616;color:#f4f4f4;padding:22px;font:12px/1.7 PlexMono,monospace}pre code{background:none;color:inherit;padding:0}table{width:100%;border-collapse:collapse;margin:18px 0 24px;font-size:13px;line-height:1.5;table-layout:auto}th{background:#edf5ff;color:#001141;text-align:left;border-bottom:2px solid #a6c8ff;padding:11px 12px;font-weight:600}td{padding:11px 12px;border-bottom:1px solid var(--line);vertical-align:top;overflow-wrap:anywhere}tr:nth-child(even) td{background:#fafafa}td code{font-size:10.5px}figure{margin:30px -18px;border:1px solid #e0e0e0}figure img{display:block;width:100%;height:auto}details{margin-top:24px}summary{cursor:pointer;font-weight:600;color:var(--blue)}section>hr{margin-top:35px;border:0;border-top:1px solid #e0e0e0}footer{font-size:12px;color:#525252;padding:16px 0}.note{color:var(--muted)}
@media(max-width:1050px){aside{position:relative;width:auto;max-height:330px}aside nav{display:grid;grid-template-columns:repeat(2,1fr);gap:0 20px}main{margin:0;padding:0 22px 45px}.cover{margin:0 -22px 24px;padding:45px 28px}.cover h1{font-size:52px}section{padding:30px 25px}table{font-size:12px}}
@media print{@page{size:A4;margin:15mm}body{background:white;font-size:10pt}aside{display:none}main{margin:0;padding:0;max-width:none}.cover{margin:0;padding:22mm 14mm;break-after:page}.cover h1{font-size:46pt}.intro{border:0;padding:0}section{border:0;padding:0;margin:0;break-before:page}h2{font-size:24pt}h3{font-size:13pt;break-after:avoid}table{font-size:8pt}th,td{padding:6px}tr{break-inside:avoid}figure{margin:12pt 0;break-inside:avoid}details{display:none}a{color:inherit}nav,.actions{display:none}}
'''
html_doc='<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Veyra — IBM Z Pitch Content</title><style>'+css+'</style></head><body>'
html_doc+='<aside><div class="brand">Veyra</div><div class="sub">IBM Z DATATHON<br>CONTENT + EVIDENCE</div><nav>'+''.join(nav)+'</nav></aside><main>'
html_doc+='<header class="cover"><div class="eyebrow">IBM Z DATATHON / PRESENTATION CONTENT</div><h1>Veyra</h1><div class="tagline">Observe change.<br>Verify recovery. Keep control.</div><p class="scope">Ten slide briefs. Exact evidence. A native LinuxONE story.<br>This is a content workbook, not a generated PowerPoint or a product UI.</p><div class="actions"><a href="Veyra-Pitch-Content.pdf">PDF workbook</a><a href="Veyra-Pitch-Content.txt">Copy-ready text</a><a href="visuals/architecture.svg">Editable architecture</a><a href="visuals/architecture.mmd">Mermaid source</a></div></header>'
html_doc+='<div class="intro">'+markdown.markdown(intro,extensions=['tables','fenced_code'])+'</div>'+''.join(content)
html_doc+='<footer>Evidence cutoff: public release 8a86ce0 / qualified core c5fad06. Charts derive from frozen native evidence. No raw requests, weights, credentials, or server addresses are included.</footer></main></body></html>'
(ROOT/'Veyra-Pitch-Content.html').write_text(html_doc)

for name,file in [('Plex','IBMPlexSans-Regular.ttf'),('PlexSemi','IBMPlexSans-SemiBold.ttf'),('PlexItalic','IBMPlexSans-Italic.ttf'),('PlexMono','IBMPlexMono-Regular.ttf')]:
    pdfmetrics.registerFont(TTFont(name,str(ASSETS/file)))
pdfmetrics.registerFontFamily('Plex',normal='Plex',bold='PlexSemi',italic='PlexItalic',boldItalic='PlexSemi')
W,H=A4
styles={
 'body':ParagraphStyle('body',fontName='Plex',fontSize=9.4,leading=13.4,spaceAfter=8,textColor=colors.HexColor('#262626'),splitLongWords=True),
 'h2':ParagraphStyle('h2',fontName='PlexSemi',fontSize=25,leading=29,spaceAfter=20,textColor=colors.HexColor('#161616'),keepWithNext=True),
 'h3':ParagraphStyle('h3',fontName='PlexSemi',fontSize=12.5,leading=16,spaceBefore=15,spaceAfter=7,textColor=colors.HexColor('#0043ce'),keepWithNext=True),
 'cell':ParagraphStyle('cell',fontName='Plex',fontSize=7.7,leading=10.5,spaceAfter=0,splitLongWords=True),
 'headcell':ParagraphStyle('headcell',fontName='PlexSemi',fontSize=7.8,leading=10.5,spaceAfter=0,textColor=colors.HexColor('#001141'),splitLongWords=True),
 'kicker':ParagraphStyle('kicker',fontName='PlexMono',fontSize=9,leading=12,textColor=colors.HexColor('#0f62fe'),spaceAfter=9,keepWithNext=True),
 'bullet':ParagraphStyle('bullet',fontName='Plex',fontSize=9.4,leading=13.4,leftIndent=13,firstLineIndent=-10,spaceAfter=6,splitLongWords=True),
 'code':ParagraphStyle('code',fontName='PlexMono',fontSize=6.4,leading=9,backColor=colors.HexColor('#f4f4f4'),borderPadding=8,spaceBefore=6,spaceAfter=10),
}

def inline(node):
    parts=[html.escape(node.text or '')]
    for child in node:
        content=inline(child)
        if child.tag=='strong':content='<b>'+content+'</b>'
        elif child.tag=='em':content='<i>'+content+'</i>'
        elif child.tag=='code':content='<font name="PlexMono" size="7.5">'+content+'</font>'
        elif child.tag=='a':content='<a href="'+html.escape(child.attrib.get('href',''),quote=True)+'" color="#0043ce">'+content+'</a>'
        elif child.tag=='br':content='<br/>'
        parts.append(content);parts.append(html.escape(child.tail or ''))
    return ''.join(parts)

def flows(fragment):
    tree=ET.fromstring('<root>'+fragment+'</root>')
    output=[]
    for node in tree:
        if node.tag in ('h1','h2'):output.append(Paragraph(inline(node),styles['h2']))
        elif node.tag=='h3':output.append(Paragraph(inline(node),styles['h3']))
        elif node.tag=='p':output.append(Paragraph(inline(node),styles['body']))
        elif node.tag in ('ul','ol'):
            for i,item in enumerate(node.findall('li'),1):
                prefix=f'{i}. ' if node.tag=='ol' else '• '
                output.append(Paragraph(prefix+inline(item),styles['bullet']))
        elif node.tag=='table':
            rows=[]
            for row in node.findall('.//tr'):
                cells=[]
                for cell in row:
                    cells.append(Paragraph(inline(cell),styles['headcell'] if cell.tag=='th' else styles['cell']))
                rows.append(cells)
            count=len(rows[0]);available=W-92
            if count==2: widths=[available*.49,available*.51]
            elif count==3:widths=[available*.28,available*.34,available*.38]
            elif count==4:widths=[available*.23,available*.23,available*.32,available*.22]
            else:widths=[available/count]*count
            table=Table(rows,colWidths=widths,repeatRows=1,hAlign='LEFT')
            table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#edf5ff')),
                ('LINEBELOW',(0,0),(-1,0),.8,colors.HexColor('#a6c8ff')),
                ('LINEBELOW',(0,1),(-1,-1),.25,colors.HexColor('#e0e0e0')),
                ('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),6),
                ('RIGHTPADDING',(0,0),(-1,-1),6),('TOPPADDING',(0,0),(-1,-1),7),
                ('BOTTOMPADDING',(0,0),(-1,-1),7),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#fafafa')])]))
            output += [Spacer(1,5),table,Spacer(1,12)]
        elif node.tag=='pre':
            value=''.join(node.itertext())
            lines=[]
            for line in value.splitlines():
                while len(line)>104:
                    cut=line.rfind(' ',0,104)
                    if cut<40:cut=104
                    lines.append(line[:cut]);line='  '+line[cut:].lstrip()
                lines.append(line)
            output.append(Preformatted('\n'.join(lines),styles['code']))
        elif node.tag=='hr':output.append(Spacer(1,8))
    return output

class Workbook(SimpleDocTemplate):
    def afterFlowable(self,flowable):
        if isinstance(flowable,Paragraph) and flowable.style.name=='h2':
            title=flowable.getPlainText()
            key='section-'+str(self.page)+'-'+str(len(title))
            self.canv.bookmarkPage(key)
            self.canv.addOutlineEntry(title,key,0,False)

def frame(c,doc):
    c.saveState()
    if doc.page>1:
        c.setFont('PlexMono',7.5);c.setFillColor(colors.HexColor('#525252'))
        c.drawString(46,H-28,'VEYRA  /  IBM Z DATATHON  /  CONTENT + EVIDENCE')
        c.setStrokeColor(colors.HexColor('#e0e0e0'));c.line(46,H-36,W-46,H-36)
    c.setFont('PlexMono',7.5);c.setFillColor(colors.HexColor('#6f6f6f'))
    c.drawString(46,25,'Native LinuxONE · Frozen release evidence')
    c.drawRightString(W-46,25,f'{doc.page:02d}')
    c.restoreState()

story=[Spacer(1,85),Paragraph('IBM Z DATATHON',styles['kicker']),
       Paragraph('Veyra',ParagraphStyle('coverTitle',fontName='PlexSemi',fontSize=43,leading=49,spaceAfter=24)),
       Paragraph('Observe change.<br/>Verify recovery.<br/>Keep control.',ParagraphStyle('coverTag',fontName='Plex',fontSize=26,leading=32,textColor=colors.HexColor('#0f62fe'),spaceAfter=28)),
       Paragraph('Ten-slide content workbook',styles['h3']),
       Paragraph('Presentation-ready points, technical detail, speaker notes, source references, native evidence, editable diagrams, and a three-minute demonstration plan.',styles['body']),
       Spacer(1,26),Paragraph('Public release: 8a86ce0<br/>Qualified native core: c5fad06<br/>Selected study: synthetic-100k-20261017-01',styles['body']),
       Spacer(1,24),Paragraph('This document is not a PowerPoint or a product interface. Main-slide copy stays short; detailed evidence supports preparation and questions.',styles['body']),PageBreak()]
story += [Paragraph('How to use this workbook',styles['h2'])]
intro_body=re.sub(r'^#.*\n','',intro,count=1)
intro_body=re.sub(r'^IBM Z Datathon.*\n','',intro_body)
story += flows(markdown.markdown(intro_body,extensions=['tables','fenced_code']))
for sec in sections:
    story.append(PageBreak())
    story.append(Paragraph(('SLIDE '+str(sec['number']).zfill(2)) if sec['number'] else sec['title'].split(' — ')[0].upper(),styles['kicker']))
    story.append(Paragraph(sec['title'].split(' — ',1)[1],styles['h2']))
    if sec['number'] in figures:
        for name in figures[sec['number']]:
            story.append(Image(str(ROOT/'visuals'/f'{name}.png'),width=W-92,height=(W-92)*9/16))
            story.append(Spacer(1,14))
    story += flows(sec['html'])
doc=Workbook(str(ROOT/'Veyra-Pitch-Content.pdf'),pagesize=A4,
             rightMargin=46,leftMargin=46,topMargin=52,bottomMargin=48,
             title='Veyra — IBM Z Datathon Pitch Content',author='Veyra')
doc.build(story,onFirstPage=frame,onLaterPages=frame)
print('Created HTML and PDF workbooks.')
