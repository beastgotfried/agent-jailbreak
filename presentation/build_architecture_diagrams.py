"""Generate Veyra diagram images and a vector PDF. No model execution occurs."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import subprocess
import zipfile

from PIL import Image, ImageDraw
from pypdf import PdfReader
from reportlab.lib.colors import HexColor
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / "web/veyra/public/resources"
IMAGES = PUBLIC / "architecture-images"
PDF = PUBLIC / "Veyra-Architecture-and-Flowcharts.pdf"
ZIP = PUBLIC / "Veyra-Diagram-Images.zip"
FONT_DIR = ROOT / "web/veyra/public/fonts"
W, H = 1280, 720
TOTAL = 4
FONT = "VeyraSans"
BOLD = "VeyraSansBold"
MONO = "VeyraMono"
PALETTE = {
    "step": ("#F0F5FF", "#7C9CC7", "#102A43"),
    "decision": ("#E8F0FF", "#0F62FE", "#102A43"),
    "verified": ("#E5F5F1", "#007D79", "#004D49"),
    "blocked": ("#FFF2D9", "#B07716", "#684400"),
    "record": ("#F2F4F7", "#7B899B", "#243448"),
    "error": ("#FFF0F0", "#B34242", "#762828"),
}


def hexcolor(value):
    return HexColor(value)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def wrap(text, width, font, size):
    lines = []
    for paragraph in text.split("\n"):
        line = ""
        for word in paragraph.split():
            candidate = (line + " " + word).strip()
            if line and pdfmetrics.stringWidth(candidate, font, size) > width:
                lines.append(line)
                line = word
            else:
                line = candidate
        lines.append(line)
    return lines


@dataclass
class Node:
    key: str
    x: float
    y: float
    w: float
    h: float
    label: str
    kind: str = "step"
    big: bool = False

    def port(self, side):
        return {
            "left": (self.x, self.y + self.h / 2),
            "right": (self.x + self.w, self.y + self.h / 2),
            "top": (self.x + self.w / 2, self.y),
            "bottom": (self.x + self.w / 2, self.y + self.h),
        }[side]


class Diagram:
    def __init__(self, c, number, slug, title, subtitle, notes):
        self.c = c
        self.number, self.slug = number, slug
        self.title, self.subtitle, self.notes = title, subtitle, notes
        self.nodes = {}
        self.edges = []
        self.font_sizes = []

    def node(self, key, x, y, w, h, label, kind="step", big=False):
        assert key not in self.nodes
        self.nodes[key] = Node(key, x, y, w, h, label, kind, big)
        return key

    def edge(self, a, b, start="right", end="left", via=(), label="", at=None):
        self.edges.append((a, b, start, end, tuple(via), label, at))

    def label(self, text, x, y, size=12.5):
        c = self.c
        width = pdfmetrics.stringWidth(text, BOLD, size)
        c.setFillColor(hexcolor("#FFFFFF"))
        c.roundRect(x - width / 2 - 6, H - y - 10, width + 12, 21, 4, fill=1, stroke=0)
        c.setFillColor(hexcolor("#40516A"))
        c.setFont(BOLD, size)
        c.drawCentredString(x, H - y - 4, text)

    def draw_edge(self, edge):
        a, b, start, end, via, label, at = edge
        points = [self.nodes[a].port(start), *via, self.nodes[b].port(end)]
        points = [p for i, p in enumerate(points) if i == 0 or p != points[i - 1]]
        c = self.c
        c.setStrokeColor(hexcolor("#62748B"))
        c.setLineWidth(2)
        path = c.beginPath()
        path.moveTo(points[0][0], H - points[0][1])
        for x, y in points[1:]:
            assert 12 <= x <= W - 12 and 136 <= y <= 602, (self.slug, points)
            path.lineTo(x, H - y)
        c.drawPath(path)
        x0, y0 = points[-2]
        x1, y1 = points[-1]
        angle = math.atan2(y1 - y0, x1 - x0)
        length, spread = 10, 4.5
        backx, backy = x1 - length * math.cos(angle), y1 - length * math.sin(angle)
        dx, dy = spread * math.sin(angle), -spread * math.cos(angle)
        triangle = c.beginPath()
        triangle.moveTo(x1, H - y1)
        triangle.lineTo(backx + dx, H - (backy + dy))
        triangle.lineTo(backx - dx, H - (backy - dy))
        triangle.close()
        c.setFillColor(hexcolor("#62748B"))
        c.drawPath(triangle, fill=1, stroke=0)
        if label:
            assert at is not None
            self.label(label, *at)

    def draw_node(self, n):
        c = self.c
        fill, border, ink = PALETTE[n.kind]
        c.setFillColor(hexcolor(fill))
        c.setStrokeColor(hexcolor(border))
        c.setLineWidth(1.7 if n.kind == "decision" else 1.2)
        c.roundRect(n.x, H - n.y - n.h, n.w, n.h, 9, fill=1, stroke=1)
        c.setFillColor(hexcolor(ink))
        if n.big:
            first, *rest = n.label.split("\n")
            size = 46
            assert pdfmetrics.stringWidth(first, BOLD, size) <= n.w - 26
            body = wrap("\n".join(rest), n.w - 32, FONT, 16)
            height = 52 + len(body) * 21
            assert height <= n.h - 24, (n.key, height, n.h)
            y = n.y + (n.h - height) / 2 + 43
            c.setFont(BOLD, size)
            c.drawCentredString(n.x + n.w / 2, H - y, first)
            c.setFont(FONT, 16)
            for i, line in enumerate(body):
                c.drawCentredString(n.x + n.w / 2, H - y - 28 - i * 21, line)
            self.font_sizes.append(16)
            return
        size = 17.0
        while size >= 12:
            lines = wrap(n.label, n.w - 28, BOLD, size)
            leading = size * 1.25
            if len(lines) * leading <= n.h - 20:
                break
            size -= 0.25
        assert size >= 12, (self.slug, n.key, n.label)
        self.font_sizes.append(size)
        y = n.y + (n.h - len(lines) * leading) / 2 + size
        for i, line in enumerate(lines):
            c.setFont(BOLD if i == 0 else FONT, size)
            c.drawCentredString(n.x + n.w / 2, H - y - i * leading, line)

    def render(self):
        c = self.c
        c.setFillColor(hexcolor("#FFFFFF"))
        c.rect(0, 0, W, H, fill=1, stroke=0)
        c.setFillColor(hexcolor("#0B1E3D"))
        c.rect(0, H - 132, W, 132, fill=1, stroke=0)
        c.setFillColor(hexcolor("#FFFFFF"))
        c.setFont(BOLD, 18)
        c.drawString(56, H - 36, "Veyra")
        c.setFont(MONO, 10.5)
        c.setFillColor(hexcolor("#A8C9FA"))
        c.drawString(132, H - 35, "SOLUTION ARCHITECTURE")
        c.drawRightString(1224, H - 35, f"{self.number:02d} / {TOTAL}")
        c.setFillColor(hexcolor("#FFFFFF"))
        c.setFont(BOLD, 31)
        assert pdfmetrics.stringWidth(self.title, BOLD, 31) <= 1168
        c.drawString(56, H - 80, self.title)
        c.setFillColor(hexcolor("#CBDBEF"))
        c.setFont(FONT, 14.5)
        assert pdfmetrics.stringWidth(self.subtitle, FONT, 14.5) <= 1168
        c.drawString(56, H - 110, self.subtitle)
        values = list(self.nodes.values())
        for i, n in enumerate(values):
            assert 40 <= n.x and n.x + n.w <= 1240
            assert 158 <= n.y and n.y + n.h <= 600
            for m in values[:i]:
                overlap = n.x < m.x + m.w and m.x < n.x + n.w and n.y < m.y + m.h and m.y < n.y + n.h
                assert not overlap, (self.slug, n.key, m.key)
        for edge in self.edges:
            self.draw_edge(edge)
        for n in values:
            self.draw_node(n)
        y = 618
        c.setFillColor(hexcolor("#384A61"))
        c.setFont(FONT, 13)
        for note in self.notes:
            lines = wrap(note, 1168, FONT, 13)
            for line in lines:
                c.drawString(56, H - y, line)
                y += 17
            y += 3
        assert y <= 682, (self.slug, y)
        c.setStrokeColor(hexcolor("#DCE3ED"))
        c.setLineWidth(0.6)
        c.line(56, 34, 1224, 34)
        c.setFillColor(hexcolor("#66768A"))
        c.setFont(FONT, 9.5)
        left = "Selected offline TF-IDF release | Native LinuxONE s390x | Recorded implementation"
        c.drawString(56, 19, left)
        c.setFont(MONO, 9)
        c.drawRightString(1224, 19, "veyra-ibmz.vercel.app")
        c.linkURL("https://veyra-ibmz.vercel.app/resources/Veyra-Architecture-and-Flowcharts.txt", (56, 11, 1224, 32), relative=0)
        c.bookmarkPage(self.slug)
        c.addOutlineEntry(f"{self.number:02d}. {self.title}", self.slug, level=0)
        c.showPage()
        return {"number": self.number, "slug": self.slug, "title": self.title,
                "node_count": len(values), "minimum_node_font_pt": min(self.font_sizes)}


def diagrams(c):
    pages = []
    def page(slug, title, subtitle, notes):
        d = Diagram(c, len(pages) + 1, slug, title, subtitle, notes)
        pages.append(d)
        return d

    d = page("01-system-overview", "System overview", "One request path, one observation path, and an approved response gate.", [
        "The guardrail decides. The canary observes. Both requests and probes use the active backend.",
        "The supervisor returns a decision and an execution-permission flag. The selected replay records the flag; it executes no customer action."])
    d.node("input", 56, 180, 224, 76, "Ordered requests\nFrozen policy and artifacts", "record")
    d.node("supervisor", 350, 180, 260, 76, "SupervisedRuntime\nService-state checks")
    d.node("backend", 680, 180, 240, 76, "Active backend\nGuardrail + canary")
    d.node("policy", 990, 180, 240, 76, "Guardrail policy only\nDecision + permission", "decision")
    d.node("probes", 56, 365, 224, 76, "Fixed trusted probes\nRecorded paired targets", "record")
    d.node("watch", 350, 365, 260, 76, "RuntimeWatch\nCompare paired outputs")
    d.node("recovery", 680, 365, 240, 76, "RecoveryManager\nExecute and verify")
    d.node("activate", 990, 365, 240, 76, "Approved activation\nFresh verification", "verified")
    for a,b in [("input","supervisor"),("supervisor","backend"),("backend","policy"),("probes","watch"),("watch","recovery"),("recovery","activate")]: d.edge(a,b)
    d.edge("backend","watch","bottom","top",[(800,305),(480,305)],"Probe outputs",(645,305))
    d.edge("activate","backend","right","top",[(1250,403),(1250,150),(800,150)])

    d = page("02-request-flow", "Request decision flow", "Only the guardrail policy can return allow or refuse.", [
        "A hold, unavailable service, invalid outputs, or refusal blocks execution. These outcomes retain their separate decision types.",
        "An allow result permits a caller action; it does not prove that the request is universally safe."])
    d.node("input",56,180,220,76,"Next ordered request","record")
    d.node("state",340,180,260,76,"Service held\nor unavailable?","decision")
    d.node("pair",670,180,240,76,"Get paired outputs\nActive guardrail + canary")
    d.node("valid",980,180,250,76,"Both outputs and\nmodel identity valid?","decision")
    d.node("held",340,360,260,76,"No execution\nHold or unavailable","blocked")
    d.node("policy",670,360,240,76,"Apply frozen policy\nGuardrail score only","decision")
    d.node("invalid",980,360,250,76,"Invalid outputs\nOpen incident; contain","error")
    d.node("refuse",340,500,260,76,"REFUSE\nexecute = false","blocked")
    d.node("allow",670,500,240,76,"ALLOW\nexecute = true","verified")
    d.edge("input","state")
    d.edge("state","pair",label="NO",at=(635,218))
    d.edge("state","held","bottom","top",label="YES",at=(510,307))
    d.edge("pair","valid")
    d.edge("valid","invalid","bottom","top",label="NO",at=(1147,307))
    d.edge("valid","policy","bottom","top",[(1105,300),(790,300)],"YES",(955,300))
    d.edge("policy","allow","bottom","top",label="ALLOW",at=(834,468))
    d.edge("policy","refuse","left","right",[(630,398),(630,538)],"REFUSE",(630,463))

    d = page("03-evidence-channels", "Three evidence channels", "Traffic change, runtime change, and classifier quality answer different questions.", [
        "A traffic warning cannot authorise a replacement. The selected response loop uses RuntimeWatch, not a traffic KS alarm.",
        "Known evaluation labels remain outside live request decisions. The canary does not override the guardrail policy."])
    columns = [(56,"traffic"),(456,"runtime"),(856,"quality")]
    labels = {
        "traffic":["TRAFFIC OBSERVATION\nOrdinary paired score gaps","Legacy empirical KS monitor\nSeparate study path","Warning only\nNo replacement authority"],
        "runtime":["RUNTIME INTEGRITY\nFixed probes + trusted targets","RuntimeWatch\nCompare both score vectors","Runtime incident evidence\nResponse still needs approval"],
        "quality":["CLASSIFIER QUALITY\nRecorded decisions + labels","Independent offline audit\nErrors, holds, and costs","Quality report\nNot a live decision input"]}
    for x,key in columns:
        for i,y in enumerate([178,343,508]):d.node(f"{key}{i}",x,y,368,76,labels[key][i],"blocked" if key=="traffic" and i==2 else "record" if key=="quality" else "step")
        d.edge(key+"0",key+"1","bottom","top")
        d.edge(key+"1",key+"2","bottom","top")

    d = page("04-runtime-health-check", "Scheduled runtime health check", "Eight fixed probes check the active pair every 16 request positions.", [
        "A valid changed score schedules one response opportunity after 32 positions. Unusable outputs require immediate containment.",
        "A matching small panel can close a pending incident. It cannot release a hold or establish universal recovery."])
    d.node("position",56,180,280,76,"Next request position","record")
    d.node("due",456,180,368,76,"Health check due?\nEvery 16 positions","decision")
    d.node("continue",936,180,288,76,"Keep existing service state\nContinue the request flow","record")
    d.node("check",456,327,368,86,"Run 8 trusted probes\nCompare both score vectors\nTolerance: 1e-8","decision")
    d.node("match",56,510,280,76,"Record panel agreement\nNot universal recovery","record")
    d.node("incident",456,510,368,76,"Open runtime incident\nResponse due in 32 positions")
    d.node("error",936,510,288,76,"No usable outputs\nContain immediately","error")
    d.edge("position","due")
    d.edge("due","continue",label="NO",at=(882,218))
    d.edge("due","check","bottom","top",label="YES",at=(680,291))
    d.edge("check","match","bottom","top",[(640,462),(196,462)],"MATCH",(350,462))
    d.edge("check","incident","bottom","top",label="CHANGE",at=(690,467))
    d.edge("check","error","bottom","top",[(640,462),(1080,462)],"ERROR",(935,462))

    d = page("05-candidate-proposal", "Execute and verify a candidate", "A proposed score correction is not a verified replacement.", [
        "If fresh paired outputs already match, report no change and activate nothing. Missing outputs use the trusted-reload path.",
        "The affine rule adjusts guardrail scores, not model weights or thresholds. Canary outputs must also pass verification."])
    d.node("due",56,180,250,80,"Response is due\nRefresh probe observations")
    d.node("approved",386,180,280,80,"Recovery approval\npresent?","decision")
    d.node("fit",746,180,300,80,"Fit guardrail score correction\n64 trusted fit probes")
    d.node("verify",746,350,300,80,"Execute actual candidate\nVerify on 128 separate probes","decision")
    d.node("verified",56,350,250,80,"Verified candidate\nCreate activation plan","verified")
    d.node("no",386,500,280,76,"No activation\nApply containment rule","blocked")
    d.node("reload",746,500,300,76,"Try trusted backend reload\nExecute and verify again","decision")
    d.node("fail",1110,500,120,76,"FAIL\nContain","blocked")
    d.edge("due","approved")
    d.edge("approved","fit",label="YES",at=(706,220))
    d.edge("approved","no","bottom","top",label="NO",at=(565,390))
    d.edge("fit","verify","bottom","top")
    d.edge("fit","reload","right","top",[(1080,220),(1080,465),(896,465)],"UNSUPPORTED",(1144,300))
    d.edge("verify","verified","left","right",label="PASS",at=(526,350))
    d.edge("verify","reload","bottom","top",label="FAIL",at=(938,465))
    d.edge("reload","verified","left","right",[(700,538),(700,390)],"PASS",(610,390))
    d.edge("reload","fail",label="FAIL",at=(1077,538))

    d = page("06-activation-gate", "Fresh verification before activation", "A manager-owned, single-use plan cannot self-certify its candidate.", [
        "The manager uses its retained candidate and probe inputs, not a caller-supplied report or a cached result.",
        "Activation does not automatically release an existing hold. Separate release approval and full verification still apply."])
    d.node("plan",56,190,260,80,"Approved, verified candidate\nManager-owned single-use plan","record")
    d.node("identity",400,190,300,80,"Plan identities match?\nIssued here and unused?","decision")
    d.node("execute",820,190,400,80,"Execute retained candidate again\n128 verification inputs")
    d.node("verify",820,360,400,80,"Both score vectors, decisions,\nand required class checks pass?","decision")
    d.node("reject",400,510,300,76,"Reject activation\nCandidate does not become active","blocked")
    d.node("activate",820,510,400,76,"Activate verified candidate\nAdvance generation; consume plan","verified")
    d.edge("plan","identity")
    d.edge("identity","execute",label="YES",at=(760,230))
    d.edge("identity","reject","bottom","top",label="NO",at=(595,390))
    d.edge("execute","verify","bottom","top")
    d.edge("verify","activate","bottom","top",label="PASS",at=(1062,478))
    d.edge("verify","reject","left","top",[(760,400),(760,470),(550,470)],"FAIL",(700,470))

    d = page("07-missing-outputs", "Recovery without usable current outputs", "No fabricated scores enter the correction or verification process.", [
        "Missing current observations skip the affine-fit path. A trusted reload must match archived paired targets through actual execution.",
        "A verified replacement still needs separate release if the service already enters a hold or unavailable state."])
    d.node("missing",56,190,260,80,"No usable model outputs\nDo not invent observations","error")
    d.node("contain",400,190,300,80,"Contain immediately\nNo current request execution","blocked")
    d.node("approval",820,190,400,80,"At the due response position:\nrecovery approval present?","decision")
    d.node("reload",820,350,400,80,"Create a trusted replacement\nNo affine fit from missing samples")
    d.node("verify",400,350,300,80,"Execute and compare with\narchived paired targets","decision")
    d.node("activate",56,350,260,80,"Use activation gate\nFresh paired verification","verified")
    d.node("release",56,510,260,76,"Use separate release process\nExisting hold remains until release","blocked")
    d.node("remain",400,510,300,76,"Verification fails\nRemain held or unavailable","blocked")
    d.node("no",820,510,400,76,"No recovery approval\nNo replacement activation","blocked")
    d.edge("missing","contain")
    d.edge("contain","approval")
    d.edge("approval","reload","bottom","top",label="YES",at=(1066,310))
    d.edge("approval","no","right","right",[(1250,230),(1250,548)],"NO",(1250,405))
    d.edge("reload","verify","left","right")
    d.edge("verify","activate","left","right",label="PASS",at=(358,390))
    d.edge("verify","remain","bottom","top",label="FAIL",at=(595,475))
    d.edge("activate","release","bottom","top")

    d = page("08-containment-authority", "Containment is a separate authority", "Neither hold nor unavailable service means a successful automatic decision.", [
        "Containment approval is separate from recovery approval. Invalid outputs or an unresolved response follow this rule.",
        "Neither outcome executes a request. Both receive zero automatic-correctness credit."])
    d.node("reason",440,180,400,76,"Invalid outputs or\na response cannot resolve the incident","record")
    d.node("approval",440,335,400,80,"Containment approval\npresent?","decision")
    d.node("hold",56,510,360,76,"HOLD\nApproved containment; no execution","blocked")
    d.node("unavailable",864,510,360,76,"UNAVAILABLE\nNo approved service path; no execution","blocked")
    d.edge("reason","approval","bottom","top")
    d.edge("approval","hold","bottom","top",[(640,460),(236,460)],"YES",(430,460))
    d.edge("approval","unavailable","bottom","top",[(640,460),(1044,460)],"NO",(856,460))

    d = page("09-hold-release", "Full verification before service release", "A small-panel match or a new backend alone cannot release a hold.", [
        "Separate release approval, a valid actor ID, and full paired verification control the release.",
        "A later fault can open a new incident. The default watch limit is 128 incidents; unbounded-process use is not qualified."])
    d.node("state",56,190,260,80,"Service held\nor unavailable","blocked")
    d.node("approval",400,190,300,80,"Separate release approval\nand valid actor ID?","decision")
    d.node("verify",820,190,400,80,"Execute all 128 verification probes\nBoth vectors and policy checks pass?","decision")
    d.node("clear",820,355,400,80,"Clear hold or unavailable state\nClose after full verification","verified")
    d.node("rearm",400,355,300,80,"Rearm RuntimeWatch\nLater incidents remain detectable","verified")
    d.node("resume",56,355,260,80,"Resume request flow\nGuardrail policy still applies","record")
    d.node("keep",400,510,300,76,"Reject release\nKeep the existing service state","blocked")
    d.edge("state","approval")
    d.edge("approval","verify",label="YES",at=(760,230))
    d.edge("approval","keep","bottom","top",[(730,270),(730,480),(550,480)],"NO",(730,465))
    d.edge("verify","clear","bottom","top",label="PASS",at=(1070,310))
    d.edge("verify","keep","left","top",[(770,230),(770,470),(550,470)],"FAIL",(770,451))
    d.edge("clear","rearm","left","right")
    d.edge("rearm","resume","left","right")

    d = page("10-data-model-lifecycle", "Data and model lifecycle", "Related families stay together before synthetic expansion.", [
        "The catalogue contains 100 template families and 91 reviewed semantic groups. Variants are not independent human requests.",
        "The selected model does not use the reserved calibration fallback. Test labels do not select its weights or thresholds."])
    d.node("catalogue",56,180,260,76,"Project synthetic catalogue\nIntent and family definitions","record")
    d.node("generate",400,180,300,76,"Fix grouped split roles\nBefore synthetic expansion")
    d.node("split",820,180,400,76,"Generate natively on LinuxONE\n100,000 rows in fixed partitions","decision")
    d.node("fit",56,350,260,80,"70,000 fit rows\nFit guardrail and canary")
    d.node("validation",400,350,300,80,"10,000 validation rows\nThresholds and separate probe roles")
    d.node("test",820,350,400,80,"20,000 held-out test rows\nEvaluate only after the freeze","record")
    d.node("freeze",400,515,300,76,"Freeze models and policy\nRecord identities and file hashes","verified")
    d.edge("catalogue","generate")
    d.edge("generate","split")
    d.edge("split","fit","bottom","top",[(1020,303),(186,303)])
    d.edge("split","validation","bottom","top",[(1020,303),(550,303)])
    d.edge("split","test","bottom","top")
    d.edge("fit","freeze","bottom","top",[(186,480),(550,480)])
    d.edge("validation","freeze","bottom","top")
    d.edge("freeze","test","right","left",[(770,553),(770,390)],"After freeze",(770,478))

    d = page("11-probe-roles", "Probe sets have separate roles", "The eight health probes are a subset of the 64 fit probes.", [
        "The fit and verification sets have disjoint request identities. Both remain separate from policy calibration identities.",
        "All trusted reference outputs stay fixed. Current traffic does not silently replace the baseline."])
    d.node("fitrows",56,180,500,76,"1,000 fit-role validation rows","record")
    d.node("verifyrows",724,180,500,76,"Separate 1,000 verification-role rows","record")
    d.node("fit64",56,335,500,80,"Select 64 fit probes\n32 per class")
    d.node("verify128",724,335,500,80,"Select 128 verification probes\n64 per class")
    d.node("health8",56,510,230,76,"Select 8 for health checks\nFrequent paired checks")
    d.node("correction",326,510,230,76,"Use all 64 for correction fit\nNo verification overlap")
    d.node("full",724,510,500,76,"Verify actual candidates\nRepeat for activation and release","verified")
    d.edge("fitrows","fit64","bottom","top")
    d.edge("verifyrows","verify128","bottom","top")
    d.edge("fit64","health8","bottom","top",[(306,465),(171,465)])
    d.edge("fit64","correction","bottom","top",[(306,465),(441,465)])
    d.edge("verify128","full","bottom","top")

    d = page("12-paired-evaluation", "Fair paired evaluation", "60 paired episodes use the same requests, policy, faults, and input order.", [
        "The full-stream result includes detection and response delay. Holds, review, and unavailable service receive zero correctness credit.",
        "The guardrail-only arm makes no canary calls. The monitored arm includes its actual model and probe cost."])
    d.node("stream",390,164,500,76,"Same frozen request stream\nSame policy and fault schedule","record")
    d.node("a",56,310,500,80,"A: guardrail only\nNo canary inference")
    d.node("b",724,310,500,80,"B: guardrail + canary + probes\nVerified response or containment")
    d.node("adecision",56,450,500,66,"All A decisions","record")
    d.node("bdecision",724,450,500,66,"All B decisions, holds, and unavailable outcomes","record")
    d.node("audit",390,550,500,48,"Independent audit: all 30,720 observations","verified")
    d.edge("stream","a","bottom","top",[(640,275),(306,275)])
    d.edge("stream","b","bottom","top",[(640,275),(974,275)])
    d.edge("a","adecision","bottom","top")
    d.edge("b","bdecision","bottom","top")
    d.edge("adecision","audit","bottom","left",[(306,574)])
    d.edge("bdecision","audit","bottom","right",[(974,574)])

    d = page("13-evidence-boundary", "Public and private evidence boundaries", "Public evidence is separate from raw study inputs and private model artifacts.", [
        "The static website and dashboard show recorded evidence. They do not provide live telemetry or runtime control.",
        "Private archives can exist outside the execution host. This diagram does not claim hardware-enforced data residency."])
    d.node("native",390,165,500,76,"Native experiment + independent audit","record")
    d.node("private",56,320,500,106,"PRIVATE\nRaw requests, model binaries, and vocabularies","record")
    d.node("public",724,320,500,106,"PUBLIC\nSource, aggregate metrics, numeric charts,\nand source/licence records")
    d.node("archives",56,510,500,76,"Separate private archives\nOutside public Git","record")
    d.node("site",724,510,500,76,"Public Git + static evidence pages\nNo live control path")
    d.edge("native","private","bottom","top",[(640,280),(306,280)])
    d.edge("native","public","bottom","top",[(640,280),(974,280)])
    d.edge("private","archives","bottom","top")
    d.edge("public","site","bottom","top")

    d = page("14-recorded-timeline", "Recorded scale-fault timeline", "A frozen seed-0 example, not a live inference demonstration.", [
        "Request positions use zero-based indices. The delays belong to the fixed experimental schedule, not a production latency guarantee.",
        "The final endpoint check is separate from the complete-stream result. A recovered endpoint does not erase earlier exposure."])
    d.node("fault",56,260,250,194,"128\nFault begins\nInjected scale change","record",True)
    d.node("confirm",370,260,250,194,"143\nRuntime confirms\nFixed paired probes","step",True)
    d.node("activate",684,260,250,194,"175\nVerified activation\nApproved correction","verified",True)
    d.node("endpoint",998,260,226,194,"1,000\nEndpoint requests\n0 decision mismatches","record",True)
    d.edge("fault","confirm",label="15 positions",at=(338,230))
    d.edge("confirm","activate",label="32 positions",at=(652,230))
    d.edge("activate","endpoint",label="Final check",at=(966,230))

    d = page("15-score-change-example", "A valid score can hide a changed path", "Recorded request index 135: same request, same weights, and same threshold.", [
        "The source labels this synthetic study request harmful. The displayed scores and threshold use rounded values.",
        "Scores are not calibrated harm probabilities. Restoration returns the tested baseline, not universal semantic safety."])
    d.node("fixed",390,165,500,76,"Fixed request + model weights + policy threshold","record")
    d.node("baseline",56,310,500,142,"0.7426\nApproved baseline score","step",True)
    d.node("fault",724,310,500,142,"0.1856\nAfter a score-scale fault of x0.25","blocked",True)
    d.node("refuse",56,515,500,76,"REFUSE\nAbove the fixed threshold 0.2083","record")
    d.node("allow",724,515,500,76,"ALLOW\nBelow the fixed threshold 0.2083","blocked")
    d.edge("fixed","baseline","bottom","top",[(640,280),(306,280)])
    d.edge("fixed","fault","bottom","top",[(640,280),(974,280)])
    d.edge("baseline","refuse","bottom","top")
    d.edge("fault","allow","bottom","top")

    d = page("16-cli-entry-points", "Three CLI entry points", "The classifier command alone does not start the runtime response loop.", [
        "Use the complete paired replay to exercise the monitor, recovery, and containment path.",
        "The safety-watch command names remain compatibility aliases. Native model execution uses the pinned LinuxONE environment."])
    data = [
        (56,"inspect",["veyra inspect","Verify 4 model-file hashes\nCheck bound metadata","Report only\nDo not load model objects"]),
        (456,"score",["veyra score","Verify and load the frozen pair\nReturn scores and classification","Classifier only\nNo monitor or action execution"]),
        (856,"replay",["veyra-replay","Run the complete offline paired loop\nProbes, response, and containment","Record incidents and metrics\nNot a live production service"]),
    ]
    for x,key,labels in data:
        for i,y in enumerate([178,343,508]):d.node(key+str(i),x,y,368,76,labels[i],"blocked" if key=="score" and i==2 else "record" if i==2 else "step")
        d.edge(key+"0",key+"1","bottom","top")
        d.edge(key+"1",key+"2","bottom","top")

    d = page("17-explicit-rollback", "Explicit rollback is not recovery", "Rollback is a separate RecoveryManager API operation.", [
        "The previous backend can still be degraded. Rollback does not establish recovery or release an existing hold.",
        "Hold release needs its own approval and complete paired verification."])
    d.node("request",56,190,260,80,"Explicit rollback request","record")
    d.node("approved",400,190,300,80,"Rollback approval\npresent?","decision")
    d.node("previous",820,190,400,80,"Previous backend\navailable?","decision")
    d.node("restore",820,355,400,80,"Restore previous backend")
    d.node("generation",400,355,300,80,"Advance generation\nRecord rollback receipt")
    d.node("unresolved",56,355,260,80,"incident_resolved = false\nNo recovery claim","blocked")
    d.node("hold",56,510,260,76,"No automatic hold release\nBackend can remain degraded","blocked")
    d.node("reject",400,510,300,76,"Reject rollback\nNo backend change","blocked")
    d.edge("request","approved")
    d.edge("approved","previous",label="YES",at=(760,230))
    d.edge("approved","reject","bottom","top",[(730,270),(730,480),(550,480)],"NO",(730,460))
    d.edge("previous","restore","bottom","top",label="YES",at=(1065,310))
    d.edge("previous","reject","left","top",[(770,230),(770,470),(550,470)],"NO",(770,448))
    d.edge("restore","generation","left","right")
    d.edge("generation","unresolved","left","right")
    d.edge("unresolved","hold","bottom","top")
    assert len(pages) == 17
    return pages


def main():
    PUBLIC.mkdir(parents=True, exist_ok=True)
    IMAGES.mkdir(parents=True, exist_ok=True)
    for name,path in [(FONT,"IBMPlexSans-Regular.ttf"),(BOLD,"IBMPlexSans-SemiBold.ttf"),(MONO,"IBMPlexMono-Regular.ttf")]:
        pdfmetrics.registerFont(TTFont(name, str(FONT_DIR / path)))
    selected = json.loads((ROOT / "src/release/selected.json").read_text())
    demos = json.loads((ROOT / "presentation/evidence/demo-cases.json").read_text())
    assert demos["scale"]["causal_detection"][0] == {"confirmation_delay":15,"confirmed_index":143,"fault_index":128,"response_index":175}
    assert selected["guardrail_threshold"] == 0.20832091569900513
    c = canvas.Canvas(str(PDF), pagesize=(W,H), pageCompression=1, invariant=1)
    c.setTitle("Veyra - Solution Architecture and Flowcharts")
    c.setAuthor("Veyra project")
    c.setSubject("Four key diagrams for the selected offline guardrail integrity system")
    catalogue = diagrams(c)
    selected_diagrams = [catalogue[i] for i in (0, 1, 5, 9)]
    metadata = []
    for number, diagram in enumerate(selected_diagrams, 1):
        diagram.number = number
        diagram.slug = f"{number:02d}-" + diagram.slug.split("-", 1)[1]
        metadata.append(diagram.render())
    c.save()
    reader = PdfReader(PDF)
    assert len(reader.pages) == TOTAL
    for i,page in enumerate(reader.pages,1):
        assert page.extract_text().strip(), i
    subprocess.run(["pdftoppm","-r","150","-png",str(PDF),str(IMAGES / "diagram")],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
    generated = sorted(IMAGES.glob("diagram-*.png"))
    assert len(generated) == TOTAL
    for source,row in zip(generated,metadata):
        target=IMAGES/(row["slug"]+".png")
        source.replace(target)
        with Image.open(target) as image:
            assert image.width>=2600 and image.height>=1400
            row.update(file=target.name,width=image.width,height=image.height,sha256=sha(target))
    manifest={"product":"Veyra","pdf":PDF.name,"pdf_sha256":sha(PDF),"pages":TOTAL,
        "release_id":selected["release_id"],"scope":"Architecture documentation only. No model fit, inference, or qualification occurs.",
        "source_text":"https://veyra-ibmz.vercel.app/resources/Veyra-Architecture-and-Flowcharts.txt",
        "diagrams":metadata}
    (IMAGES/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    with zipfile.ZipFile(ZIP,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=8) as z:
        for row in metadata:z.write(IMAGES/row["file"],"Veyra-Diagram-Images/"+row["file"])
        z.write(IMAGES/"manifest.json","Veyra-Diagram-Images/manifest.json")
    with zipfile.ZipFile(ZIP) as z:assert z.testzip() is None
    preview=Image.new("RGB",(2000,300*((TOTAL+3)//4)),"#E4E9EF")
    draw=ImageDraw.Draw(preview)
    for i,row in enumerate(metadata):
        with Image.open(IMAGES/row["file"]) as image:
            image.thumbnail((480,270))
            x,y=(i%4)*500+10,(i//4)*300+8
            preview.paste(image,(x,y))
            draw.text((x,y+276),row["slug"],fill="#102A43")
    private=ROOT.parent/"artifacts/architecture-images-20261018"
    private.mkdir(parents=True,exist_ok=True)
    preview.save(private/"diagram-contact.png")
    (private/"build-receipt.json").write_text(json.dumps({"status":"generated","pdf_bytes":PDF.stat().st_size,"pdf_sha256":sha(PDF),"image_count":TOTAL,"image_zip_bytes":ZIP.stat().st_size,"image_zip_sha256":sha(ZIP),"minimum_node_font_pt":min(r["minimum_node_font_pt"] for r in metadata)},indent=2)+"\n")
    print(json.dumps({"pdf":str(PDF),"pages":TOTAL,"images":TOTAL,"image_zip":str(ZIP),"pdf_bytes":PDF.stat().st_size},indent=2))


if __name__ == "__main__":
    main()
