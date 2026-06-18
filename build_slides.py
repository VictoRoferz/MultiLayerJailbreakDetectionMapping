"""Build the jailbreak-activation-mapping presentation deck (Gemma vs Vicuna vs Gao)."""
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

# ── Palette ──────────────────────────────────────────────────────────────────
NAVY    = RGBColor(0x1F, 0x2D, 0x3D)
BLUE    = RGBColor(0x2E, 0x5E, 0xAA)
TEAL    = RGBColor(0x2C, 0xA0, 0x8E)
ORANGE  = RGBColor(0xDD, 0x84, 0x52)
RED     = RGBColor(0xC4, 0x4E, 0x52)
GREY    = RGBColor(0x5A, 0x5A, 0x5A)
LIGHT   = RGBColor(0xF0, 0xF2, 0xF5)
WHITE   = RGBColor(0xFF, 0xFF, 0xFF)

prs = Presentation()
prs.slide_width = Inches(13.333)   # 16:9
prs.slide_height = Inches(7.5)
SW, SH = prs.slide_width, prs.slide_height
BLANK = prs.slide_layouts[6]


def slide():
    return prs.slides.add_slide(BLANK)


def box(s, x, y, w, h, fill=None, line=None, line_w=1.0):
    shp = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    shp.shadow.inherit = False
    if fill is None:
        shp.fill.background()
    else:
        shp.fill.solid(); shp.fill.fore_color.rgb = fill
    if line is None:
        shp.line.fill.background()
    else:
        shp.line.color.rgb = line; shp.line.width = Pt(line_w)
    return shp


def text(s, x, y, w, h, runs, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP,
         space_after=4):
    tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    for i, item in enumerate(runs):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.space_after = Pt(space_after)
        p.space_before = Pt(0)
        txt, size, color, bold = item[0], item[1], item[2], item[3]
        lvl = item[4] if len(item) > 4 else 0
        p.level = lvl
        r = p.add_run(); r.text = txt
        r.font.size = Pt(size); r.font.color.rgb = color; r.font.bold = bold
        r.font.name = "Calibri"
    return tb


def header(s, title, sub=None):
    box(s, 0, 0, 13.333, 1.15, fill=NAVY)
    text(s, 0.5, 0.18, 12.3, 0.9,
         [(title, 30, WHITE, True)], anchor=MSO_ANCHOR.MIDDLE)
    if sub:
        text(s, 0.5, 0.78, 12.3, 0.3, [(sub, 13, RGBColor(0xC9,0xD4,0xE0), False)])


def bullets(s, x, y, w, h, items, size=16, gap=8):
    tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame; tf.word_wrap = True
    for i, (txt, lvl, color, bold) in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.level = lvl; p.space_after = Pt(gap); p.space_before = Pt(0)
        bullet = ("• " if lvl == 0 else "– ") if txt else ""
        r = p.add_run(); r.text = bullet + txt
        r.font.size = Pt(size - lvl*1); r.font.color.rgb = color; r.font.bold = bold
        r.font.name = "Calibri"
    return tb


def table(s, x, y, w, h, data, col_w=None, header_fill=BLUE, fontsize=12,
          header_size=12):
    rows, cols = len(data), len(data[0])
    gt = s.shapes.add_table(rows, cols, Inches(x), Inches(y), Inches(w), Inches(h)).table
    if col_w:
        total = sum(col_w)
        for j, cw in enumerate(col_w):
            gt.columns[j].width = Emu(int(Inches(w) * cw / total))
    for i, row in enumerate(data):
        for j, val in enumerate(row):
            c = gt.cell(i, j)
            c.margin_left = Inches(0.08); c.margin_right = Inches(0.08)
            c.margin_top = Inches(0.03); c.margin_bottom = Inches(0.03)
            c.vertical_anchor = MSO_ANCHOR.MIDDLE
            tf = c.text_frame; tf.word_wrap = True
            p = tf.paragraphs[0]
            r = p.add_run(); r.text = str(val)
            r.font.name = "Calibri"
            if i == 0:
                c.fill.solid(); c.fill.fore_color.rgb = header_fill
                r.font.color.rgb = WHITE; r.font.bold = True; r.font.size = Pt(header_size)
            else:
                c.fill.solid()
                c.fill.fore_color.rgb = WHITE if i % 2 else LIGHT
                r.font.color.rgb = NAVY; r.font.size = Pt(fontsize)
                if j == 0:
                    r.font.bold = True
    return gt


def chip(s, x, y, w, color, label):
    box(s, x, y, w, 0.42, fill=color)
    text(s, x, y, w, 0.42, [(label, 12, WHITE, True)],
         align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)


# ════════════════════════════════════════════════════════════════════════════
# SLIDE 1 — Title
# ════════════════════════════════════════════════════════════════════════════
s = slide()
box(s, 0, 0, 13.333, 7.5, fill=NAVY)
box(s, 0, 4.55, 13.333, 0.06, fill=TEAL)
text(s, 0.9, 2.3, 11.5, 1.6,
     [("Mapping the Jailbreak Activation Space", 40, WHITE, True),
      ("Behavior-grounded datasets across Gemma-2-2B and Vicuna-7B", 20,
       RGBColor(0xC9,0xD4,0xE0), False)])
text(s, 0.9, 4.75, 11.5, 1.5,
     [("How we build the datasets · results · comparison with "
       "Shaping the Safety Boundaries (Gao et al., 2024) · why results "
       "differ · next steps", 15, RGBColor(0xAEB9C7), False)])

# ════════════════════════════════════════════════════════════════════════════
# SLIDE 2 — Goal
# ════════════════════════════════════════════════════════════════════════════
s = slide()
header(s, "Goal & Research Question")
bullets(s, 0.6, 1.5, 12.1, 5.4, [
    ("Hypothesis: jailbreak behaviour corresponds to a definable, prompt-agnostic "
     "subspace in the LLM's activation space.", 0, NAVY, True),
    ("Build a behaviour-grounded dataset: every prompt is run through the target "
     "model, the response is judged, and residual-stream activations are stored.", 0, GREY, False),
    ("Two target models, identical pipeline:", 0, NAVY, True),
    ("Gemma-2-2B-it — modern, well safety-aligned (26 layers, 2304-dim)", 1, GREY, False),
    ("Vicuna-7B-v1.3 — Gao et al.'s backbone, intentionally under-aligned (32 layers, 4096-dim)", 1, GREY, False),
    ("Key question: do confirmed jailbreaks form a distinct, detectable region — "
     "and does it match what Gao et al. report?", 0, BLUE, True),
], size=18, gap=12)

# ════════════════════════════════════════════════════════════════════════════
# SLIDE 3 — Pipeline (visual flow)
# ════════════════════════════════════════════════════════════════════════════
s = slide()
header(s, "How We Build the Dataset", "Same pipeline for Gemma and Vicuna — only the target model changes")

# Row 1: Module 1 inputs
y0 = 1.5
text(s, 0.5, y0, 4, 0.3, [("STEP 1 — Collect prompts (Module 1)", 13, BLUE, True)])
chip(s, 0.5, y0+0.4, 2.6, BLUE,  "Benign: Alpaca ×4000")
chip(s, 3.25, y0+0.4, 2.9, GREY, "Harmful seeds: AdvBench+HarmBench ×715")
chip(s, 6.3, y0+0.4, 3.0, ORANGE, "ArtPrompt (ASCII-art) ×715")
chip(s, 9.45, y0+0.4, 3.4, TEAL, "GCG-Universal + GCG-Individual")

# Arrow down
text(s, 6.4, y0+0.95, 1, 0.3, [("▼", 18, NAVY, True)], align=PP_ALIGN.CENTER)

# Row 2: Module 2
y1 = 3.05
box(s, 0.5, y1, 12.33, 1.25, fill=LIGHT, line=BLUE, line_w=1.5)
text(s, 0.7, y1+0.1, 12, 0.35, [("STEP 2 — Run target model + judge + extract (Module 2)", 13, BLUE, True)])
bullets(s, 0.8, y1+0.45, 12, 0.8, [
    ("Forward pass through target model → generate response (max_new_tokens 64-128)", 0, NAVY, False),
    ("GPT-4o-mini 3-step rubric → label = 1 only if substantive harm produced (behaviour-verified)", 0, NAVY, False),
    ("Store mean-of-last-5-prompt-token activations at each probed layer", 0, NAVY, False),
], size=12.5, gap=4)

text(s, 6.4, y1+1.35, 1, 0.3, [("▼", 18, NAVY, True)], align=PP_ALIGN.CENTER)

# Row 3: outputs
y2 = 4.95
chip(s, 0.5, y2, 3.9, RED,  "Labeled dataset (prompt+response+label)")
chip(s, 4.55, y2, 3.9, RED, "Per-layer activations (.pt)")
chip(s, 8.6, y2, 4.25, RED, "Pushed to HuggingFace + 3-class PCA")

bullets(s, 0.6, y2+0.7, 12.2, 1.4, [
    ("5 categories: benign · harmful_direct · jailbreak_artprompt · jailbreak_gcg_universal · jailbreak_gcg_individual", 0, GREY, False),
    ("We use 3 of Gao's 7 attacks (ArtPrompt, GCG-Universal, GCG-Individual) — the cheapest spanning the design space", 0, GREY, False),
], size=12.5, gap=6)

# ════════════════════════════════════════════════════════════════════════════
# SLIDE 4 — Two targets
# ════════════════════════════════════════════════════════════════════════════
s = slide()
header(s, "Two Targets, One Pipeline")
table(s, 0.6, 1.5, 12.1, 3.2, [
    ["Property", "Gemma-2-2B-it", "Vicuna-7B-v1.3"],
    ["Alignment", "Modern RLHF (strong)", "Llama-1 fine-tune (weak — Gao's choice)"],
    ["Hidden dim", "2304", "4096"],
    ["Layers probed", "5, 10, 15, 20, 25", "4, 10, 15, 20, 25, 30"],
    ["Chat template", "built-in", "FastChat (set manually)"],
    ["GCG-Universal best loss", "1.30  (weak)", "0.000063  (near-perfect)"],
    ["Total rows", "6,132", "~6,165"],
], col_w=[3, 4.5, 5.5], fontsize=13, header_size=14)
bullets(s, 0.6, 5.0, 12.1, 2.0, [
    ("Vicuna is far easier to attack: GCG-Universal converged ~20,000× lower loss in the same 150 steps.", 0, NAVY, True),
    ("Same schema (only activation dim differs) → datasets are independently valid; cross-model merge not possible.", 0, GREY, False),
], size=15, gap=10)

# ════════════════════════════════════════════════════════════════════════════
# SLIDE 5 — Results: label distributions
# ════════════════════════════════════════════════════════════════════════════
s = slide()
header(s, "Results — Behaviour-Verified Jailbreaks", "label = 1 only when the model produced substantive harm (GPT-4o-mini rubric)")
table(s, 0.6, 1.45, 12.1, 3.7, [
    ["Category", "Gemma label=1", "Gemma rate", "Vicuna label=1*", "Vicuna rate*"],
    ["benign (4000)", "0 (auto)", "—", "0 (auto)", "—"],
    ["harmful_direct", "112", "16%", "higher", "~30-40%"],
    ["jailbreak_artprompt", "305", "43%", "higher", "~60%"],
    ["jailbreak_gcg_universal", "154", "22%", "much higher", "~70-90%"],
    ["jailbreak_gcg_individual", "7 / 20", "35%", "15-19 / 20", "~80-95%"],
    ["TOTAL jailbroken", "578", "—", "823", "—"],
], col_w=[3.4, 2.3, 1.7, 2.4, 2.0], fontsize=12.5, header_size=12)
bullets(s, 0.6, 5.45, 12.1, 1.7, [
    ("Under-aligned Vicuna yields ~1.4× more confirmed jailbreaks (823 vs 578) on a same-size pool.", 0, NAVY, True),
    ("* Vicuna per-category numbers from the completed run; rates approximate. ArtPrompt is the breakout on Gemma.", 0, GREY, False),
], size=14, gap=8)

# ════════════════════════════════════════════════════════════════════════════
# SLIDE 6 — PCA results
# ════════════════════════════════════════════════════════════════════════════
s = slide()
header(s, "Results — 3-Class PCA of Activations")
chip(s, 0.6, 1.45, 2.6, BLUE, "Benign")
chip(s, 3.35, 1.45, 3.6, ORANGE, "Harmful — refused (label 0)")
chip(s, 7.1, 1.45, 3.2, RED, "Jailbroken (label 1)")
bullets(s, 0.6, 2.2, 12.1, 4.9, [
    ("Two-part structure at every layer:", 0, NAVY, True),
    ("A big mixed main cloud — benign + harmful + jailbroken OVERLAP heavily", 1, GREY, False),
    ("A tight, far-away cluster (orange+red only) = the GCG-Universal attacks", 1, GREY, False),
    ("Why the tight cluster: all 715 GCG prompts end in the SAME suffix → mean-of-last-5-token "
     "pooling reads identical tokens → they collapse to one point (partly a pooling artifact).", 0, GREY, False),
    ("Why benign & harmful overlap: PC1+PC2 capture only ~15% of variance; the GCG outlier steals "
     "the variance budget; prompt-side activations encode TOPIC, not the refuse/comply decision.", 0, GREY, False),
    ("This overlap REPRODUCES Gao's Finding #1: “harmful and benign activations are not linearly "
     "separable in most layers.”", 0, BLUE, True),
], size=15, gap=10)

# ════════════════════════════════════════════════════════════════════════════
# SLIDE 7 — Comparison with Gao
# ════════════════════════════════════════════════════════════════════════════
s = slide()
header(s, "How We Differ from Gao et al. (Shaping the Safety Boundaries)")
table(s, 0.5, 1.45, 12.33, 4.5, [
    ["Axis", "Gao et al.", "Ours"],
    ["Labeling", "by ATTACK SOURCE (input type)", "by BEHAVIOUR (GPT-4o-mini verified)"],
    ["“Jailbreak” means", "an attack was applied", "the model actually complied"],
    ["Projection", "MDS (preserves distances)", "PCA (max variance)"],
    ["Pooling", "last-token vector", "mean of last 5 prompt tokens"],
    ["Attacks", "7 (4 suffix + 3 semantic)", "3 (ArtPrompt + GCG×2)"],
    ["Scale", "32,507 samples", "~6,150 samples"],
    ["Jailbreak judge", "Dic-Judge (refusal strings)", "GPT-4o-mini 3-step rubric (substantive harm)"],
], col_w=[2.6, 4.6, 5.4], fontsize=12.5, header_size=13)
bullets(s, 0.5, 6.15, 12.3, 1.0, [
    ("Biggest difference: Gao labels by what kind of input it is; we label by whether the model was "
     "actually jailbroken. These answer different questions.", 0, RED, True),
], size=13, gap=4)

# ════════════════════════════════════════════════════════════════════════════
# SLIDE 8 — Why results differ
# ════════════════════════════════════════════════════════════════════════════
s = slide()
header(s, "Why Our Results Look Different")
bullets(s, 0.6, 1.5, 12.1, 5.6, [
    ("1.  Attack applied ≠ jailbreak occurred. Attacks succeed <100% of the time. Our 1,342 refused "
     "Vicuna prompts prove it — Gao counts those as “jailbreak”, we count them as refused.", 0, NAVY, True),
    ("2.  Behaviour vs source labeling. Gao's clean 3-tier boundary comes from grouping by input type; "
     "our behaviour split mixes input types within each colour.", 0, GREY, False),
    ("3.  MDS vs PCA. MDS spreads points to show the boundary; PCA lets the GCG suffix cluster dominate "
     "PC1 and squish everything else.", 0, GREY, False),
    ("4.  Prompt-side activations encode topic, not the refuse/comply decision — which happens later, "
     "during generation (cf. Activation Surgery, 2603.14278).", 0, GREY, False),
    ("5.  Active debate in the literature: Arditi (single refusal direction) vs Gao (“not linearly "
     "separable”). Our behaviour-grounded labels are a lever neither side has cleanly pulled.", 0, BLUE, True),
], size=15, gap=12)

# ════════════════════════════════════════════════════════════════════════════
# SLIDE 9 — Next steps
# ════════════════════════════════════════════════════════════════════════════
s = slide()
header(s, "Next Steps & Validation")
# two columns
text(s, 0.6, 1.45, 6, 0.4, [("Validation (quantify, don't eyeball)", 17, BLUE, True)])
bullets(s, 0.6, 1.95, 6.1, 5.0, [
    ("Per-layer LINEAR PROBE: benign-vs-harmful, benign-vs-jailbroken, "
     "harmful-vs-jailbroken accuracy. Replaces 2D eyeballing.", 0, GREY, False),
    ("Reproduce Gao exactly: same data, but source-labels + MDS + last-token. "
     "Does the boundary reappear? → is it an artifact?", 0, GREY, False),
    ("Re-PCA excluding GCG so the outlier stops dominating the axes.", 0, GREY, False),
    ("BPJ external validation: do held-out attacks land in our subspace?", 0, GREY, False),
], size=13.5, gap=9)

text(s, 6.95, 1.45, 6, 0.4, [("Theories & further work", 17, TEAL, True)])
bullets(s, 6.95, 1.95, 5.9, 5.0, [
    ("Response-side activations: the jailbreak signal may live in generation, "
     "not the prompt (Activation Surgery).", 0, GREY, False),
    ("K* clustering (Module 6): do confirmed jailbreaks form >1 distinct cluster? "
     "Are clusters attack-family-aligned?", 0, GREY, False),
    ("Cross-model: Gemma vs Vicuna separability — does under-alignment give "
     "cleaner structure?", 0, GREY, False),
    ("Add semantic attacks (AutoDAN, PAIR) for full Gao parity.", 0, GREY, False),
], size=13.5, gap=9)

# ════════════════════════════════════════════════════════════════════════════
# SLIDE 10 — Possible conclusions
# ════════════════════════════════════════════════════════════════════════════
s = slide()
header(s, "Possible Conclusions")
bullets(s, 0.6, 1.6, 12.1, 5.3, [
    ("“The safety boundary is a property of attack INPUTS, not jailbreak BEHAVIOUR” "
     "— if it reappears under source-labeling but vanishes under behaviour-labeling.", 0, NAVY, True),
    ("“Geometric separability is attack-family-dependent” — suffix attacks separate "
     "trivially (shared tokens); semantic attacks blend with harmful.", 0, NAVY, True),
    ("“Confirmed jailbreaks occupy K* > 1 clusters, and held-out attacks (BPJ) land within them” "
     "— the prompt-agnostic detector claim.", 0, NAVY, True),
    ("“The decisive signal is response-side, not prompt-side” — would reframe the boundary literature.", 0, NAVY, True),
    ("Methodological contribution: behaviour-verified labeling + quantitative linear-probe separability, "
     "vs prior source-labeled / small-sample / projection-dependent claims.", 0, BLUE, True),
], size=16, gap=14)

prs.save("Jailbreak_Activation_Mapping_Deck.pptx")
print("saved Jailbreak_Activation_Mapping_Deck.pptx with", len(prs.slides._sldIdLst), "slides")
