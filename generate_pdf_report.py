"""
Script: generate_pdf_report.py
Description: Generates the finalized, verified PDF technical report for the
Multimodal Depression Detection Capstone Project (UE23CS320B).
Strictly adheres to all 24 factual consistency and verification rules.
"""

import os
import sys
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.pdfgen import canvas


class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#555555"))
        
        # Header (pages > 1)
        if self._pageNumber > 1:
            self.drawString(54, 11 * inch - 36, "UE23CS320B Capstone Project — Final Technical Report: Detecting Depression Using Multimodal Vocal & Textual Cues")
            self.setStrokeColor(colors.HexColor("#CCCCCC"))
            self.setLineWidth(0.5)
            self.line(54, 11 * inch - 42, 8.5 * inch - 54, 11 * inch - 42)

        # Footer
        page_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(8.5 * inch - 54, 36, page_text)
        self.drawString(54, 36, "CONFIDENTIAL & PROPRIETARY — ACADEMIC CAPSTONE PROJECT REPORT")
        self.setStrokeColor(colors.HexColor("#CCCCCC"))
        self.setLineWidth(0.5)
        self.line(54, 46, 8.5 * inch - 54, 46)
        self.restoreState()


def build_pdf(filename="Capstone_Project_Information_Report.pdf"):
    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )

    styles = getSampleStyleSheet()

    # Brand Colors
    primary_color = colors.HexColor("#1A365D")   # Deep navy
    secondary_color = colors.HexColor("#2B6CB0") # Steel blue
    dark_neutral = colors.HexColor("#2D3748")    # Charcoal body text
    light_bg = colors.HexColor("#F7FAFC")        # Table row background

    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        textColor=primary_color,
        spaceAfter=4,
        alignment=0
    )

    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10.5,
        leading=14,
        textColor=secondary_color,
        spaceAfter=10
    )

    h1_style = ParagraphStyle(
        'SectionH1',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=15,
        textColor=primary_color,
        spaceBefore=12,
        spaceAfter=5,
        keepWithNext=True
    )

    h2_style = ParagraphStyle(
        'SectionH2',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=13,
        textColor=secondary_color,
        spaceBefore=8,
        spaceAfter=3,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        'BodyDark',
        parent=styles['BodyText'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=11.5,
        textColor=dark_neutral,
        spaceAfter=5
    )

    bullet_style = ParagraphStyle(
        'BulletText',
        parent=body_style,
        leftIndent=12,
        firstLineIndent=-8,
        spaceAfter=3
    )

    table_header_style = ParagraphStyle(
        'TableHeader',
        fontName='Helvetica-Bold',
        fontSize=8,
        leading=10,
        textColor=colors.white,
        alignment=1
    )

    table_body_style = ParagraphStyle(
        'TableBody',
        fontName='Helvetica',
        fontSize=7.5,
        leading=9.5,
        textColor=dark_neutral
    )

    table_body_center = ParagraphStyle(
        'TableBodyCenter',
        parent=table_body_style,
        alignment=1
    )

    table_body_bold = ParagraphStyle(
        'TableBodyBold',
        fontName='Helvetica-Bold',
        fontSize=7.5,
        leading=9.5,
        textColor=primary_color
    )

    story = []

    # Title Block
    story.append(Paragraph("CAPSTONE PROJECT TECHNICAL INFORMATION REPORT", subtitle_style))
    story.append(Paragraph("Detecting Depression Using Multimodal Vocal, Acoustic & Textual Cues", title_style))
    story.append(Paragraph("<b>Course:</b> UE23CS320B — Capstone Project &nbsp;|&nbsp; <b>Project ID:</b> PW26_CBH_04 &nbsp;|&nbsp; <b>Guide:</b> Ms. Chandravva Hebbi", body_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=primary_color, spaceBefore=4, spaceAfter=8))

    # PART 1: Project Information Table
    story.append(Paragraph("Part 1 — Project Title & Basic Information", h1_style))
    p1_data = [
        [Paragraph("<b>Parameter</b>", table_header_style), Paragraph("<b>Verified Project Metadata</b>", table_header_style)],
        [Paragraph("<b>Exact Project Title</b>", table_body_bold), Paragraph("Detecting Depression using vocal cues", table_body_style)],
        [Paragraph("<b>Project ID</b>", table_body_bold), Paragraph("PW26_CBH_04", table_body_style)],
        [Paragraph("<b>Course / Subject</b>", table_body_bold), Paragraph("UE23CS320B – Capstone Project (Phase 2 & Phase 3)", table_body_style)],
        [Paragraph("<b>Project Guide</b>", table_body_bold), Paragraph("Ms. Chandravva Hebbi", table_body_style)],
        [Paragraph("<b>Team Members</b>", table_body_bold), Paragraph("Vishruth Rallapalli, Yuvraj Rao, Vivan Jeevan Naik, Srujan Kumar J", table_body_style)],
        [Paragraph("<b>Target Dataset</b>", table_body_bold), Paragraph("DAIC-WOZ (Distress Analysis Interview Corpus — Wizard of Oz)", table_body_style)],
        [Paragraph("<b>Clinical Metric Target</b>", table_body_bold), Paragraph("PHQ-8 (Patient Health Questionnaire-8) 3-Class Severity, Binary Status, & Continuous Score", table_body_style)]
    ]
    t_p1 = Table(p1_data, colWidths=[1.8 * inch, 5.2 * inch])
    t_p1.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (1, 0), primary_color),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, light_bg]),
    ]))
    story.append(t_p1)
    story.append(Spacer(1, 6))

    # PART 2: Abstract and Scope
    story.append(Paragraph("Part 2 — Abstract & Technical Scope", h1_style))
    story.append(Paragraph(
        "<b>Abstract:</b> This project presents a participant-independent, multimodal machine learning screening prototype for depression detection and severity assessment using the DAIC-WOZ dataset. The system processes semi-structured interview audio recordings (.wav) and matching transcripts (.csv). The pipeline extracts three distinct information streams: (1) 154-dimensional acoustic features comprising 88 openSMILE eGeMAPS features, 52 MFCC statistics, and 14 prosodic/spectral statistics; (2) 768-dimensional contextual speech representations from frozen Wav2Vec 2.0 (`facebook/wav2vec2-base-960h`); and (3) 768-dimensional contextual linguistic representations extracted from participant-only speech using frozen BERT (`bert-base-uncased`). The standardized 1,690-dimensional multimodal embedding feeds into the final <b>CompactTriModalMLP</b> (225,413 parameters) that simultaneously predicts: (a) 3-class categorical depression severity (Minimal/Mild, Moderate, Moderately Severe/Severe), (b) binary clinical depression status (PHQ-8 ≥ 10), and (c) continuous PHQ-8 score regression (0–24 scale).",
        body_style
    ))
    story.append(Paragraph(
        "<b>Scope & Limitations:</b> The system is strictly intended as a preliminary depression screening / research decision-support prototype to assist clinicians by extracting objective paralinguistic and semantic cues. It is <i>not</i> an autonomous diagnostic authority. The prototype operates within the empirical constraints of the DAIC-WOZ dataset (183 development sessions, 9 reserved unseen test sessions) and incorporates training-set class weighting to handle an 84.15% majority class skew.",
        body_style
    ))
    story.append(Spacer(1, 6))

    # PART 3: Phase-1 / Previous Work Summary
    story.append(Paragraph("Part 3 — Previous Work vs. Current Implementation", h1_style))
    story.append(Paragraph("<b>Previously Implemented (Phase 1 & Phase 2 Baseline):</b> Initial work investigated single-modality acoustic pipelines (basic MFCC extraction via Librosa, pitch tracking) and an 847,749-parameter recurrent BiLSTM with modality gating. Experimental evaluation revealed that the 847k-parameter BiLSTM severely overfit the small training cohort (116 sessions), collapsing into the 84% majority Minimal/Mild class (achieving 83.78% accuracy but 0.00% recall on moderate and severe depression on holdout data).", body_style))
    story.append(Paragraph("<b>Currently Implemented in Final Model:</b>", body_style))
    story.append(Paragraph("• <b>Interviewer Speech Filtering:</b> Automated transcript parsing extracts strictly participant utterances (`speaker == 'Participant'`), discarding Ellie interviewer speech to eliminate dialogue-structure shortcut learning.", bullet_style))
    story.append(Paragraph("• <b>Tri-Modal Feature Standardization:</b> Standardized fusion of 154d Acoustic (88 eGeMAPS + 52 MFCC stats + 14 prosodic stats) + 768d Wav2Vec 2.0 + 768d BERT embeddings (1,690 dimensions).", bullet_style))
    story.append(Paragraph("• <b>Compact MLP Replacement:</b> Replaced the overparameterized 847k-parameter BiLSTM with the 225,413-parameter <b>CompactTriModalMLP</b> regularized with LayerNorm and Dropout.", bullet_style))
    story.append(Paragraph("• <b>Leak-Free 5-Fold Stratification:</b> Implemented a 5-fold Stratified Group Split ensuring strict participant isolation and valid minority representation across folds.", bullet_style))
    story.append(Paragraph("• <b>Macro-F1 Model Selection:</b> Checkpoints are selected based on Validation Macro-F1 and Balanced Accuracy rather than raw cross-entropy loss.", bullet_style))
    story.append(Spacer(1, 6))

    # PART 4: Literature Survey
    story.append(Paragraph("Part 4 — Literature Survey & Inferences", h1_style))
    lit_data = [
        [Paragraph("<b>Reference</b>", table_header_style), Paragraph("<b>Key Methodology</b>", table_header_style), Paragraph("<b>Inference & Impact on Project</b>", table_header_style)],
        [
            Paragraph("<b>Alghowinem et al. (2013)</b><br/>IEEE ICASSP", table_body_bold),
            Paragraph("Analyzed acoustic differences between spontaneous and read clinical speech.", table_body_style),
            Paragraph("Observed flattened fundamental frequency (F0) and reduced formant dynamics in depressed speech. Supported extracting pitch variance and openSMILE acoustic functionals.", table_body_style)
        ],
        [
            Paragraph("<b>Cummins et al. (2015)</b><br/>Speech Communication", table_body_bold),
            Paragraph("Reviewed depression and suicide risk assessment using speech analysis.", table_body_style),
            Paragraph("Identified vocal tract psychomotor retardation (slower speech rate, extended pause duration, spectral energy drop) as valid clinical indicators.", table_body_style)
        ],
        [
            Paragraph("<b>Morales et al. (2018)</b><br/>Interspeech / ACM MM", table_body_bold),
            Paragraph("Multimodal fusion of acoustic features and linguistic transcripts.", table_body_style),
            Paragraph("Demonstrated that combining textual semantics with speech acoustics outperforms unimodal acoustic baselines, informing our BERT + Wav2Vec2 fusion design.", table_body_style)
        ],
        [
            Paragraph("<b>Ringeval et al. (2019)</b><br/>AVEC / ACM MM", table_body_bold),
            Paragraph("AVEC Depression Sub-challenge benchmark on DAIC-WOZ.", table_body_style),
            Paragraph("Provided the baseline eGeMAPS acoustic feature protocol and PHQ-8 clinical severity categorization standards (0-9, 10-14, 15-24).", table_body_style)
        ]
    ]
    t_lit = Table(lit_data, colWidths=[1.8 * inch, 2.4 * inch, 2.8 * inch])
    t_lit.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), primary_color),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, light_bg]),
    ]))
    story.append(t_lit)
    story.append(Spacer(1, 6))

    # PART 5: Complete System Architecture
    story.append(Paragraph("Part 5 — Complete System Architecture & Data Flow", h1_style))
    story.append(Paragraph(
        "The final verified architecture follows a leak-free multimodal pipeline:",
        body_style
    ))
    arch_flow = [
        [Paragraph("<b>Pipeline Stage</b>", table_header_style), Paragraph("<b>Component / Technology</b>", table_header_style), Paragraph("<b>Input → Output</b>", table_header_style), Paragraph("<b>Verified Details</b>", table_header_style)],
        [
            Paragraph("<b>1. Raw Ingestion</b>", table_body_bold),
            Paragraph("DAIC-WOZ Reader<br/>(Librosa, Pandas)", table_body_style),
            Paragraph("Raw .wav + .csv → Audio array & transcript dataframe", table_body_style),
            Paragraph("Session validation against master index; 9 reserved unseen PIDs isolated.", table_body_style)
        ],
        [
            Paragraph("<b>2. Preprocessing</b>", table_body_bold),
            Paragraph("Audio Cleaner & Text Filter<br/>(AudioCleaner, BertExtractor)", table_body_style),
            Paragraph("Raw audio/text → Participant-only speech & text", table_body_style),
            Paragraph("16 kHz resampling, silence trimming, Ellie speech discarded (`speaker == 'Participant'`).", table_body_style)
        ],
        [
            Paragraph("<b>3. Feature Extraction</b>", table_body_bold),
            Paragraph("eGeMAPS + Wav2Vec2 + BERT Extractor", table_body_style),
            Paragraph("Audio/Text → 154d Acoustic, 768d W2V2, 768d BERT", table_body_style),
            Paragraph("154d Acoustic (88 eGeMAPS + 52 MFCC + 14 prosodic stats); frozen Wav2Vec 2.0; frozen BERT-base.", table_body_style)
        ],
        [
            Paragraph("<b>4. Scaling & Fusion</b>", table_body_bold),
            Paragraph("StandardScaler + Concatenation", table_body_style),
            Paragraph("3 feature sets → 1,690d Standardized Fused Vector", table_body_style),
            Paragraph("`StandardScaler` fit strictly on `train_df` (1690d); transformed on val/test/unseen.", table_body_style)
        ],
        [
            Paragraph("<b>5. Final Model</b>", table_body_bold),
            Paragraph("CompactTriModalMLP<br/>(PyTorch)", table_body_style),
            Paragraph("1,690d Vector → 3-Class Softmax, Binary Prob, PHQ-8 Score", table_body_style),
            Paragraph("Linear(1690→128) → LayerNorm → ReLU → Dropout(0.3) → Linear(128→64) → LayerNorm → 3 Heads.", table_body_style)
        ]
    ]
    t_arch = Table(arch_flow, colWidths=[1.1 * inch, 1.8 * inch, 2.0 * inch, 2.1 * inch])
    t_arch.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), primary_color),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, light_bg]),
    ]))
    story.append(t_arch)
    story.append(Spacer(1, 6))

    # PART 6: Dataset Statistics
    story.append(Paragraph("Part 6 — Dataset Statistics & Partitions", h1_style))
    ds_data = [
        [Paragraph("<b>Partition</b>", table_header_style), Paragraph("<b>Sessions</b>", table_header_style), Paragraph("<b>Minimal / Mild (0-9)</b>", table_header_style), Paragraph("<b>Moderate (10-14)</b>", table_header_style), Paragraph("<b>Mod.Severe / Severe (15-24)</b>", table_header_style), Paragraph("<b>Gender (M / F)</b>", table_header_style)],
        [Paragraph("<b>Development Master Index</b>", table_body_bold), Paragraph("183 (100.0%)", table_body_center), Paragraph("154 (84.15%)", table_body_center), Paragraph("20 (10.93%)", table_body_center), Paragraph("9 (4.92%)", table_body_center), Paragraph("113 M / 70 F", table_body_center)],
        [Paragraph("<b>Training Split</b>", table_body_style), Paragraph("116 (63.4%)", table_body_center), Paragraph("98 (84.5%)", table_body_center), Paragraph("12 (10.3%)", table_body_center), Paragraph("6 (5.2%)", table_body_center), Paragraph("72 M / 44 F", table_body_center)],
        [Paragraph("<b>Validation Split</b>", table_body_style), Paragraph("30 (16.4%)", table_body_center), Paragraph("25 (83.3%)", table_body_center), Paragraph("4 (13.3%)", table_body_center), Paragraph("1 (3.3%)", table_body_center), Paragraph("18 M / 12 F", table_body_center)],
        [Paragraph("<b>Test Holdout Split</b>", table_body_style), Paragraph("37 (20.2%)", table_body_center), Paragraph("31 (83.8%)", table_body_center), Paragraph("4 (10.8%)", table_body_center), Paragraph("2 (5.4%)", table_body_center), Paragraph("23 M / 14 F", table_body_center)],
        [Paragraph("<b>Reserved Unseen Set</b>", table_body_bold), Paragraph("9 (Isolated)", table_body_center), Paragraph("4 (44.4%)", table_body_center), Paragraph("2 (22.2%)", table_body_center), Paragraph("3 (33.3%)", table_body_center), Paragraph("5 M / 4 F", table_body_center)]
    ]
    t_ds = Table(ds_data, colWidths=[1.6 * inch, 1.0 * inch, 1.1 * inch, 1.1 * inch, 1.1 * inch, 1.1 * inch])
    t_ds.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), primary_color),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, light_bg]),
    ]))
    story.append(t_ds)
    story.append(Spacer(1, 6))

    # PART 7, 8, 9, 10: Modalities & Acoustic Table
    story.append(Paragraph("Parts 7, 8, 9 & 10 — Modality Extraction Pipelines & Acoustic Inventory", h1_style))
    story.append(Paragraph("• <b>Audio Pipeline (Wav2Vec 2.0):</b> Ingests 16 kHz resampled audio. Frozen `facebook/wav2vec2-base-960h` extracts contextual speech representations pooled into a 768-dimensional vector per session.", body_style))
    story.append(Paragraph("• <b>Transcript Pipeline (BERT):</b> Participant utterances are extracted by filtering `speaker == 'Participant'`. Frozen `bert-base-uncased` processes text in 512-token sliding windows with 128-token overlap, mean-pooling `[CLS]` tokens into a 768-dimensional contextual linguistic vector per session.", body_style))
    story.append(Paragraph("• <b>154-Dimensional Acoustic Feature Breakdown:</b>", body_style))

    ac_table = [
        [Paragraph("<b>Feature Group</b>", table_header_style), Paragraph("<b>Dimensions</b>", table_header_style), Paragraph("<b>Detailed Processing</b>", table_header_style), Paragraph("<b>Clinical Indicator</b>", table_header_style)],
        [
            Paragraph("<b>eGeMAPS v02 Set</b>", table_body_bold),
            Paragraph("88 features", table_body_center),
            Paragraph("openSMILE standard set: Pitch (F0), jitter, shimmer, loudness, alpha ratio, Hammarberg index, formants F1-F3.", table_body_style),
            Paragraph("Physiological biomarkers of psychomotor vocal tract slowing and vocal fold tension.", table_body_style)
        ],
        [
            Paragraph("<b>MFCC Statistics</b>", table_body_bold),
            Paragraph("52 features (26 mean + 26 std)", table_body_center),
            Paragraph("Librosa frame extraction (25ms window, 10ms hop): 13 static MFCCs + 13 delta MFCCs (mean & std across frames).", table_body_style),
            Paragraph("Captures articulation dynamics and spectral timbre variability.", table_body_style)
        ],
        [
            Paragraph("<b>Prosodic & Spectral Statistics</b>", table_body_bold),
            Paragraph("14 features (7 mean + 7 std)", table_body_center),
            Paragraph("RMS Energy, F0 fundamental frequency, Spectral Centroid, Spectral Rolloff, Zero Crossing Rate (mean & std across frames).", table_body_style),
            Paragraph("Monitors flat affect, low acoustic energy, and reduced pitch modulation.", table_body_style)
        ]
    ]
    t_ac = Table(ac_table, colWidths=[1.6 * inch, 1.4 * inch, 2.2 * inch, 1.8 * inch])
    t_ac.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), primary_color),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, light_bg]),
    ]))
    story.append(t_ac)
    story.append(Spacer(1, 6))

    # PART 11, 12, 13: Model Configuration & Parameters
    story.append(Paragraph("Parts 11, 12 & 13 — Model Inventory & Final CompactTriModalMLP Configuration", h1_style))
    story.append(Paragraph("<b>Model Parameter Comparison:</b> Earlier BiLSTM baseline = <b>847,749 parameters</b> (replaced due to severe overfitting). Final <b>CompactTriModalMLP = 225,413 total parameters</b> (225,413 trainable | Encoders frozen).", body_style))

    cfg_data = [
        [Paragraph("<b>Configuration Property</b>", table_header_style), Paragraph("<b>Verified Value / Setting</b>", table_header_style), Paragraph("<b>Implementation Details</b>", table_header_style)],
        [Paragraph("<b>Final Model Name</b>", table_body_bold), Paragraph("<b>CompactTriModalMLP</b>", table_body_style), Paragraph("2-layer regularized multi-task Multimodal MLP.", table_body_style)],
        [Paragraph("<b>Input Representation</b>", table_body_bold), Paragraph("1,690-dim Fused Vector", table_body_style), Paragraph("154d Acoustic + 768d Wav2Vec2 + 768d BERT (StandardScaled).", table_body_style)],
        [Paragraph("<b>Hidden Layers & Regularization</b>", table_body_bold), Paragraph("Linear(1690→128) → LayerNorm → ReLU → Dropout(0.3) → Linear(128→64) → LayerNorm → ReLU", table_body_style), Paragraph("LayerNorm and 30% dropout prevent co-adaptation on small training cohorts.", table_body_style)],
        [Paragraph("<b>Loss Function</b>", table_body_bold), Paragraph("Multi-Task: `L = CrossEntropy(weight=W) + 0.5*BCE + 0.2*MSE(y/24)`", table_body_style), Paragraph("Combines 3-class severity classification, binary depression diagnosis, and normalized PHQ regression.", table_body_style)],
        [Paragraph("<b>Class Weights (Train Only)</b>", table_body_bold), Paragraph("<b>Class 0: 0.3946 | Class 1: 3.2222 | Class 2: 6.4444</b>", table_body_style), Paragraph("Inverse-frequency weights computed strictly on `train_df` (98 / 12 / 6 distribution).", table_body_style)],
        [Paragraph("<b>Optimizer & Schedule</b>", table_body_bold), Paragraph("AdamW (lr=1e-3, weight_decay=1e-3) + CosineAnnealingLR (T_max=40)", table_body_style), Paragraph("Smooth learning rate decay with batch size = 16.", table_body_style)],
        [Paragraph("<b>Checkpoint Selection</b>", table_body_bold), Paragraph("Primary: <b>Validation Macro-F1</b> | Secondary: <b>Balanced Accuracy</b>", table_body_style), Paragraph("Prevents majority-class collapse by rejecting models with 0 minority recall.", table_body_style)],
        [Paragraph("<b>Post-Hoc Calibration</b>", table_body_bold), Paragraph("Post-hoc Temperature Scaling: <b>T = 1.6742</b>", table_body_style), Paragraph("Optimized on validation split logits via L-BFGS.", table_body_style)]
    ]
    t_cfg = Table(cfg_data, colWidths=[1.8 * inch, 2.5 * inch, 2.7 * inch])
    t_cfg.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), primary_color),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, light_bg]),
    ]))
    story.append(t_cfg)
    story.append(Spacer(1, 6))

    # PART 14, 15, 16, 17: Results & Confusion Matrices
    story.append(Paragraph("Parts 14, 15, 16 & 17 — Experimental Results & Confusion Matrices", h1_style))
    story.append(Paragraph("<b>Final Verified Results Across Splits (CompactTriModalMLP, Seed 42):</b>", body_style))

    res_data = [
        [Paragraph("<b>Metric</b>", table_header_style), Paragraph("<b>Train Split (116)</b>", table_header_style), Paragraph("<b>Validation Split (30)</b>", table_header_style), Paragraph("<b>Test Holdout (37)</b>", table_header_style), Paragraph("<b>Majority Baseline</b>", table_header_style)],
        [Paragraph("<b>3-Class Accuracy</b>", table_body_bold), Paragraph("100.00%", table_body_center), Paragraph("80.00%", table_body_center), Paragraph("<b>83.78%</b>", table_body_center), Paragraph("83.78% (31/37)", table_body_center)],
        [Paragraph("<b>3-Class Macro-F1</b>", table_body_bold), Paragraph("100.00%", table_body_center), Paragraph("57.11%", table_body_center), Paragraph("<b>42.36%</b>", table_body_center), Paragraph("30.39%", table_body_center)],
        [Paragraph("<b>Balanced Accuracy</b>", table_body_bold), Paragraph("100.00%", table_body_center), Paragraph("71.00%", table_body_center), Paragraph("<b>48.92%</b>", table_body_center), Paragraph("33.33%", table_body_center)],
        [Paragraph("<b>Binary Accuracy (≥10)</b>", table_body_bold), Paragraph("100.00%", table_body_center), Paragraph("76.67%", table_body_center), Paragraph("<b>86.49%</b>", table_body_center), Paragraph("83.78%", table_body_center)],
        [Paragraph("<b>Binary F1 (≥10)</b>", table_body_bold), Paragraph("100.00%", table_body_center), Paragraph("36.36%", table_body_center), Paragraph("<b>54.55%</b>", table_body_center), Paragraph("0.00%", table_body_center)],
        [Paragraph("<b>Severe Class Recall</b>", table_body_bold), Paragraph("100.00% (6/6)", table_body_center), Paragraph("100.00% (1/1)", table_body_center), Paragraph("<b>50.00% (1/2)*</b>", table_body_center), Paragraph("0.00% (0/2)", table_body_center)],
        [Paragraph("<b>PHQ-8 Score MAE</b>", table_body_bold), Paragraph("2.91 pts", table_body_center), Paragraph("3.09 pts", table_body_center), Paragraph("<b>3.07 pts</b>", table_body_center), Paragraph("3.32 pts", table_body_center)]
    ]
    t_res = Table(res_data, colWidths=[1.8 * inch, 1.3 * inch, 1.3 * inch, 1.3 * inch, 1.3 * inch])
    t_res.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), primary_color),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, light_bg]),
    ]))
    story.append(t_res)
    story.append(Paragraph("<i>*Note on Severe Recall: 50.00% reflects 1 out of 2 severe test participants correctly classified; small sample denominator noted.</i>", body_style))
    story.append(Spacer(1, 4))

    story.append(Paragraph("<b>Confusion Matrices (Rows = True Class, Columns = Predicted Class):</b>", h2_style))
    cm_data = [
        [Paragraph("<b>Split</b>", table_header_style), Paragraph("<b>True Class \\ Predicted Class</b>", table_header_style), Paragraph("<b>Minimal/Mild (0)</b>", table_header_style), Paragraph("<b>Moderate (1)</b>", table_header_style), Paragraph("<b>Mod.Severe/Severe (2)</b>", table_header_style)],
        [Paragraph("<b>Train (116)</b>", table_body_bold), Paragraph("Minimal/Mild (0)<br/>Moderate (1)<br/>Severe (2)", table_body_style), Paragraph("98<br/>0<br/>0", table_body_center), Paragraph("0<br/>12<br/>0", table_body_center), Paragraph("0<br/>0<br/>6", table_body_center)],
        [Paragraph("<b>Val (30)</b>", table_body_bold), Paragraph("Minimal/Mild (0)<br/>Moderate (1)<br/>Severe (2)", table_body_style), Paragraph("22<br/>3<br/>0", table_body_center), Paragraph("1<br/>1<br/>0", table_body_center), Paragraph("2<br/>0<br/>1", table_body_center)],
        [Paragraph("<b>Test (37)</b>", table_body_bold), Paragraph("Minimal/Mild (0)<br/>Moderate (1)<br/>Severe (2)", table_body_style), Paragraph("30<br/>2<br/>1", table_body_center), Paragraph("0<br/>0<br/>0", table_body_center), Paragraph("1<br/>2<br/>1", table_body_center)]
    ]
    t_cm = Table(cm_data, colWidths=[1.1 * inch, 2.3 * inch, 1.2 * inch, 1.2 * inch, 1.2 * inch])
    t_cm.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), primary_color),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, light_bg]),
    ]))
    story.append(t_cm)
    story.append(Spacer(1, 6))

    # PART 18: Unseen Evaluation
    story.append(Paragraph("Part 18 — Evaluation on 9 Reserved Unseen Clinical Sessions", h1_style))
    story.append(Paragraph(
        "The 9 sessions were permanently isolated from training, validation, scaling, threshold tuning, and calibration. The table below presents the verified single-session evaluation results from `evaluate_unseen.py`:",
        body_style
    ))

    unseen_data = [
        [Paragraph("<b>Session / PID</b>", table_header_style), Paragraph("<b>Words</b>", table_header_style), Paragraph("<b>Actual Severity</b>", table_header_style), Paragraph("<b>Predicted Severity</b>", table_header_style), Paragraph("<b>Outcome</b>", table_header_style), Paragraph("<b>Softmax Conf.</b>", table_header_style), Paragraph("<b>GT / Pred PHQ</b>", table_header_style)],
        [Paragraph("<b>304_AUDIO</b>", table_body_bold), Paragraph("976", table_body_center), Paragraph("Minimal / Mild (6)", table_body_style), Paragraph("Minimal / Mild", table_body_style), Paragraph("<font color='green'><b>CORRECT</b></font>", table_body_center), Paragraph("90.3%", table_body_center), Paragraph("6 / 3.2", table_body_center)],
        [Paragraph("<b>397_AUDIO</b>", table_body_bold), Paragraph("1,654", table_body_center), Paragraph("Minimal / Mild (5)", table_body_style), Paragraph("Minimal / Mild", table_body_style), Paragraph("<font color='green'><b>CORRECT</b></font>", table_body_center), Paragraph("93.8%", table_body_center), Paragraph("5 / 2.9", table_body_center)],
        [Paragraph("<b>303_AUDIO</b>", table_body_bold), Paragraph("1,960", table_body_center), Paragraph("Minimal / Mild (0)", table_body_style), Paragraph("Minimal / Mild", table_body_style), Paragraph("<font color='green'><b>CORRECT</b></font>", table_body_center), Paragraph("84.9%", table_body_center), Paragraph("0 / 3.4", table_body_center)],
        [Paragraph("<b>408_AUDIO</b>", table_body_bold), Paragraph("849", table_body_center), Paragraph("Minimal / Mild (1)", table_body_style), Paragraph("Minimal / Mild", table_body_style), Paragraph("<font color='green'><b>CORRECT</b></font>", table_body_center), Paragraph("93.9%", table_body_center), Paragraph("1 / 2.8", table_body_center)],
        [Paragraph("<b>409_AUDIO</b>", table_body_bold), Paragraph("2,352", table_body_center), Paragraph("Moderate Depression (10)", table_body_style), Paragraph("Moderate Depression", table_body_style), Paragraph("<font color='green'><b>CORRECT</b></font>", table_body_center), Paragraph("71.6%", table_body_center), Paragraph("10 / 3.5", table_body_center)],
        [Paragraph("<b>321_AUDIO</b>", table_body_bold), Paragraph("910", table_body_center), Paragraph("Mod.Severe / Severe (20)", table_body_style), Paragraph("Mod.Severe / Severe", table_body_style), Paragraph("<font color='green'><b>CORRECT</b></font>", table_body_center), Paragraph("87.7%", table_body_center), Paragraph("20 / 3.5", table_body_center)],
        [Paragraph("<b>300_AUDIO</b>", table_body_bold), Paragraph("352", table_body_center), Paragraph("Moderate Depression (12)", table_body_style), Paragraph("Minimal / Mild", table_body_style), Paragraph("<font color='red'>INCORRECT</font>", table_body_center), Paragraph("94.8%", table_body_center), Paragraph("12 / 2.8", table_body_center)],
        [Paragraph("<b>338_AUDIO</b>", table_body_bold), Paragraph("839", table_body_center), Paragraph("Mod.Severe / Severe (15)", table_body_style), Paragraph("Minimal / Mild", table_body_style), Paragraph("<font color='red'>INCORRECT</font>", table_body_center), Paragraph("84.1%", table_body_center), Paragraph("15 / 3.1", table_body_center)],
        [Paragraph("<b>414_AUDIO</b>", table_body_bold), Paragraph("1,442", table_body_center), Paragraph("Mod.Severe / Severe (16)", table_body_style), Paragraph("Minimal / Mild", table_body_style), Paragraph("<font color='red'>INCORRECT</font>", table_body_center), Paragraph("94.0%", table_body_center), Paragraph("16 / 2.6", table_body_center)]
    ]
    t_unseen = Table(unseen_data, colWidths=[1.1 * inch, 0.7 * inch, 1.5 * inch, 1.5 * inch, 0.8 * inch, 0.8 * inch, 0.6 * inch])
    t_unseen.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), primary_color),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, light_bg]),
    ]))
    story.append(t_unseen)
    story.append(Paragraph("<b>Overall Unseen Accuracy: 6 / 9 (66.67%)</b> &nbsp;|&nbsp; Correctly identified 4/4 Minimal/Mild, 1/2 Moderate (PID 409 at 71.6% conf), and 1/3 Severe (PID 321 at 87.7% conf).", body_style))
    story.append(Spacer(1, 6))

    # PART 19, 20, 21, 22: Modules, Tech Stack & Verification
    story.append(Paragraph("Parts 19, 20, 21 & 22 — Modules, Testing & Technologies", h1_style))
    mod_data = [
        [Paragraph("<b>Module Name</b>", table_header_style), Paragraph("<b>Purpose</b>", table_header_style), Paragraph("<b>Input → Output</b>", table_header_style), Paragraph("<b>Technologies Used</b>", table_header_style)],
        [Paragraph("<b>Feature Store</b>", table_body_bold), Paragraph("Compressed caching of multimodal features.", table_body_style), Paragraph("Raw audio/text → `.npz` feature files", table_body_style), Paragraph("NumPy, Joblib", table_body_style)],
        [Paragraph("<b>BERT Extractor</b>", table_body_bold), Paragraph("Participant-only linguistic feature extraction.", table_body_style), Paragraph(".csv transcript → 768d CLS embedding", table_body_style), Paragraph("HuggingFace Transformers, PyTorch", table_body_style)],
        [Paragraph("<b>Wav2Vec Extractor</b>", table_body_bold), Paragraph("Self-supervised speech representation extraction.", table_body_style), Paragraph("16 kHz audio → 768d speech embedding", table_body_style), Paragraph("Torchaudio, PyTorch", table_body_style)],
        [Paragraph("<b>eGeMAPS Extractor</b>", table_body_bold), Paragraph("Standard Geneva acoustic biomarker extraction.", table_body_style), Paragraph("16 kHz audio → 88d acoustic vector", table_body_style), Paragraph("openSMILE, Librosa", table_body_style)],
        [Paragraph("<b>Training Engine</b>", table_body_bold), Paragraph("Leak-free training, stratification & benchmarking.", table_body_style), Paragraph("Features → `best_depression_model.pth`", table_body_style), Paragraph("PyTorch, Scikit-learn", table_body_style)],
        [Paragraph("<b>Single Evaluator</b>", table_body_bold), Paragraph("Inference on single session recordings.", table_body_style), Paragraph(".wav + .csv → Clinical severity report", table_body_style), Paragraph("PyTorch, Joblib, Librosa", table_body_style)],
        [Paragraph("<b>Unseen Benchmark</b>", table_body_bold), Paragraph("Automated evaluation on reserved unseen dataset.", table_body_style), Paragraph("`unseen/` folder → Full clinical report", table_body_style), Paragraph("Pandas, PyTorch", table_body_style)]
    ]
    t_mod = Table(mod_data, colWidths=[1.4 * inch, 1.8 * inch, 2.0 * inch, 1.8 * inch])
    t_mod.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), primary_color),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, light_bg]),
    ]))
    story.append(t_mod)
    story.append(Spacer(1, 6))

    # PART 23, 24, 25, 26: Team, Timeline, Limitations
    story.append(Paragraph("Parts 23, 24, 25 & 26 — Team Roles, Demonstrables & Limitations", h1_style))
    story.append(Paragraph("<b>Team Task Distribution:</b>", h2_style))
    story.append(Paragraph("• <b>Vivan Jeevan Naik:</b> Multi-modal neural architecture design, Tri-Modal feature fusion, leak-free Stratified Group Splitting, and training pipeline optimization.", bullet_style))
    story.append(Paragraph("• <b>Vishruth Rallapalli:</b> Acoustic feature extraction pipeline (eGeMAPS 88d, Librosa temporal features), audio cleaner, and silence suppression.", bullet_style))
    story.append(Paragraph("• <b>Yuvraj Rao:</b> Transcript parsing, participant-only text extraction (`speaker == 'Participant'`), and BERT embedding pipeline.", bullet_style))
    story.append(Paragraph("• <b>Srujan Kumar J:</b> Unseen dataset benchmarking engine, calibration (TemperatureScaler), and clinical metric evaluation suite.", bullet_style))
    story.append(Paragraph("<i>Note on Development Work, LOC & Time Spent:</i> Exact individual hours and lines of code are not formally tracked in the repository; modules were integrated pair-wise.", body_style))
    story.append(Spacer(1, 4))

    story.append(Paragraph("<b>Demonstrable CLI Commands:</b>", h2_style))
    story.append(Paragraph("1. Training Execution: `PYTHONPATH=app python3 app/src/models/train_leak_free.py 40` (Trains CompactTriModalMLP, saves checkpoint, reports Train/Val/Test metrics).", bullet_style))
    story.append(Paragraph("2. Unseen Batch Report: `PYTHONPATH=app python3 app/evaluate_unseen.py --full-report` (Evaluates 9 reserved sessions, outputs 66.67% accuracy).", bullet_style))
    story.append(Paragraph("3. Single-Session Prediction: `PYTHONPATH=app python3 app/predict_single_audio.py unseen/severe/321_AUDIO_interviewee.wav` (Outputs instant clinical prediction).", bullet_style))
    story.append(Spacer(1, 4))

    story.append(Paragraph("<b>Documented Limitations:</b>", h2_style))
    story.append(Paragraph("• <b>Small Sample Size & Class Imbalance:</b> Only 9 severe depression sessions exist in the 183-session development corpus (4.92%), making statistical convergence of deep sequential models challenging.", bullet_style))
    story.append(Paragraph("• <b>Moderate Class Transition Overlap:</b> Moderate depression (PHQ-8 10-14) exhibits boundary overlap with both mild and severe classes, resulting in borderline classification confusion.", bullet_style))
    story.append(Spacer(1, 6))

    # PART 27: References (IEEE Style)
    story.append(Paragraph("Part 27 — References (IEEE Format)", h1_style))
    refs = [
        "[1] S. Alghowinem, R. Goecke, M. Wagner, G. Parker, and M. Breakspear, 'Detecting depression: A comparison between spontaneous and read speech,' in <i>IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)</i>, 2013, pp. 7542-7546.",
        "[2] N. Cummins, S. Scherer, J. Krajewski, S. Schnieder, J. Epps, and T. F. Quatieri, 'A review of depression and suicide risk assessment using speech analysis,' <i>Speech Communication</i>, vol. 71, pp. 10-49, 2015.",
        "[3] M. R. A. Morales, P. Stefanov, and F. Ringeval, 'Speech vs. text: A comparative study of multimodal depression detection,' in <i>Interspeech</i>, 2018, pp. 2484-2488.",
        "[4] F. Ringeval et al., 'AVEC 2019 workshop and challenge: state-of-mind, detecting depression with AI, and cross-cultural affect recognition,' in <i>Proceedings of the 9th International on Audio/Visual Emotion Challenge and Workshop</i>, 2019, pp. 3-12.",
        "[5] J. R. Williamson, T. F. Quatieri, B. S. Helfer, G. Ciccarelli, and N. Malyska, 'Vocal biomarkers of depression based on motor coordination,' in <i>Interspeech</i>, 2013, pp. 3384-3388.",
        "[6] F. Eyben et al., 'The Geneva Minimalistic Acoustic Parameter Set (GeMAPS) for Voice Research and Affective Computing,' <i>IEEE Transactions on Affective Computing</i>, vol. 7, no. 2, pp. 190-202, 2016."
    ]
    for r in refs:
        story.append(Paragraph(r, ParagraphStyle('RefText', parent=body_style, fontSize=7.5, leading=10, leftIndent=12, firstLineIndent=-12)))
    story.append(Spacer(1, 6))

    # PART 28 & 29: Final Summary & 2-Minute Viva Script
    story.append(Paragraph("Parts 28 & 29 — Final Summary & 2-Minute Viva Script", h1_style))
    story.append(Paragraph("<b>Final Technical Viva Script (2-Minute Summary):</b>", h2_style))
    viva_text = """
    <i>\"In this Capstone project, we developed a participant-independent, multimodal screening prototype for depression detection using the clinical DAIC-WOZ dataset. Our system processes semi-structured interview recordings by extracting three complementary information streams: 154-dimensional acoustic features (openSMILE eGeMAPS, MFCCs, and prosodic statistics), 768-dimensional deep speech embeddings from frozen Wav2Vec 2.0, and 768-dimensional participant-only contextual linguistic embeddings from frozen BERT.
    <br/><br/>
    During our forensic investigation, we discovered that naive deep recurrent architectures (such as an 847k-parameter BiLSTM) severely overfit the small 116-session training cohort, collapsing into the 84% majority non-depressed class. To overcome this without data leakage, we implemented three key methodological corrections: (1) an automated transcript filter that strips virtual interviewer speech to prevent shortcut learning, (2) a 5-fold Stratified Group Split ensuring strict participant isolation and minority class representation across all folds, and (3) a regularized Compact Tri-Modal Multimodal MLP (225,413 parameters) trained with inverse-frequency class-weighted cross-entropy and validation Macro-F1 checkpoint selection.
    <br/><br/>
    On the held-out test split, our final CompactTriModalMLP achieved <b>83.78% accuracy, 42.36% Macro-F1, 48.92% Balanced Accuracy, 54.55% Binary F1, and 3.07 PHQ-8 MAE</b> (with 50.00% severe recall, 1/2 severe cases). When evaluated on 9 completely untouched unseen clinical sessions, the system attained <b>66.67% accuracy (6/9 correct)</b>, correctly identifying 4/4 Minimal/Mild cases, 1/2 Moderate cases (PID 409 at 71.6% confidence), and 1/3 Severe cases (PID 321 at 87.7% confidence). This provides preliminary evidence that compact multimodal fusion can improve participant-level generalization on this small, highly imbalanced evaluation set.\"</i>
    """
    story.append(Paragraph(viva_text, body_style))

    # Build document
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Successfully generated corrected PDF report: {filename}")


if __name__ == "__main__":
    build_pdf("Capstone_Project_Information_Report.pdf")
