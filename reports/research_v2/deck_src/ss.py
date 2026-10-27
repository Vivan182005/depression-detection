from PIL import Image, ImageDraw, ImageFont
import sys
def shot(lines, out, title):
    f = ImageFont.truetype('/System/Library/Fonts/Menlo.ttc', 26)
    fb = ImageFont.truetype('/System/Library/Fonts/Menlo.ttc', 26, index=1)
    W = max(int(f.getlength(l)) for l, _ in lines) + 80
    H = 70 + 38 * len(lines) + 40
    im = Image.new('RGB', (W, H), (30, 33, 40)); d = ImageDraw.Draw(im)
    d.rectangle([0, 0, W, 50], fill=(52, 56, 66))
    for i, c in enumerate([(237, 106, 94), (245, 191, 79), (98, 197, 84)]):
        d.ellipse([20 + i * 34, 15, 40 + i * 34, 35], fill=c)
    d.text((W // 2 - f.getlength(title) // 2, 11), title, font=f, fill=(200, 200, 200))
    y = 70
    for l, col in lines:
        d.text((40, y), l, font=fb if col == 'hi' else f,
               fill={'p': (120, 200, 255), 'n': (225, 225, 225), 'hi': (255, 170, 120), 'm': (150, 155, 165), 'g': (130, 220, 140)}[col])
        y += 38
    im.save(out)
hdr = '   model                               AUC      acc  bal-acc   sens   spec'
R = [('$ python run_project.py --report', 'p'), ('', 'n'),
     ('Phase 9: nested cross-validation, N = 272 (baseline 68.8%)', 'm'), (hdr, 'm'),
     ('   P4  (74.3% model)                   0.764    74.3%    67.0%    48%    87%', 'n'),
     ('   R0_voice_text  (75% model)          0.785    75.2%    68.3%    50%    87%', 'n'),
     ('   F1_voice_text_llm  (FINAL)          0.843    79.1%    72.4%    55%    90%', 'hi'), ('', 'n'),
     ('Holdout: train 239 -> test the 33 unseen (baseline 66.7%)', 'm'), (hdr, 'm'),
     ('   P4  (74.3% model)                   0.777    72.7%    68.2%    55%    82%', 'n'),
     ('   R0_voice_text  (75% model)          0.785    72.7%    70.5%    64%    77%', 'n'),
     ('   F1_voice_text_llm  (FINAL)          0.868    78.8%    70.5%    45%    95%', 'hi'), ('', 'n'),
     ('Final saved model: models/final_phase9_f1.joblib  (trained on all 272)', 'g')]
shot(R, 'ss_results.png', 'Terminal: results')
A = [('$ python predict_audio.py NEW_AUDIO_DAIC/606_AUDIO_interviewee.wav --domain avatar', 'p'), ('', 'n'),
     ('Scoring PHQ-8 items with the local LLM ...', 'm'),
     ('LLM PHQ-8 item estimates (0-3): interest 0.1, depressed 0.0, sleep 0.0,', 'n'),
     ('   tired 0.4, appetite 0.0, failure 0.0, concentration 1.3, psychomotor 0.0 | sum 1.9', 'n'), ('', 'n'),
     ('Depression risk score: 0.036  (0 = low, 1 = high)', 'hi'),
     ('  Max-accuracy threshold 0.542: not flagged', 'n'),
     ('  Balanced threshold     0.196: not flagged', 'n'), ('', 'n'),
     ('Model: Phase 9 F1, nested-CV AUC 0.84 / accuracy 79%; unseen-33 holdout 78.8%.', 'g'),
     ('(True PHQ-8 of participant 606 = 5: not depressed -> correct)', 'm')]
shot(A, 'ss_app.png', 'Terminal: predict_audio.py on one recording')
