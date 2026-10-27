const pptxgen = require('pptxgenjs');
const { applyTheme } = require('/Users/vivannaik/Library/Application Support/Claude/local-agent-mode-sessions/skills-plugin/66dc01cf-241a-460b-a57e-18cd3cb9e0ba/b0356cca-278b-4e01-addb-2b643b8a933b/skills/pptx/scripts/apply_theme.js');

const HEX = { navy: '14213D', teal: '2A9D8F', coral: 'E76F51', gray: '8D99AE', light: 'EEF3F6', white: 'FFFFFF', ink: '1D2433', muted: '5C6578' };
const THEME = {
  name: 'Depression Screening', headFontFace: 'Cambria', bodyFontFace: 'Calibri',
  colors: { dk1: HEX.ink, lt1: HEX.white, dk2: HEX.navy, lt2: HEX.light, accent1: HEX.teal, accent2: HEX.coral, accent3: HEX.gray,
            accent4: 'E9C46A', accent5: '264653', accent6: 'A8DADC', hlink: HEX.teal, folHlink: HEX.gray },
};
const OUT = process.argv[2];

(async () => {
  const pres = new pptxgen();
  pres.layout = 'LAYOUT_WIDE'; // 13.33 x 7.5
  pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
  pres.title = 'Depression Detection from Interview Speech: Progress and Final Model';
  pres.author = 'Vivan';
  const C = pres.SchemeColor;

  pres.defineSlideMaster({ title: 'DARK', background: { color: HEX.navy },
    objects: [{ placeholder: { options: { name: 'title', type: 'title', x: 0.8, y: 2.0, w: 11.7, h: 1.8, fontSize: 44, bold: true, color: C.background1, valign: 'bottom', align: 'left' }, text: '' } },
              { placeholder: { options: { name: 'body', type: 'body', x: 0.8, y: 4.0, w: 11.7, h: 2.0, fontSize: 20, color: 'CADCFC', valign: 'top' }, text: '' } }] });
  pres.defineSlideMaster({ title: 'CONTENT', background: { color: HEX.white }, slideNumber: { x: 12.4, y: 7.0, w: 0.6, h: 0.3, fontSize: 10, color: HEX.gray },
    objects: [{ placeholder: { options: { name: 'title', type: 'title', x: 0.6, y: 0.35, w: 12.1, h: 0.85, fontSize: 32, bold: true, color: C.text2, valign: 'middle' }, text: '' } },
              { text: { text: 'Depression detection capstone | progress and final model', options: { x: 0.6, y: 7.0, w: 8, h: 0.3, fontSize: 10, color: HEX.gray } } }] });

  let s;
  const card = (x, y, w, h, fill, name) => s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, fill: { color: fill }, line: { color: fill }, rectRadius: 0.12, objectName: name,
    shadow: { type: 'outer', color: '9AA5B1', opacity: 0.25, blur: 6, offset: 2, angle: 90 } });
  const T = (text, o) => s.addText(text, Object.assign({ isTextBox: true, margin: 0, fontFace: 'Calibri', color: HEX.ink, valign: 'top' }, o));
  const arrow = (x1, y1, x2, y2) => s.addShape(pres.shapes.LINE, { x: x1, y: Math.min(y1, y2), w: x2 - x1, h: Math.abs(y2 - y1), flipV: y2 < y1, line: { color: HEX.gray, width: 2, endArrowType: 'triangle' } });
  const box = (x, y, w, h, head, sub, fill, fc, hs) => {
    card(x, y, w, h, fill, head);
    T(head, { x: x + 0.12, y: y + 0.12, w: w - 0.24, h: 0.4, fontSize: hs || 15, bold: true, color: fc, align: 'center' });
    if (sub) T(sub, { x: x + 0.12, y: y + 0.55, w: w - 0.24, h: h - 0.6, fontSize: 13, color: fc, align: 'center' });
  };
  const flow = (y, steps, fills) => { // horizontal chain of boxes with arrows
    const n = steps.length, gap = 0.35, w = (12.1 - gap * (n - 1)) / n;
    steps.forEach(([h, sub], i) => {
      const x = 0.6 + i * (w + gap);
      box(x, y, w, 1.45, h, sub, fills[i] || HEX.light, (fills[i] && fills[i] !== HEX.light) ? HEX.white : HEX.ink, 14);
      if (i < n - 1) arrow(x + w + 0.02, y + 0.72, x + w + gap - 0.02, y + 0.72);
    });
  };
  const bullets = (items, o) => T(items.map((t, i) => (typeof t === 'string'
    ? { text: t, options: { bullet: true, breakLine: i < items.length - 1 } }
    : { text: t[0], options: Object.assign({ bullet: true, breakLine: i < items.length - 1 }, t[1]) })), Object.assign({ fontSize: 16, paraSpaceAfter: 8 }, o));
  const kpi = (x, y, w, n, l, col) => { card(x, y, w, 1.5, HEX.light, 'kpi' + n);
    T(n, { x: x + 0.25, y: y + 0.12, w: w - 0.5, h: 0.75, fontSize: 34, bold: true, color: col || HEX.coral, fontFace: 'Cambria' });
    T(l, { x: x + 0.25, y: y + 0.85, w: w - 0.5, h: 0.6, fontSize: 13, color: HEX.muted }); };
  const tag = (x, y, text, fill) => { s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w: 2.3, h: 0.45, fill: { color: fill }, line: { color: fill }, rectRadius: 0.2 });
    T(text, { x, y, w: 2.3, h: 0.45, fontSize: 14, bold: true, color: HEX.white, align: 'center', valign: 'middle' }); };
  const hdr = (t, col) => ({ text: t, options: { bold: true, color: HEX.white, fill: { color: col || HEX.navy }, align: 'center', valign: 'middle' } });
  const TBL = { fontFace: 'Calibri', color: HEX.ink, border: { type: 'solid', color: 'D5DCE3', pt: 0.75 }, valign: 'middle', margin: 0.08 };

  // ================= INTRO =================
  pres.addSection({ title: 'Introduction' });
  s = pres.addSlide({ masterName: 'DARK', sectionTitle: 'Introduction' });
  s.addText('Detecting Depression from Interview Speech', { placeholder: 'title' });
  s.addText('How the model improved over three versions, the results, and how the final model works\n\nFinal: 79.1% accuracy (cross-validation), 78.8% on 33 unseen people', { placeholder: 'body' });

  s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: 'Introduction' });
  s.addText('The task: screen for depression from one interview', { placeholder: 'title' });
  T([{ text: 'Input: ', options: { bold: true } }, { text: 'audio of a 7-20 minute clinical interview', options: { breakLine: true } },
     { text: 'Output: ', options: { bold: true } }, { text: 'is PHQ-8 >= 10 (clinically depressed)? yes / no', options: { breakLine: true } },
     { text: 'Data: ', options: { bold: true } }, { text: 'DAIC-WOZ + E-DAIC (USC), the standard research benchmark', options: {} }],
    { x: 0.6, y: 1.5, w: 12, h: 1.3, fontSize: 18, paraSpaceAfter: 6 });
  [['272', 'people interviewed'], ['85', 'depressed (31%)'], ['68.8%', 'accuracy of always guessing\n"not depressed" (the baseline)']].forEach(([n, l], i) => {
    const x = 0.6 + i * 4.15;
    card(x, 3.2, 3.85, 3.3, HEX.light, `stat${i}`);
    T(n, { x: x + 0.3, y: 3.55, w: 3.25, h: 1.3, fontSize: 60, bold: true, color: i === 2 ? HEX.coral : HEX.teal, fontFace: 'Cambria' });
    T(l, { x: x + 0.3, y: 5.0, w: 3.25, h: 1.2, fontSize: 16, color: HEX.muted });
  });
  s.addNotes('Any model must beat 68.8% to be useful, because 69% of people are not depressed.');

  // ================= PROGRESS =================
  pres.addSection({ title: 'Progress: three models' });
  s = pres.addSlide({ masterName: 'DARK', sectionTitle: 'Progress: three models' });
  s.addText('Part 1: our progress in three models', { placeholder: 'title' });
  s.addText('Model 1 (Phase 7) -> Model 2 (Phase 8) -> Model 3 (Phase 9, final)', { placeholder: 'body' });

  s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: 'Progress: three models' });
  s.addText('Progress at a glance', { placeholder: 'title' });
  s.addImage({ path: 'chart_progress.png', x: 0.6, y: 1.3, w: 12.1, h: 3.0, altText: 'Accuracy of model 1, 2 and 3: 74.3, 75.2, 79.1 percent' });
  [['Model 1', 'Voice + text. Picks its best voice settings again in every training round.', HEX.gray],
   ['Model 2', 'Same, but the 3 voice settings are fixed. 12 extra learning tricks tested: none helped.', HEX.gray],
   ['Model 3 (final)', 'Model 2 + a local LLM that reads the transcript and rates the 8 PHQ-8 symptoms.', HEX.coral]].forEach(([h, d, col], i) => {
    const x = 0.6 + i * 4.15;
    card(x, 4.55, 3.85, 2.1, HEX.light, 'pg' + i);
    T(h, { x: x + 0.25, y: 4.75, w: 3.35, h: 0.45, fontSize: 18, bold: true, color: col, fontFace: 'Cambria' });
    T(d, { x: x + 0.25, y: 5.25, w: 3.35, h: 1.3, fontSize: 14 });
  });
  s.addNotes('Cross-validated accuracy. The big jump comes from Model 3, the LLM symptom scoring.');

  // Model 1
  s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: 'Progress: three models' });
  s.addText('Model 1 (Phase 7): voice + text, 74.3%', { placeholder: 'title' });
  tag(0.6, 1.35, 'MODEL 1', HEX.gray);
  flow(2.05, [['Audio', 'speech parts found\nby Silero VAD'], ['Voice features', 'wav2vec2, HuBERT,\nWhisper: 15 layers'],
    ['Pick best 3', 'out of 60 voice\nsettings, re-chosen\nin every round'], ['Text', 'transcript ->\nemotion-RoBERTa\nor BERT'], ['Combine', 'logistic\nregression']],
    [HEX.light, HEX.teal, HEX.gray, HEX.teal, HEX.navy]);
  card(0.6, 3.85, 5.9, 2.8, HEX.light, 'm1res');
  T('Result', { x: 0.85, y: 4.0, w: 5.4, h: 0.45, fontSize: 18, bold: true, color: HEX.navy, fontFace: 'Cambria' });
  bullets(['Accuracy 74.3% (baseline 68.8%)', 'AUC 0.764, balanced accuracy 67.0%', 'Catches 64% of depressed people (balanced threshold)'], { x: 0.85, y: 4.55, w: 5.4, h: 2.0 });
  card(6.8, 3.85, 5.9, 2.8, HEX.light, 'm1lim');
  T('What held it back', { x: 7.05, y: 4.0, w: 5.4, h: 0.45, fontSize: 18, bold: true, color: HEX.coral, fontFace: 'Cambria' });
  bullets(['Voice alone carries a weak signal (AUC ~0.72)', 'Text model only captures emotional tone, not symptoms', 'Choosing among 60 settings each round adds noise'], { x: 7.05, y: 4.55, w: 5.4, h: 2.0 });
  s.addNotes('Model 1 was our honest, fully nested model: it re-selects which voice layers to use inside every training fold.');

  // Model 2
  s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: 'Progress: three models' });
  s.addText('Model 2 (Phase 8): fixed voice settings, 75.2%', { placeholder: 'title' });
  tag(0.6, 1.35, 'MODEL 2', HEX.gray);
  T([{ text: 'What changed', options: { bold: true, fontSize: 18, color: HEX.navy, breakLine: true } },
     { text: 'Always use the same 3 voice settings: wav2vec2 layer 6 and HuBERT layers 19 and 20, each with ridge regression.', options: { breakLine: true } },
     { text: ' ', options: { breakLine: true } },
     { text: 'Then we tested 12 learning tricks (teacher\'s ideas included)', options: { bold: true, fontSize: 18, color: HEX.navy, breakLine: true } },
     { text: 'Data augmentation, mixup, class weights, few-shot (prototypical network), one-shot (kNN, prototypes), speaker referencing, domain centring, statistics pooling.', options: { breakLine: true } },
     { text: ' ', options: { breakLine: true } },
     { text: 'Result: 75.2% accuracy, AUC 0.785. None of the tricks helped.', options: { bold: true, color: HEX.coral } }],
    { x: 0.6, y: 2.0, w: 5.7, h: 4.7, fontSize: 15 });
  s.addImage({ path: 'chart_phase8.png', x: 6.6, y: 1.35, w: 6.1, h: 6.1 * 5.2 / 6.4, altText: 'AUC of 12 learning tricks compared with plain ridge' });
  s.addNotes('Lesson from Model 2: the limit was the information in the features, not the learning method. We needed a new kind of information.');

  // Model 3
  s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: 'Progress: three models' });
  s.addText('Model 3 (Phase 9, final): + LLM symptom scores, 79.1%', { placeholder: 'title' });
  tag(0.6, 1.35, 'MODEL 3', HEX.coral);
  card(0.6, 2.0, 5.9, 3.7, HEX.light, 'm3why');
  T('Why we added it', { x: 0.85, y: 2.15, w: 5.4, h: 0.45, fontSize: 18, bold: true, color: HEX.navy, fontFace: 'Cambria' });
  bullets(['Model 2 showed new tricks do not help: we needed new information', 'Our literature review ranked "LLM scores PHQ-8 items from the transcript" as the most promising next step',
    'People mention symptoms ("I can\'t sleep", "always tired"); the old text model ignored this'], { x: 0.85, y: 2.75, w: 5.4, h: 3.8, fontSize: 15 });
  card(6.8, 2.0, 5.9, 3.7, HEX.light, 'm3what');
  T('What changed', { x: 7.05, y: 2.15, w: 5.4, h: 0.45, fontSize: 18, bold: true, color: HEX.coral, fontFace: 'Cambria' });
  bullets(['Mistral-7B reads the transcript and rates each of the 8 PHQ-8 symptoms (0-3)', 'Voice says HOW they speak; LLM says WHAT they report: the two add up',
    ['Accuracy 79.1%, AUC 0.843 (+0.057 over Model 2, statistically significant)', { bold: true }]], { x: 7.05, y: 2.75, w: 5.4, h: 3.8, fontSize: 15 });

  // What is inside each model
  s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: 'Progress: three models' });
  s.addText('What is inside each model', { placeholder: 'title' });
  const k = (t) => ({ text: t, options: { bold: true, fill: { color: HEX.light } } });
  const c = (t, b) => ({ text: t, options: { align: 'center', bold: !!b, color: b ? HEX.coral : HEX.ink } });
  s.addTable([[hdr(''), hdr('Model 1 (Phase 7)'), hdr('Model 2 (Phase 8)'), hdr('Model 3 (FINAL)', HEX.coral)],
    [k('Voice'), c('best 3 of 60 settings,\nre-picked each round'), c('fixed: wav2vec2 L6,\nHuBERT L19 + L20'), c('same as Model 2')],
    [k('Text'), c('emotion-RoBERTa'), c('emotion-RoBERTa'), c('emotion-RoBERTa')],
    [k('Symptom scores (LLM)'), c('no'), c('no'), c('YES: Mistral-7B,\n8 PHQ-8 items', 1)],
    [k('Combiner'), c('logistic regression'), c('logistic regression'), c('logistic regression')],
    [k('Accuracy, cross-validation'), c('74.3%'), c('75.2%'), c('79.1%', 1)],
    [k('Accuracy, 33 unseen'), c('72.7%'), c('72.7%'), c('78.8%', 1)]],
    Object.assign({ x: 0.6, y: 1.45, w: 12.1, colW: [3.1, 3.0, 3.0, 3.0], fontSize: 15, rowH: 0.72 }, TBL));
  T('Only one ingredient is new in Model 3, so the gain comes from the LLM symptom scores.', { x: 0.6, y: 6.55, w: 12.1, h: 0.4, fontSize: 15, color: HEX.muted, italic: true });

  // ================= RESULTS =================
  pres.addSection({ title: 'Results' });
  s = pres.addSlide({ masterName: 'DARK', sectionTitle: 'Results' });
  s.addText('Part 2: results', { placeholder: 'title' });
  s.addText('Test A: cross-validation on all 272 people\nTest B: train on 239, test once on 33 unseen people', { placeholder: 'body' });

  s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: 'Results' });
  s.addText('Two honest tests, no data leakage', { placeholder: 'title' });
  [['Test A: cross-validation', 'All 272 people. Train on 4/5, test on the other 1/5, rotate until everyone is tested; repeat with 5 different splits.', 'Each person is predicted by a model that never saw them.', HEX.teal],
   ['Test B: unseen holdout', 'Train only on 239 people. Test once on 33 people kept aside in the "unseen" folder.', 'All settings (strengths, combiner, threshold) chosen on the 239 only.', HEX.coral]].forEach(([h, a, b, col], i) => {
    const x = 0.6 + i * 6.2;
    card(x, 1.5, 5.9, 3.4, HEX.light, `test${i}`);
    s.addShape(pres.shapes.OVAL, { x: x + 0.3, y: 1.8, w: 0.6, h: 0.6, fill: { color: col }, line: { color: col } });
    T(i === 0 ? 'A' : 'B', { x: x + 0.3, y: 1.8, w: 0.6, h: 0.6, fontSize: 22, bold: true, color: HEX.white, align: 'center', valign: 'middle' });
    T(h, { x: x + 1.1, y: 1.85, w: 4.6, h: 0.5, fontSize: 20, bold: true, fontFace: 'Cambria', color: HEX.navy });
    T([{ text: a, options: { breakLine: true } }, { text: ' ', options: { breakLine: true } }, { text: b, options: { bold: true } }], { x: x + 0.3, y: 2.65, w: 5.3, h: 2.1, fontSize: 15 });
  });
  card(0.6, 5.2, 12.1, 1.35, HEX.white, 'rules');
  T([{ text: 'Rules we kept: ', options: { bold: true } },
     { text: 'nobody appears in both training and test; the LLM never sees a label; interviewer questions removed from transcripts; plan and adoption rule written and pushed to GitHub before running.' }],
    { x: 0.9, y: 5.4, w: 11.5, h: 1.0, fontSize: 15, valign: 'middle' });

  s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: 'Results' });
  s.addText('Result A: accuracy rises from 74.3% to 79.1%', { placeholder: 'title' });
  s.addImage({ path: 'chart_cv.png', x: 0.6, y: 1.4, w: 7.6, h: 5.3, altText: 'Accuracy of baseline and three models under cross-validation' });
  kpi(8.6, 1.5, 4.1, '0.843', 'AUC (Model 1: 0.764)'); kpi(8.6, 3.25, 4.1, '72.4%', 'balanced accuracy (Model 1: 67.0%)');
  kpi(8.6, 5.0, 4.1, '77%', 'of depressed people caught at the balanced threshold (Model 1: 64%)');

  s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: 'Results' });
  s.addText('Result B: 26 of 33 unseen people correct', { placeholder: 'title' });
  s.addImage({ path: 'chart_holdout.png', x: 0.6, y: 1.4, w: 7.6, h: 5.3, altText: 'Accuracy of baseline and three models on 33 unseen people' });
  kpi(8.6, 1.5, 4.1, '0.868', 'AUC (Model 1: 0.777)'); kpi(8.6, 3.25, 4.1, '21 / 22', 'healthy people correctly cleared');
  kpi(8.6, 5.0, 4.1, '9 / 11', 'depressed caught at the balanced threshold (Model 1: 7)');
  s.addNotes('Test A 79.1% and Test B 78.8% agree within 0.3 points. With 33 people, one person is 3 percentage points.');

  s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: 'Results' });
  s.addText('All results side by side', { placeholder: 'title' });
  const sec = (t) => [{ text: t, options: { bold: true, color: HEX.navy, fill: { color: HEX.light }, colspan: 4 } }];
  s.addTable([[{ text: 'Metric', options: { bold: true, color: HEX.white, fill: { color: HEX.navy } } }, hdr('Model 1'), hdr('Model 2'), hdr('Model 3 (FINAL)', HEX.coral)],
    sec('Test A: cross-validation, 272 people (baseline 68.8%)'),
    ['Accuracy', c('74.3%'), c('75.2%'), c('79.1%', 1)], ['Balanced accuracy', c('67.0%'), c('68.3%'), c('72.4%', 1)],
    ['AUC', c('0.764'), c('0.785'), c('0.843', 1)], ['MCC / macro-F1', c('0.37 / 0.68'), c('0.39 / 0.69'), c('0.49 / 0.74', 1)],
    sec('Test B: train 239, test 33 unseen (baseline 66.7%)'),
    ['Accuracy', c('72.7% (24/33)'), c('72.7% (24/33)'), c('78.8% (26/33)', 1)], ['AUC', c('0.777'), c('0.785'), c('0.868', 1)],
    ['Depressed caught (balanced thr.)', c('7 / 11'), c('7 / 11'), c('9 / 11', 1)]],
    Object.assign({ x: 0.6, y: 1.4, w: 12.1, colW: [3.7, 2.8, 2.8, 2.8], fontSize: 15, rowH: 0.48 }, TBL));
  T('Model 3 vs Model 1: AUC +0.079 [95% CI +0.035, +0.124]. Model 3 vs Model 2: +0.057 [+0.015, +0.102]. Both significant.', { x: 0.6, y: 6.45, w: 12.1, h: 0.4, fontSize: 14, color: HEX.muted, italic: true });

  s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: 'Results' });
  s.addText('Screenshot: the results as printed', { placeholder: 'title' });
  s.addImage({ path: 'ss_results.png', x: 0.6, y: 1.4, w: 8.3, h: 8.3 * 680 / 1269, altText: 'Terminal output of the cross-validation and holdout results' });
  T([{ text: 'Top block: ', options: { bold: true } }, { text: 'Test A, cross-validation on 272.', options: { breakLine: true } }, { text: ' ', options: { breakLine: true } },
     { text: 'Bottom block: ', options: { bold: true } }, { text: 'Test B, trained on 239, tested on 33 unseen.', options: { breakLine: true } }, { text: ' ', options: { breakLine: true } },
     { text: 'P4 = Model 1, R0 = Model 2, F1 = Model 3 (orange).', options: { bold: true, color: HEX.coral, breakLine: true } }, { text: ' ', options: { breakLine: true } },
     { text: 'acc = accuracy, bal-acc = balanced accuracy, sens = depressed caught, spec = healthy cleared.', options: { color: HEX.muted } }],
    { x: 9.3, y: 1.5, w: 3.4, h: 4.8, fontSize: 15 });

  // ================= NEW MODEL IN DEPTH =================
  pres.addSection({ title: 'The final model in depth' });
  s = pres.addSlide({ masterName: 'DARK', sectionTitle: 'The final model in depth' });
  s.addText('Part 3: the final model, step by step', { placeholder: 'title' });
  s.addText('Step 1 voice  |  Step 2 text  |  Step 3 LLM symptom scores  |  Step 4 combine', { placeholder: 'body' });

  s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: 'The final model in depth' });
  s.addText('The final model: four steps', { placeholder: 'title' });
  box(0.5, 2.9, 2.0, 1.4, 'Interview audio', 'participant\nspeech only', HEX.light, HEX.ink);
  box(3.2, 1.45, 5.9, 1.2, 'Step 1. Voice: how they speak', 'wav2vec2 + HuBERT (frozen) -> ridge', HEX.teal, HEX.white);
  box(3.2, 3.6, 2.6, 1.4, 'Transcript', 'Whisper speech-\nto-text', HEX.light, HEX.ink);
  box(6.5, 2.95, 2.6, 1.3, 'Step 2. Text', 'emotion-RoBERTa\nemotional tone', HEX.teal, HEX.white);
  box(6.5, 4.5, 2.6, 1.45, 'Step 3. LLM (NEW)', 'Mistral-7B rates the\n8 PHQ-8 symptoms', HEX.coral, HEX.white);
  box(9.8, 1.45, 3.0, 4.5, 'Step 4. Combine', 'logistic regression\n\n-> risk score 0-1\n\n-> flag if above\nthreshold', HEX.navy, HEX.white);
  arrow(2.5, 3.6, 3.2, 2.05); arrow(2.5, 3.6, 3.2, 4.3); arrow(5.8, 4.3, 6.5, 3.6); arrow(5.8, 4.3, 6.5, 5.2);
  arrow(9.1, 2.05, 9.8, 2.05); arrow(9.1, 3.6, 9.8, 3.6); arrow(9.1, 5.2, 9.8, 5.2);
  T('Each step gives its own score from 0 to 1; step 4 merges the three into one risk score.', { x: 0.6, y: 6.3, w: 12, h: 0.5, fontSize: 15, color: HEX.muted, italic: true });

  s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: 'The final model in depth' });
  s.addText('Steps 1 and 2: voice and text scores', { placeholder: 'title' });
  T('Step 1: Voice (how they speak)', { x: 0.6, y: 1.35, w: 8, h: 0.45, fontSize: 18, bold: true, color: HEX.teal, fontFace: 'Cambria' });
  flow(1.9, [['Find speech', 'Silero VAD cuts\nout silence'], ['Voice encoders', 'wav2vec2 layer 6,\nHuBERT layers 19, 20'], ['Average', 'over the whole\ninterview -> 768 /\n1024 numbers'],
    ['Ridge regression', 'predicts the\nPHQ-8 score'], ['Voice score', 'average of the 3,\nas a 0-1 rank']], [HEX.light, HEX.teal, HEX.light, HEX.light, HEX.navy]);
  T('Step 2: Text (emotional tone of the words)', { x: 0.6, y: 3.75, w: 8, h: 0.45, fontSize: 18, bold: true, color: HEX.teal, fontFace: 'Cambria' });
  flow(4.3, [['Transcript', 'Whisper-tiny writes\nwhat they said'], ['Emotion model', 'emotion-RoBERTa-\nlarge reads it'], ['Embedding', '1024 numbers\ndescribing tone'],
    ['Ridge regression', 'predicts the\nPHQ-8 score'], ['Text score', 'as a 0-1 rank']], [HEX.light, HEX.teal, HEX.light, HEX.light, HEX.navy]);
  T('The encoders are pre-trained and frozen; we only train the small ridge regressions (strength chosen by inner cross-validation).', { x: 0.6, y: 6.1, w: 12.1, h: 0.6, fontSize: 14, color: HEX.muted, italic: true });
  s.addNotes('The 0-1 rank: each raw prediction is converted to its position among the training predictions, so the three parts are on the same scale before combining.');

  s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: 'The final model in depth' });
  s.addText('Step 3: how the LLM scores symptoms', { placeholder: 'title' });
  flow(1.4, [['Transcript', 'participant\'s words\nonly'], ['Prompt', '"Rate each PHQ-8\nitem 0-3"'], ['Mistral-7B', '4-bit, runs offline\non the laptop'],
    ['8 symptom scores', 'one number 0-3\nper symptom'], ['Ridge -> LLM score', 'learns which symptoms\nmatter, 0-1 rank']], [HEX.light, HEX.light, HEX.coral, HEX.light, HEX.navy]);
  s.addImage({ path: 'chart_ev.png', x: 0.6, y: 3.15, w: 5.2, h: 3.6, altText: 'Example probabilities for the sleep item' });
  card(6.2, 3.2, 6.5, 3.45, HEX.light, 'ev');
  T('Not just the answer: the probabilities', { x: 6.45, y: 3.35, w: 6.0, h: 0.45, fontSize: 18, bold: true, color: HEX.navy, fontFace: 'Cambria' });
  T([{ text: 'For each symptom we read how likely the model thinks 0, 1, 2 and 3 are, and take the average:', options: { breakLine: true } },
     { text: ' ', options: { breakLine: true } },
     { text: '0 x 0.10 + 1 x 0.45 + 2 x 0.35 + 3 x 0.10 = 1.45', options: { fontFace: 'Courier New', fontSize: 13, bold: true, color: HEX.coral, breakLine: true } },
     { text: ' ', options: { breakLine: true } },
     { text: 'A score of 1.45 keeps the model\'s uncertainty, which works better than a hard "1".', options: {} }],
    { x: 6.45, y: 3.9, w: 6.0, h: 2.6, fontSize: 15 });
  s.addNotes('The bar chart is an illustration of the method. Zero-shot: Mistral was never trained or tuned on our data and never sees a PHQ-8 score.');

  s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: 'The final model in depth' });
  s.addText('Step 3: which symptoms the LLM detects best', { placeholder: 'title' });
  s.addImage({ path: 'chart_items.png', x: 0.6, y: 1.35, w: 6.3, h: 5.1, altText: 'AUC for each of the 8 LLM symptom scores' });
  card(7.3, 1.5, 5.4, 3.9, HEX.light, 'items');
  T('What this shows', { x: 7.55, y: 1.65, w: 4.9, h: 0.45, fontSize: 18, bold: true, color: HEX.navy, fontFace: 'Cambria' });
  bullets(['Sleep, concentration, mood and tiredness each spot depression well on their own (AUC ~0.8)', 'Interest is weak: the LLM gives almost everyone the same score',
    'All 8 together, with no training: AUC 0.82', 'Ridge then learns the best weighting: AUC 0.83'], { x: 7.55, y: 2.25, w: 4.9, h: 4.1, fontSize: 15 });

  s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: 'The final model in depth' });
  s.addText('Step 4: combining the three scores', { placeholder: 'title' });
  T('How much each part counts (logistic-regression weights)', { x: 0.6, y: 1.4, w: 6.2, h: 0.45, fontSize: 18, bold: true, color: HEX.navy, fontFace: 'Cambria' });
  [['Voice', 1.61, HEX.teal], ['Text', 1.32, HEX.teal], ['LLM symptoms', 3.00, HEX.coral]].forEach(([n, wgt, col], i) => {
    const y = 2.1 + i * 0.95;
    T(n, { x: 0.6, y, w: 1.8, h: 0.6, fontSize: 16, bold: true, valign: 'middle' });
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 2.4, y: y + 0.08, w: wgt * 1.25, h: 0.45, fill: { color: col }, line: { color: col }, rectRadius: 0.08 });
    T(wgt.toFixed(2), { x: 2.5 + wgt * 1.25, y, w: 0.9, h: 0.6, fontSize: 16, bold: true, valign: 'middle' });
  });
  T('The LLM counts about twice as much as voice or text.', { x: 0.6, y: 5.0, w: 6.2, h: 0.5, fontSize: 15, color: HEX.muted, italic: true });
  card(7.1, 1.5, 5.6, 4.95, HEX.light, 'thr');
  T('Two decision thresholds', { x: 7.35, y: 1.65, w: 5.1, h: 0.45, fontSize: 18, bold: true, color: HEX.navy, fontFace: 'Cambria' });
  T([{ text: 'Risk score 0 to 1 from the three weighted parts.', options: { breakLine: true } }, { text: ' ', options: { breakLine: true } },
     { text: 'Max-accuracy: flag if > 0.542', options: { bold: true, breakLine: true } }, { text: 'fewest mistakes overall (79% accuracy), but catches only about half of depressed people', options: { breakLine: true } },
     { text: ' ', options: { breakLine: true } },
     { text: 'Balanced: flag if > 0.196', options: { bold: true, color: HEX.coral, breakLine: true } }, { text: 'catches 77% of depressed people; better for screening', options: {} }],
    { x: 7.35, y: 2.25, w: 5.1, h: 4.1, fontSize: 15 });
  s.addNotes('Thresholds were chosen on out-of-fold training scores, never on test people.');

  s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: 'The final model in depth' });
  s.addText('Training the final model and using it', { placeholder: 'title' });
  s.addImage({ path: 'ss_app.png', x: 0.6, y: 1.4, w: 8.4, h: 8.4 * 566 / 1410, altText: 'Terminal output of predict_audio.py scoring participant 606' });
  T([{ text: 'Trained on all 272 people ', options: { bold: true } }, { text: 'with exactly the steps that were tested.', options: { breakLine: true } }, { text: ' ', options: { breakLine: true } },
     { text: 'One command: ', options: { bold: true } }, { text: 'audio in, risk score out, with the 8 symptom ratings shown.', options: { breakLine: true } }, { text: ' ', options: { breakLine: true } },
     { text: 'Matches the training pipeline exactly (score 0.036 for this person).', options: { bold: true, color: HEX.teal } }],
    { x: 9.3, y: 1.5, w: 3.4, h: 4.6, fontSize: 15 });
  T('Saved as models/final_phase9_f1.joblib. Runs offline on an 8 GB MacBook in about 1-3 minutes. Screening aid only, not a diagnosis.', { x: 0.6, y: 5.3, w: 8.4, h: 0.7, fontSize: 14, color: HEX.muted, italic: true });

  // ================= CONCLUSION =================
  pres.addSection({ title: 'Conclusion' });
  s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: 'Conclusion' });
  s.addText('Why we trust it, and its limits', { placeholder: 'title' });
  const list = (x, head, col, items) => {
    card(x, 1.5, 5.9, 4.0, HEX.light, head);
    T(head, { x: x + 0.3, y: 1.75, w: 5.3, h: 0.5, fontSize: 20, bold: true, color: col, fontFace: 'Cambria' });
    bullets(items, { x: x + 0.3, y: 2.4, w: 5.3, h: 3.0, fontSize: 15 });
  };
  list(0.6, 'Why it is real', HEX.teal, ['Both tests agree: 79.1% and 78.8%', 'Without the 17 people who say "I am depressed": AUC still 0.835',
    'Works for both interview setups (AUC 0.84 / 0.86) and both genders', 'LLM never sees labels; nothing tuned on test people']);
  list(6.8, 'Limits', HEX.coral, ['33 unseen people is small: 1 person = 3 points', 'The 33 were opened once before (Oct 19), so a repeat check',
    'At the max-accuracy threshold only about half of depressed people are caught', 'Research prototype, not a diagnosis']);

  s = pres.addSlide({ masterName: 'DARK', sectionTitle: 'Conclusion' });
  s.addText('74.3% -> 79.1% accuracy', { placeholder: 'title' });
  s.addText('Model 1 to Model 2: fixed voice settings, +0.9 points. Model 2 to Model 3: added local LLM symptom scoring, +3.9 points.\nFinal model: 79.1% (cross-validation, AUC 0.84) and 78.8% on 33 unseen people (AUC 0.87), against a 68.8% / 66.7% baseline.', { placeholder: 'body' });

  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log('wrote', OUT);
})();
