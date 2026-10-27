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
  pres.title = 'Depression Detection from Interview Speech: Final Model';
  pres.author = 'Vivan';
  const C = pres.SchemeColor;

  pres.defineSlideMaster({ title: 'DARK', background: { color: HEX.navy },
    objects: [{ placeholder: { options: { name: 'title', type: 'title', x: 0.8, y: 2.2, w: 11.7, h: 1.6, fontSize: 44, bold: true, color: C.background1, valign: 'bottom', align: 'left' }, text: '' } },
              { placeholder: { options: { name: 'body', type: 'body', x: 0.8, y: 4.0, w: 11.7, h: 1.6, fontSize: 20, color: 'CADCFC', valign: 'top' }, text: '' } }] });
  pres.defineSlideMaster({ title: 'CONTENT', background: { color: HEX.white }, slideNumber: { x: 12.4, y: 7.0, w: 0.6, h: 0.3, fontSize: 10, color: HEX.gray },
    objects: [{ placeholder: { options: { name: 'title', type: 'title', x: 0.6, y: 0.35, w: 12.1, h: 0.85, fontSize: 32, bold: true, color: C.text2, valign: 'middle' }, text: '' } },
              { text: { text: 'Depression detection capstone | Phase 9 final model', options: { x: 0.6, y: 7.0, w: 8, h: 0.3, fontSize: 10, color: HEX.gray } } }] });

  const card = (s, x, y, w, h, fill, name) => s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, fill: { color: fill }, line: { color: fill }, rectRadius: 0.12, objectName: name,
    shadow: { type: 'outer', color: '9AA5B1', opacity: 0.25, blur: 6, offset: 2, angle: 90 } });
  const T = (s, text, o) => s.addText(text, Object.assign({ isTextBox: true, margin: 0, fontFace: 'Calibri', color: HEX.ink, valign: 'top' }, o));
  const chartText = { catAxisLabelFontFace: '+mn-lt', valAxisLabelFontFace: '+mn-lt', dataLabelFontFace: '+mn-lt', titleFontFace: '+mn-lt' };

  // 1. Title
  pres.addSection({ title: 'Introduction' });
  let s = pres.addSlide({ masterName: 'DARK', sectionTitle: 'Introduction' });
  s.addText('Detecting Depression from Interview Speech', { placeholder: 'title' });
  s.addText('Final model: voice + text + local LLM symptom scoring\nAccuracy 79% (cross-validation), 78.8% on 33 unseen people', { placeholder: 'body' });
  s.addNotes('Capstone final model. One sentence: we screen for depression (PHQ-8 >= 10) from a recorded interview, and adding a local language model that scores PHQ-8 symptoms raised accuracy from 74% to 79%.');

  // 2. Problem and data
  s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: 'Introduction' });
  s.addText('The task: screen for depression from one interview', { placeholder: 'title' });
  T(s, [{ text: 'Input: ', options: { bold: true } }, { text: 'audio of a 7-20 minute clinical interview', options: { breakLine: true } },
        { text: 'Output: ', options: { bold: true } }, { text: 'is PHQ-8 >= 10 (clinically depressed)? yes / no', options: { breakLine: true } },
        { text: 'Data: ', options: { bold: true } }, { text: 'DAIC-WOZ + E-DAIC (USC), the standard benchmark', options: {} }],
    { x: 0.6, y: 1.5, w: 12, h: 1.3, fontSize: 18, paraSpaceAfter: 6 });
  [['272', 'people interviewed'], ['85', 'depressed (31%)'], ['68.8%', 'accuracy of always guessing\n"not depressed" (the baseline)']].forEach(([n, l], i) => {
    const x = 0.6 + i * 4.15;
    card(s, x, 3.2, 3.85, 3.3, i === 2 ? HEX.light : HEX.light, `stat${i}`);
    T(s, n, { x: x + 0.3, y: 3.55, w: 3.25, h: 1.3, fontSize: 60, bold: true, color: i === 2 ? HEX.coral : HEX.teal, fontFace: 'Cambria' });
    T(s, l, { x: x + 0.3, y: 5.0, w: 3.25, h: 1.2, fontSize: 16, color: HEX.muted });
  });
  s.addNotes('Any model must beat 68.8% just to be useful, because 69% of people are not depressed. That is why we also report balanced accuracy and AUC.');

  // 3. Pipeline
  pres.addSection({ title: 'What we use' });
  s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: 'What we use' });
  s.addText('How the final model works', { placeholder: 'title' });
  const box = (x, y, w, h, head, sub, fill, fc) => {
    card(s, x, y, w, h, fill, head);
    T(s, head, { x: x + 0.15, y: y + 0.15, w: w - 0.3, h: 0.4, fontSize: 16, bold: true, color: fc, align: 'center' });
    T(s, sub, { x: x + 0.15, y: y + 0.6, w: w - 0.3, h: h - 0.7, fontSize: 13, color: fc, align: 'center' });
  };
  const arrow = (x1, y1, x2, y2) => s.addShape(pres.shapes.LINE, { x: x1, y: y1, w: x2 - x1, h: y2 - y1, flipV: y2 < y1, line: { color: HEX.gray, width: 2, endArrowType: 'triangle' } });
  box(0.5, 2.9, 2.0, 1.4, 'Interview audio', 'participant\nspeech only', HEX.light, HEX.ink);
  box(3.2, 1.45, 5.9, 1.2, '1. Voice: how they speak', 'wav2vec2 + HuBERT (frozen) -> ridge', HEX.teal, HEX.white);
  box(3.2, 3.6, 2.6, 1.4, 'Transcript', 'Whisper speech-\nto-text', HEX.light, HEX.ink);
  box(6.5, 2.95, 2.6, 1.3, '2. Text', 'emotion-RoBERTa\nemotional tone', HEX.teal, HEX.white);
  box(6.5, 4.5, 2.6, 1.45, '3. LLM (NEW)', 'Mistral-7B rates the\n8 PHQ-8 symptoms', HEX.coral, HEX.white);
  box(9.8, 1.45, 3.0, 4.5, 'Combine', 'logistic regression\n\n-> risk score 0-1\n\n-> flag if above\nthreshold', HEX.navy, HEX.white);
  arrow(2.5, 3.6, 3.2, 2.05); arrow(2.5, 3.6, 3.2, 4.3); arrow(5.8, 4.3, 6.5, 3.6); arrow(5.8, 4.3, 6.5, 5.2);
  arrow(9.1, 2.05, 9.8, 2.05); arrow(9.1, 3.6, 9.8, 3.6); arrow(9.1, 5.2, 9.8, 5.2);
  T(s, 'Each part gives its own score; a small logistic regression combines the three into one risk score.', { x: 0.6, y: 6.3, w: 12, h: 0.5, fontSize: 15, color: HEX.muted, italic: true });
  s.addNotes('Three views of one interview. Voice: how the person sounds. Text: the emotional tone of their words. LLM: which depression symptoms they actually describe. Only the LLM part is new in Phase 9.');

  // 4. Models table
  s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: 'What we use' });
  s.addText('Every model in the pipeline', { placeholder: 'title' });
  const hdr = (t) => ({ text: t, options: { bold: true, color: HEX.white, fill: { color: HEX.navy } } });
  const rows = [[hdr('Step'), hdr('Model'), hdr('What it does'), hdr('Trained by us?')],
    ['Find speech', 'Silero VAD', 'Cuts out silence', 'No (pre-trained)'],
    ['Voice', 'wav2vec2-base (layer 6), HuBERT-large-ER (layers 19, 20)', 'Turns speech into 768/1024 numbers describing the voice', 'No (frozen)'],
    ['Speech-to-text', 'Whisper-tiny (OpenAI)', 'Writes down what the person said', 'No (pre-trained)'],
    ['Text', 'emotion-RoBERTa-large', 'Turns the transcript into an emotion embedding', 'No (frozen)'],
    [{ text: 'Symptoms (NEW)', options: { bold: true, color: HEX.coral } }, { text: 'Mistral-7B-Instruct v0.3, 4-bit, run locally', options: { bold: true } }, 'Reads the transcript, rates 8 PHQ-8 symptoms 0-3', 'No (zero-shot)'],
    ['Per-part learner', 'Ridge regression (x5)', 'Predicts the PHQ-8 score from each part', { text: 'Yes', options: { bold: true, color: HEX.teal } }],
    ['Combiner', 'Logistic regression', 'Merges the 3 scores into one risk score', { text: 'Yes', options: { bold: true, color: HEX.teal } }]];
  s.addTable(rows, { x: 0.6, y: 1.45, w: 12.1, colW: [2.0, 4.0, 4.3, 1.8], fontSize: 14, fontFace: 'Calibri', color: HEX.ink,
    border: { type: 'solid', color: 'D5DCE3', pt: 0.75 }, fill: { color: HEX.white }, rowH: 0.62, valign: 'middle', margin: 0.08 });
  T(s, 'Big models are used frozen; only small models are trained, which is right for 272 people (no overfitting).', { x: 0.6, y: 6.5, w: 12, h: 0.4, fontSize: 15, color: HEX.muted, italic: true });

  // 5. The LLM idea + item AUC chart
  s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: 'What we use' });
  s.addText('The new idea: let an LLM read the interview', { placeholder: 'title' });
  card(s, 0.6, 1.5, 5.4, 4.9, HEX.light, 'llmcard');
  T(s, [{ text: 'Prompt', options: { bold: true, fontSize: 16, breakLine: true } },
        { text: '"Estimate how the participant would answer each PHQ-8 item, 0-3."', options: { italic: true, breakLine: true } },
        { text: ' ', options: { breakLine: true } },
        { text: 'Answer (participant 606)', options: { bold: true, fontSize: 16, breakLine: true } },
        { text: 'interest 0.1, depressed 0.0, sleep 0.0, tired 0.4, appetite 0.0, failure 0.0, concentration 1.3, psychomotor 0.0', options: { fontFace: 'Courier New', fontSize: 15, breakLine: true } },
        { text: ' ', options: { breakLine: true } },
        { text: 'Never sees any label. No training. Runs offline on the laptop (Apple MLX).', options: { color: HEX.coral, bold: true } }],
    { x: 0.9, y: 1.8, w: 4.8, h: 4.4, fontSize: 17 });
  s.addImage({ path: 'chart_items.png', x: 6.4, y: 1.4, w: 6.3, h: 5.1, altText: 'Bar chart of AUC for each of the 8 LLM symptom scores' });
  s.addNotes('AUC 0.5 = coin toss, 1.0 = perfect. Sleep, concentration, mood and tiredness are the strongest. Interest is weak because the model rates nearly everyone the same.');

  // 6. How we tested
  pres.addSection({ title: 'Testing' });
  s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: 'Testing' });
  s.addText('Two honest tests, no data leakage', { placeholder: 'title' });
  [['Test A: cross-validation', 'All 272 people. Train on 4/5, test on the other 1/5, rotate until everyone is tested; repeat with 5 different splits.', 'Each person is predicted by a model that never saw them.', HEX.teal],
   ['Test B: unseen holdout', 'Train only on 239 people. Test once on 33 people kept aside in the "unseen" folder.', 'All settings (strengths, combiner, threshold) chosen on the 239 only.', HEX.coral]].forEach(([h, a, b, col], i) => {
    const x = 0.6 + i * 6.2;
    card(s, x, 1.5, 5.9, 3.4, HEX.light, `test${i}`);
    s.addShape(pres.shapes.OVAL, { x: x + 0.3, y: 1.8, w: 0.6, h: 0.6, fill: { color: col }, line: { color: col } });
    T(s, i === 0 ? 'A' : 'B', { x: x + 0.3, y: 1.8, w: 0.6, h: 0.6, fontSize: 22, bold: true, color: HEX.white, align: 'center', valign: 'middle' });
    T(s, h, { x: x + 1.1, y: 1.85, w: 4.6, h: 0.5, fontSize: 20, bold: true, fontFace: 'Cambria', color: HEX.navy });
    T(s, [{ text: a, options: { breakLine: true } }, { text: ' ', options: { breakLine: true } }, { text: b, options: { bold: true } }], { x: x + 0.3, y: 2.65, w: 5.3, h: 2.1, fontSize: 15 });
  });
  card(s, 0.6, 5.2, 12.1, 1.35, HEX.white, 'rules');
  T(s, [{ text: 'Rules we kept: ', options: { bold: true } },
        { text: 'nobody appears in both training and test; the LLM never sees a label; interviewer questions removed from transcripts; plan and adoption rule written and pushed to GitHub before running.' }],
    { x: 0.9, y: 5.4, w: 11.5, h: 1.0, fontSize: 15, valign: 'middle' });

  // 7. Results A chart
  pres.addSection({ title: 'Results' });
  s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: 'Results' });
  s.addText('Result A: accuracy rises from 74.3% to 79.1%', { placeholder: 'title' });
  s.addImage({ path: 'chart_cv.png', x: 0.6, y: 1.4, w: 7.6, h: 5.3, altText: 'Accuracy of baseline and three models under cross-validation' });
  [['0.843', 'AUC (was 0.764)'], ['72.4%', 'balanced accuracy (was 67.0%)'], ['77%', 'of depressed people caught at the balanced threshold (was 64%)']].forEach(([n, l], i) => {
    card(s, 8.6, 1.5 + i * 1.75, 4.1, 1.5, HEX.light, `kpiA${i}`);
    T(s, n, { x: 8.85, y: 1.6 + i * 1.75, w: 3.7, h: 0.75, fontSize: 36, bold: true, color: HEX.coral, fontFace: 'Cambria' });
    T(s, l, { x: 8.85, y: 2.35 + i * 1.75, w: 3.7, h: 0.6, fontSize: 13, color: HEX.muted });
  });
  s.addNotes('The gain from Phase 8 to the final model, +0.057 AUC, has a 95% confidence interval of +0.015 to +0.102, so it is not luck.');

  // 8. Results B chart
  s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: 'Results' });
  s.addText('Result B: 26 of 33 unseen people correct', { placeholder: 'title' });
  s.addImage({ path: 'chart_holdout.png', x: 0.6, y: 1.4, w: 7.6, h: 5.3, altText: 'Accuracy of baseline and three models on 33 unseen people' });
  [['0.868', 'AUC (Phase 7: 0.777)'], ['21 / 22', 'healthy people correctly cleared'], ['9 / 11', 'depressed caught at the balanced threshold (Phase 7: 7)']].forEach(([n, l], i) => {
    card(s, 8.6, 1.5 + i * 1.75, 4.1, 1.5, HEX.light, `kpiB${i}`);
    T(s, n, { x: 8.85, y: 1.6 + i * 1.75, w: 3.7, h: 0.75, fontSize: 36, bold: true, color: HEX.coral, fontFace: 'Cambria' });
    T(s, l, { x: 8.85, y: 2.35 + i * 1.75, w: 3.7, h: 0.6, fontSize: 13, color: HEX.muted });
  });
  s.addNotes('Test A 79.1% and Test B 78.8% agree within 0.3 points, so the number is stable. With 33 people, one person is 3 percentage points.');

  // 9. Full table
  s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: 'Results' });
  s.addText('Three models side by side', { placeholder: 'title' });
  const h2 = (t, col) => ({ text: t, options: { bold: true, color: HEX.white, fill: { color: col || HEX.navy }, align: 'center' } });
  const c = (t, b) => ({ text: t, options: { align: 'center', bold: !!b, color: b ? HEX.coral : HEX.ink } });
  const sec = (t) => [{ text: t, options: { bold: true, color: HEX.navy, fill: { color: HEX.light }, colspan: 4 } }];
  s.addTable([[{ text: 'Metric', options: { bold: true, color: HEX.white, fill: { color: HEX.navy } } }, h2('Phase 7 (74.3%)'), h2('Phase 8 (75.2%)'), h2('FINAL (+LLM)', HEX.coral)],
    sec('Test A: cross-validation, 272 people (baseline 68.8%)'),
    ['Accuracy', c('74.3%'), c('75.2%'), c('79.1%', 1)], ['Balanced accuracy', c('67.0%'), c('68.3%'), c('72.4%', 1)],
    ['AUC', c('0.764'), c('0.785'), c('0.843', 1)], ['MCC / macro-F1', c('0.37 / 0.68'), c('0.39 / 0.69'), c('0.49 / 0.74', 1)],
    sec('Test B: train 239, test 33 unseen (baseline 66.7%)'),
    ['Accuracy', c('72.7% (24/33)'), c('72.7% (24/33)'), c('78.8% (26/33)', 1)], ['AUC', c('0.777'), c('0.785'), c('0.868', 1)],
    ['Depressed caught (balanced thr.)', c('7 / 11'), c('7 / 11'), c('9 / 11', 1)]],
    { x: 0.6, y: 1.4, w: 12.1, colW: [3.7, 2.8, 2.8, 2.8], fontSize: 15, fontFace: 'Calibri', color: HEX.ink, rowH: 0.48,
      border: { type: 'solid', color: 'D5DCE3', pt: 0.75 }, valign: 'middle', margin: 0.08 });
  T(s, 'Final vs Phase 7: AUC +0.079 [95% CI +0.035, +0.124]. Final vs Phase 8: +0.057 [+0.015, +0.102]. Both significant.', { x: 0.6, y: 6.45, w: 12.1, h: 0.4, fontSize: 14, color: HEX.muted, italic: true });

  // 10. Screenshot results
  s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: 'Results' });
  s.addText('Screenshot: the results as printed', { placeholder: 'title' });
  s.addImage({ path: 'ss_results.png', x: 0.6, y: 1.4, w: 8.3, h: 8.3 * 680 / 1269, altText: 'Terminal output of the cross-validation and holdout results for the three models' });
  T(s, [{ text: 'Top block: ', options: { bold: true } }, { text: 'Test A, cross-validation on 272.', options: { breakLine: true } }, { text: ' ', options: { breakLine: true } },
        { text: 'Bottom block: ', options: { bold: true } }, { text: 'Test B, trained on 239, tested on 33 unseen.', options: { breakLine: true } }, { text: ' ', options: { breakLine: true } },
        { text: 'Orange = final model.', options: { bold: true, color: HEX.coral, breakLine: true } }, { text: ' ', options: { breakLine: true } },
        { text: 'acc = accuracy, bal-acc = balanced accuracy, sens = depressed caught, spec = healthy cleared.', options: { color: HEX.muted } }],
    { x: 9.3, y: 1.5, w: 3.4, h: 4.8, fontSize: 15 });

  // 11. Screenshot app
  s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: 'Results' });
  s.addText('Screenshot: the app on one recording', { placeholder: 'title' });
  s.addImage({ path: 'ss_app.png', x: 0.6, y: 1.4, w: 8.8, h: 8.8 * 566 / 1410, altText: 'Terminal output of predict_audio.py scoring participant 606' });
  T(s, [{ text: 'One command: ', options: { bold: true } }, { text: 'audio in, risk score out.', options: { breakLine: true } }, { text: ' ', options: { breakLine: true } },
        { text: 'Shows the 8 LLM symptom ratings, so the result is explainable.', options: { breakLine: true } }, { text: ' ', options: { breakLine: true } },
        { text: 'Output matches the training pipeline exactly (score 0.036).', options: { bold: true, color: HEX.teal } }],
    { x: 9.8, y: 1.5, w: 2.9, h: 4.6, fontSize: 15 });
  T(s, 'Runs fully offline on an 8 GB MacBook in about 1-3 minutes. Screening aid only, not a diagnosis.', { x: 0.6, y: 5.4, w: 8.8, h: 0.6, fontSize: 14, color: HEX.muted, italic: true });

  // 12. Is it real / limits
  pres.addSection({ title: 'Conclusion' });
  s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: 'Conclusion' });
  s.addText('Why we trust it, and its limits', { placeholder: 'title' });
  const list = (x, head, col, items) => {
    card(s, x, 1.5, 5.9, 4.0, HEX.light, head);
    T(s, head, { x: x + 0.3, y: 1.75, w: 5.3, h: 0.5, fontSize: 20, bold: true, color: col, fontFace: 'Cambria' });
    T(s, items.map((t, i) => ({ text: t, options: { bullet: true, breakLine: i < items.length - 1 } })), { x: x + 0.3, y: 2.4, w: 5.3, h: 3.8, fontSize: 15, paraSpaceAfter: 8 });
  };
  list(0.6, 'Why it is real', HEX.teal, ['Both tests agree: 79.1% and 78.8%', 'Without the 17 people who say "I am depressed": AUC still 0.835',
    'Works for both interview setups (AUC 0.84 / 0.86) and both genders', 'LLM never sees labels; nothing tuned on test people']);
  list(6.8, 'Limits', HEX.coral, ['33 unseen people is small: 1 person = 3 points', 'The 33 were opened once before (Oct 19), so a repeat check',
    'At the max-accuracy threshold only about half of depressed people are caught; use the balanced threshold for screening', 'Research prototype, not a diagnosis']);

  // 13. Conclusion
  s = pres.addSlide({ masterName: 'DARK', sectionTitle: 'Conclusion' });
  s.addText('79% accuracy, AUC 0.84', { placeholder: 'title' });
  s.addText('Adding local LLM symptom scoring to voice + text raised accuracy from 74.3% to 79.1% (cross-validation) and from 72.7% to 78.8% on 33 unseen people, against a 68.8% / 66.7% baseline.', { placeholder: 'body' });

  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log('wrote', OUT);
})();
