# A 3–5 minute demonstration

0:00–0:30: identify the specific problem and digit-6 domain; show dataset organisation and provenance.
0:30–1:15: upload a real low-light image; run enhancement and explain Lab luminance, gamma and CLAHE.
1:15–2:00: compare Gaussian/median and Canny/Sobel/LoG; change a parameter and re-run.
2:00–2:45: show people detections, plate crops and surrounding edges; describe candidate uncertainty.
2:45–3:15: show reference-frame differences or ROI and rotation on a suitable input.
3:15–4:00: process a folder ZIP, download images/CSV and show real held-out evaluation.
4:00–4:30: discuss one false positive, one missed target and one unreadable plate.

# Be able to answer

- Why do edge maps alone not prove that an object is human?
- How does gamma below 1 affect intensity? What happens to noise?
- Why modify Lab luminance rather than separately equalising BGR?
- What do the two Canny thresholds do?
- What is the difference between first and second derivatives?
- Why smooth before applying a Laplacian?
- Why do signed gradients need float/signed storage?
- How do morphology and connected components help locate plate text?
- What makes a plate score heuristic rather than a probability?
- What does HOG represent and what does its SVM do?
- Why are small and occluded pedestrians difficult?
- Which coordinate system is used by the labels and exports?
- Why can reference-frame differences respond to changing lights?
- What do TP, FP, FN, IoU, precision, recall, F1 and count MAE mean?
- How did you keep the unseen images separate from tuning?
- Which parts were generated or adapted with assistance?

# Small modifications to practise

Change the morphology kernel, plate aspect-ratio bounds, HOG margin threshold, gamma, Gaussian sigma, IoU threshold or minimum changed area. Predict the impact, run the app and explain observed false positives/misses. Read src/pipeline.py rather than memorising only the UI.
