# NightWatch — Low-light Surveillance Image Analysis

SE3071 Digital Image Processing · IT number ending in **6** · Python/OpenCV core · localhost browser UI.

## Start on Windows (recommended)

1. Install **Python 3.10–3.13, 64-bit** from https://www.python.org/downloads/. Enable the Python launcher and Add Python to PATH. If Windows says Python was not found, use `py` rather than the Microsoft Store alias.
2. Extract the entire ZIP. Open the extracted `nightwatch` folder, not the ZIP preview.
3. Double-click **START_WINDOWS.bat**. The first launch creates `.venv` and installs requirements; Internet is required for that first installation.
4. Keep the terminal open. Visit **http://localhost:8501** in Chrome/Edge. If the browser does not open automatically, type that address.
5. Tick **Use included synthetic night-plate demo** and click **Analyze images**. Then try your own images.
6. Stop with Ctrl+C in the terminal. Run the same launcher next time.

Manual Command Prompt setup, from the extracted project folder:

```bat
py -3 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501
```

macOS/Linux:

```bash
bash start.sh
```

If port 8501 is occupied, change the run command to `--server.port 8502` and visit http://localhost:8502. Ubuntu may require the `python3-venv` system package. OpenCV is pinned to 4.11 because its bundled HOG detector API is required. Do not install a separate `opencv-python` alongside the headless package.

## What works

- Single-image upload, multiple images, image-folder ZIP, and browser camera snapshots.
- Real image-content validation, corruption errors, file and batch size limits, dimensions, channels, source bit depth and file size.
- Lab-luminance illumination correction, gamma and CLAHE enhancement, followed by grayscale analysis.
- Gaussian, median and bilateral filter comparison; configurable kernel and sigma.
- Canny, Sobel and Gaussian + Laplacian edge maps; selected edges drive plate grouping and object/context edge views.
- Classical **HOG + linear SVM** upright-person detection on original and enhanced paths with non-maximum suppression. The SVM weights are bundled with OpenCV; this is not a deep-learning model.
- Classical plate-region proposals: gradients + morphology + thresholding + contours + geometry + character-like connected components.
- Rotated-rectangle rectification and enlarged grayscale/binary plate crops.
- Optional Tesseract plate text reading; no cloud OCR. Plate-region detection does not require Tesseract.
- Reference-frame change detection using grayscale differences and connected components.
- ROI selection with percentage sliders, 90-degree rotations and bounded resizing.
- Original/enhanced previews, reset, parameter controls, CSV summaries and ZIP exports with every intermediate image, box, parameter and coordinate transformation.
- Manual ground-truth evaluation: precision, recall, F1 and count MAE at a chosen IoU threshold, plus measured processing time.
- CLI folder processing and automated regression tests.

This is an image/snapshot analysis project, not continuous CCTV video tracking. The camera control captures a still image. A reference frame provides a two-frame change comparison. It does not infer identity, intent, criminal activity or face identity.

## Recommended workflow for night scenes

1. Start with a clear original frame; avoid screenshots of screenshots. Try Night / balanced first.
2. Inspect Original ROI and Enhanced. Lower gamma to brighten, but avoid washing out reflective plates. Try switching illumination correction off when headlights produce strong glare.
3. Compare Gaussian and median on noisy footage. Bilateral can preserve boundaries but may be slower. Re-run Analyze after changing controls.
4. Compare all three edge maps. Canny is the default for continuous boundaries; use the Sobel/Laplacian binary threshold to control other maps.
5. For a small plate, use ROI sliders to crop a closer region. Resizing does not create missing image detail.
6. Green boxes are person detections; amber boxes are plate candidates. Review each plate crop. A lower score cutoff can increase false positives.
7. Object edges and Surrounding edges separate **bounding-box regions**, not precise human silhouettes. Undetected people remain in the surrounding view.
8. To detect changes, upload a reference from the same fixed camera, with identical dimensions, orientation and ROI. Lighting changes can trigger differences.
9. Download the results ZIP and summary CSV. Parameters stored in JSON describe the results actually processed, even if controls were changed afterward.

No classical method guarantees a correct human or readable plate in every night image. HOG works best on sufficiently large, upright, mostly visible people. Plate filtering favours a single row of separated, dark/light contrasting characters; unusual fonts, two-line plates, glare and complex backgrounds may fail. Enhancement cannot reconstruct unreadable source detail.

## Optional plate OCR

Install the **Tesseract OCR executable and English language data**, make `tesseract` available on PATH, restart the terminal, then tick the OCR control. On Ubuntu: `sudo apt install tesseract-ocr`. On Windows, follow the official project installation guidance at https://tesseract-ocr.github.io/tessdoc/Installation.html and add its installation directory to PATH. Check with `tesseract --version`.

The core app runs without OCR. OCR output is a suggestion, not a confirmed registration. No invented text is generated if OCR is unavailable or cannot read the crop.

## Dataset and assessment work you must complete

The uploaded brief requires **at least 30 relevant images, including at least 10 captured/produced by the student**, and testing on **at least 10 unseen images**. The included synthetic fixtures and any OpenCV development sample are functionality checks, not your personal dataset or evidence of real-world accuracy.

- Put your own captures in `dataset/student_captured/`.
- Put tuning images in `dataset/development/`.
- Reserve a disjoint test set in `dataset/unseen/`; do not tune on it.
- Record provenance and test purpose in `dataset/manifest.csv` using the provided header.
- Include empty scenes and difficult examples so false positives and misses are measured.
- Suggested declared targets to test, not achieved results: person/plate precision >=0.80, recall >=0.70, and mean processing time <=2 seconds per 1280-side frame on the lab computer. Adjust targets based on your justified scope before evaluation.

In the Evaluation tab, download the blank annotation template. Manually enter `[x,y,width,height]` boxes for both classes on each **Original ROI** image (after rotation, resizing and crop). The coordinate grid helps. Include empty lists only for images truly containing no targets. Every processed filename must appear exactly once. Re-run evaluation after any parameter/geometry change; keep geometry and labels consistent.

Definitions: IoU = intersecting box area / union area. Each prediction can match at most one ground-truth box, and each ground-truth box can be matched once. TP is a match at the IoU cutoff, FP is an unmatched detection, FN is an unmatched target. Precision = TP/(TP+FP), recall = TP/(TP+FN), F1 is their harmonic mean, count MAE is the mean absolute difference between predicted and true per-image counts. Undefined precision/recall denominators are represented as 0; interpret classes with no ground-truth targets accordingly.

A final submission also needs your completed technical report (maximum 10 pages excluding cover/references/appendices) and 3–5 minute demo video. `docs/REPORT_OUTLINE.md` and `docs/DEMO_AND_VIVA.md` guide these. No unseen-image accuracy measurements have been fabricated. Disclose AI/code assistance according to the course rules and be able to explain and modify every submitted component.

## Run a folder without the browser

```bat
.venv\Scripts\python.exe scripts\process_folder.py dataset\unseen --output outputs\unseen_results.zip
```

Linux/macOS: `.venv/bin/python scripts/process_folder.py dataset/unseen --output outputs/unseen_results.zip`.

The CLI processes recursively and writes results to ZIP without extracting input archives. Browser batch limits are 100 images/150 MB; image inputs are limited to 25 MB, 24 megapixels and minimum 16x16. ZIP inputs are read in memory without extracting paths.

## Tests

```bat
.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.venv\Scripts\python.exe -m pytest -q
```

Tests cover synthetic plate localisation, edge maps, ROI/rotation, blank scenes, corrupted input, 16-bit conversion, reference differences, exports, one-to-one evaluation, and browser application interactions. They establish functionality, not real-world detection accuracy. See `docs/VALIDATION.md` for checks performed when packaging.

## Layout

```text
app.py                  Browser interface
src/pipeline.py         Enhancement, edges, detections, geometry, differences
src/evaluation.py       Ground-truth validation and metrics
src/exporting.py        PNG / CSV / JSON / ZIP output
src/ocr.py              Optional local Tesseract execution
scripts/                Demo generation and folder processing
tests/                  Regression and Streamlit interaction tests
dataset/                Manifest and separate dataset folders
docs/                   Methodology, requirements, report and viva guidance
outputs/                Example pipeline export
```

## Sources and acknowledgements

- Project scope: supplied SE3071 Final Project 2026 assessment brief.
- OpenCV documentation: https://docs.opencv.org/4.x/ (HOGDescriptor, Canny, Sobel, Laplacian, morphology, thresholding, CLAHE, perspective transforms).
- HOG detector API and bundled coefficients: https://docs.opencv.org/4.11.0/d5/d33/structcv_1_1HOGDescriptor.html.
- Dalal and Triggs, Histograms of Oriented Gradients for Human Detection, CVPR 2005: https://lear.inrialpes.fr/people/triggs/pubs/Dalal-cvpr05.pdf.
- Streamlit interface: https://docs.streamlit.io/.
- Tesseract OCR: https://tesseract-ocr.github.io/tessdoc/.
- The synthetic fixtures are generated by `scripts/make_demo.py`. If `opencv_basketball1.png` is present, it is an OpenCV development fixture, not a student capture. Source and attribution are in `dataset/SOURCES.md`.
- The integration code was generated with OpenAI assistance for this request. This must be reviewed, adapted, understood and disclosed by the student before submission. Third-party dependencies retain their own licenses.
