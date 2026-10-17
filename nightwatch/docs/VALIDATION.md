# Packaging validation — 2026-10-17

## Environment

Python 3.12.14; OpenCV 4.11.0; Streamlit 1.66.0; NumPy 2.3.5; pandas 2.2.3; Pillow 12.3.0. Linux runtime. Windows launcher supplied but not executed on a Windows machine here.

## Completed checks

- 12 automated tests passed with `python -m pytest -q` (core algorithms and Streamlit AppTest interactions).
- Synthetic plate region matched the intended rectangle at IoU > 0.55; default settings produced one plate candidate with seven character-like components. This is a synthetic functionality check, not recognition accuracy.
- Blank frame produced no person or plate detections.
- Rotation/ROI, all three edge methods, invalid image handling, 16-bit conversion, change mask, one-to-one evaluation and ZIP export checked.
- Streamlit interaction test analyzed the demo, switched to Laplacian of Gaussian, reanalyzed and reset without an exception.
- Recursive CLI batch processed three synthetic images with zero errors and exported pipeline images plus summary.
- Local Streamlit server started; `/_stcore/health` returned HTTP 200 and `ok`.
- Sample visual outputs were inspected. Plate overlay covered the intended synthetic plate. On OpenCV basketball1, HOG located the upright left person but also produced a false detection around background equipment and failed to correctly box the cropped right person. This illustrates the detector's limits, rather than successful exact counting.
- Local Tesseract ran on the synthetic plate mask and suggested `CAB1234-` for displayed `CAB 1234`, including a spurious trailing hyphen. OCR requires human verification.
- Full Chromium visual browser testing was unavailable in this environment; Streamlit's testing runtime exercised the interface. No claim of pixel-perfect browser rendering is made.

## Not established

No 30-image student dataset was available, and no labelled held-out 10-image study was conducted. Therefore real-world precision, recall, F1, count error, low-light recognition accuracy and performance on the user's Windows lab machine are not claimed. Complete these using your images and the Evaluation tab before reporting assessment results.
