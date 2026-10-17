# Methodology and algorithm explanations

## Specific problem

Improve the inspectability of low-light surveillance frames and locate visible upright people and plausible vehicle registration-plate regions, while retaining intermediate evidence and quantifying detection errors. The workflow is image based; its output is counts, candidate boxes, edge maps, plate crops and reference-frame change regions.

## Acquisition and geometry

Pillow verifies the format/dimensions before OpenCV decodes bytes. Source metadata is recorded before converting grayscale/alpha/16-bit inputs to the 8-bit BGR working representation. Rotation is clockwise in multiples of 90 degrees. The longest image side is limited to the selected maximum. ROI percentages are then converted to pixel coordinates and cropped. Exports record this transform. All detection boxes and ground truth refer to the resulting analysis image. Rotated plate crops use a separate rotated-rectangle homography.

## Illumination and contrast

BGR converts to CIE Lab, keeping two chromatic channels while modifying luminance L. A broad Gaussian estimates illumination B. Corrected luminance is clip(L * (mean(B)+1)/(B+1), 0, 255). The +1 avoids division by zero. This suppresses slow spatial variation, but can reduce intentionally different scene brightness and cannot recover clipped highlights.

A lookup table applies I_out = 255*(I_in/255)^gamma. Gamma below 1 brightens and above 1 darkens. CLAHE uses local histograms with a clip limit and 8x8 tiles to improve local contrast with less noise amplification than unrestricted local equalisation. Lab converts back to BGR for a visual result, then grayscale feeds classical analysis. Thus the pipeline uses Lab, grayscale and binary representations meaningfully.

## Filtering comparison

Gaussian: convolution with a normalised bell-shaped kernel reduces approximately Gaussian sensor noise. Increasing kernel/sigma removes noise but also fine character strokes. Median: replaces each pixel with its neighbourhood median; effective for isolated impulse noise while preserving many boundaries. Bilateral: weights both spatial and intensity distance, preserving stronger edges at higher computational cost. All three outputs are displayed; the selected one feeds edges, plate grouping and the enhanced HOG path. LoG explicitly adds Gaussian smoothing before its Laplacian.

## Edge detection

Sobel estimates first derivatives Gx and Gy; magnitude sqrt(Gx^2+Gy^2) highlights contrast changes. Canny adds non-maximum suppression and hysteresis with two thresholds to produce thin binary contours. Laplacian-of-Gaussian estimates a second derivative after smoothing; the displayed map is the absolute response, not a zero-crossing implementation. A threshold binarises Sobel/LoG for region analysis. Gradients are computed in signed or floating point before absolute conversion, preventing negative responses from wrapping in uint8.

Edge maps describe intensity boundaries, not semantic categories. A fence and a person can both have edges. Consequently a separate gradient-descriptor classifier is used for people, and explicit structural tests are used for plate candidates.

## Person detection

OpenCV's default HOG computes local gradient orientation histograms with block normalisation; a bundled linear SVM classifies windows in an image pyramid. Its default window is 64x128 pixels. The analysis image is optionally scaled to a longest side of 800 for detection cost. The original grayscale path and selected enhanced/filtered path are both evaluated because strong enhancement can reduce gradient recognition. Boxes are scaled back to analysis coordinates and overlapping predictions from both paths undergo IoU non-maximum suppression. The reported score is an SVM margin, not a probability. Upright pedestrians of sufficient size are the intended target; crouching, occluded, tiny or atypical people are a known weakness.

## Plate localisation

Sobel x emphasises the vertical strokes common in characters; Otsu converts gradient magnitude to a binary mask. This is combined with the user's selected binary edge map. A rectangular morphological close joins nearby strokes, and opening removes small residue. A second Otsu bright-region route supplies additional plate proposals. Contours undergo size, aspect ratio and rectangularity checks.

Each plausible rectangle is rectified using minAreaRect and a homography. CLAHE followed by Otsu and adaptive Gaussian thresholding prepares character masks from a less-smoothed enhanced crop, retaining fine strokes. The mask with the most plausible character components (3–14) is selected. Connected components must have plausible relative height, width and area; at least three must look like separated glyphs. This rejects many empty rectangles. Components are inspected in both polarities. Surviving candidates receive:

score = 0.30*rectangularity + 0.35*min(1, glyph_count/7) + 0.20*max(0,1-|aspect_ratio-3.8|/3.8) + 0.15*min(1, edge_density/0.16).

The score is heuristic. It is not a calibrated confidence or evidence of a valid plate format. Non-maximum suppression removes overlap. Crops, binary masks, perimeter, area, rectangularity and aspect ratio are exported. Default aspect ratio is 1.8–6.8, favouring single-line plates. Country-specific and two-line plates require tuning/extension. A sign or other text panel can still pass these tests.

Optional OCR passes the crop to local Tesseract with single-line segmentation and an alphanumeric whitelist. Reading text is separate from localising a region. Wrong OCR must never be treated as a confirmed registration.

## Object/background edge separation

The union of detected person and plate bounding rectangles selects Object edges; its complement selects Surrounding edges. Their union equals the active binary edge map. This produces an explainable region view but does not isolate an exact human outline and inherits detector mistakes.

## Reference-frame change detection

Blurred raw grayscale frames from the same fixed viewpoint undergo absolute subtraction. Pixels above a threshold are foreground candidates; opening and closing remove isolated noise and join fragments. Connected components below the area threshold are discarded. Changed percentage uses retained pixels / analysis image pixels. It is not person-specific motion: illumination variation, shadows and camera shake also trigger it. Raw grayscale is used here to avoid per-frame CLAHE changes causing extra differences.

## Measurement and evaluation

Brightness = grayscale mean; contrast = grayscale standard deviation; edge coverage = nonzero binary pixels / all ROI pixels. These describe processing and do not prove improved recognition. Processing milliseconds use a monotonic clock around the core pipeline, excluding upload, UI, ZIP generation and optional OCR.

Evaluate a fixed parameter configuration on a held-out labelled set. Use one-to-one IoU matching for each class, aggregate TP/FP/FN, and compute precision, recall, F1 and count MAE. Include negative scenes. Compare baseline gamma=1, CLAHE=0, illumination off to the night setting on the same labelled set. Never select the best parameters using the held-out results; tune on development images and lock settings before the final test. No perfect-detection claim is warranted without these measurements.
