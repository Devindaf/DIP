# Technical report outline (complete with your own evidence)

Cover (not counted): title, full IT number, student name, module, date.

1. **Problem and success criteria (1 page):** scope, need for low-light enhancement, specific target outputs, criteria declared before evaluation.
2. **Dataset (1 page):** 30+ images, sources, 10+ own captures, resolutions, lighting and format variation; development/unseen split and provenance table.
3. **Pipeline (1 page):** acquisition → geometry → Lab illumination/gamma/CLAHE → grayscale filtering → edges → target analysis → metrics/export.
4. **Enhancement/filter experiments (1 page):** explain gamma equation, illumination formula and CLAHE; compare filters on noise and detail, include screenshots.
5. **Edge and target algorithms (2 pages):** explain Sobel/Canny/LoG, morphology, contours, plate score, connected components, HOG and its limits.
6. **Interface and output (1 page):** original/enhanced preview, plate crop, ROI, surrounding edges and export screenshots.
7. **Evaluation (1 page):** manual annotations, IoU rule, precision/recall/F1/count MAE and measured runtime on >=10 unseen images. State hardware, parameters and sample size.
8. **Results and failure cases (1 page):** actual baseline/night comparison, false positives and misses; avoid using only successful examples.
9. **Conclusion and limitations (1 page):** supported findings, scope limits and practical extensions.

References and code appendix are outside the main 10-page limit. Do not fill the results section with made-up values. Disclose AI assistance and identify external detector coefficients and sample image sources.
