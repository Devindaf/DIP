"""Explainable OpenCV pipeline. Boxes are in the transformed analysis image."""
from dataclasses import dataclass, asdict
import time
import cv2
import numpy as np

@dataclass(frozen=True)
class Settings:
    max_width: int = 1280
    rotation: int = 0
    roi: tuple = (0, 0, 100, 100)  # percentages, after rotation/resize
    gamma: float = 0.65  # y = 255 * (x / 255)^gamma; <1 brightens
    clahe: float = 2.5
    illumination: bool = True
    filter: str = 'Gaussian'
    kernel: int = 5
    sigma: float = 1.2
    edge: str = 'Canny'
    low: int = 45
    high: int = 130
    edge_threshold: int = 35
    humans: bool = True
    hog_threshold: float = 0.5
    person_min_height: float = 0.07  # fraction of analysis height
    plates: bool = True
    plate_min_score: float = 0.55
    motion_threshold: int = 25
    motion_min_area: int = 150


def decode_image(data: bytes):
    """Decode actual bytes, not the extension. Bound memory before OpenCV decode."""
    from PIL import Image
    import io
    if not data or len(data) > 25 * 1024 * 1024:
        raise ValueError('Image is empty or exceeds the 25 MB limit.')
    try:
        with Image.open(io.BytesIO(data)) as probe:
            w, h = probe.size
            if w * h > 24_000_000 or min(w, h) < 16:
                raise ValueError('Use an image of at least 16x16 and at most 24 megapixels.')
            if probe.format not in {'JPEG', 'PNG', 'BMP', 'TIFF', 'WEBP'}:
                raise ValueError('Unsupported image format.')
            probe.verify()
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError('Unsupported or corrupted image.') from exc
    raw = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise ValueError('OpenCV could not decode this image.')
    channels = 1 if raw.ndim == 2 else raw.shape[2]
    meta = dict(width=w, height=h, channels=channels, bit_depth=raw.dtype.itemsize*8,
                file_bytes=len(data))
    if raw.dtype == np.uint16:
        raw = (raw / 257).astype(np.uint8)
    if raw.dtype != np.uint8:
        raise ValueError('Only 8-bit and 16-bit integer images are supported.')
    if raw.ndim == 2:
        raw = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
    elif raw.shape[2] == 4:
        a = raw[:, :, 3:4].astype(np.float32) / 255
        raw = np.uint8(raw[:, :, :3] * a + 255 * (1 - a))
    return raw, meta


def transform(image, cfg):
    if cfg.rotation not in (0, 90, 180, 270):
        raise ValueError('Rotation must be 0, 90, 180, or 270.')
    if cfg.rotation:
        codes = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180,
                 270: cv2.ROTATE_90_COUNTERCLOCKWISE}
        image = cv2.rotate(image, codes[cfg.rotation])
    h, w = image.shape[:2]
    # Limit both dimensions; avoids an unbounded tall-image pipeline.
    scale = min(1., cfg.max_width / max(w, h))
    if scale < 1:
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    h, w = image.shape[:2]
    x1, y1, x2, y2 = cfg.roi
    if not (0 <= x1 < x2 <= 100 and 0 <= y1 < y2 <= 100):
        raise ValueError('ROI left/top must be smaller than right/bottom.')
    a, b = int(w*x1/100), int(h*y1/100)
    c, d = max(a+1, int(w*x2/100)), max(b+1, int(h*y2/100))
    crop = image[b:d, a:c].copy()
    if min(crop.shape[:2]) < 16:
        raise ValueError('Selected ROI is too small; select at least 16x16 pixels.')
    return crop, dict(rotation=cfg.rotation, resize_scale=scale,
                      crop_xyxy=[a, b, c, d], coordinate_space='analysis image, after rotation, resize and crop')


def enhance(image, cfg):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lum, a, b = cv2.split(lab)
    if cfg.illumination:
        # Division by a smooth illumination field, normalised to its mean.
        field = cv2.GaussianBlur(lum.astype(np.float32), (0, 0), 25)
        lum = np.uint8(np.clip(lum.astype(np.float32) * (field.mean()+1) / (field+1), 0, 255))
    lut = np.uint8(np.clip(255*(np.arange(256)/255.)**cfg.gamma, 0, 255))
    lum = cv2.LUT(lum, lut)
    if cfg.clahe > 0:
        lum = cv2.createCLAHE(clipLimit=cfg.clahe, tileGridSize=(8, 8)).apply(lum)
    color = cv2.cvtColor(cv2.merge((lum, a, b)), cv2.COLOR_LAB2BGR)
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    filters = {
        'Gaussian': cv2.GaussianBlur(gray, (cfg.kernel, cfg.kernel), cfg.sigma),
        'Median': cv2.medianBlur(gray, cfg.kernel),
        'Bilateral': cv2.bilateralFilter(gray, cfg.kernel, 50, 50),
    }
    return color, gray, filters


def edges(gray, cfg):
    # Gaussian smoothing is always explicit for Laplacian of Gaussian.
    log_source = cv2.GaussianBlur(gray, (cfg.kernel, cfg.kernel), cfg.sigma)
    lap = cv2.convertScaleAbs(cv2.Laplacian(log_source, cv2.CV_64F, ksize=3))
    sx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    sobel = cv2.convertScaleAbs(cv2.magnitude(sx, sy))
    canny = cv2.Canny(gray, cfg.low, cfg.high, L2gradient=True)
    return {'Canny': canny, 'Sobel': sobel, 'Laplacian of Gaussian': lap}


def iou(a, b):
    x, y = max(a[0], b[0]), max(a[1], b[1])
    xx, yy = min(a[0]+a[2], b[0]+b[2]), min(a[1]+a[3], b[1]+b[3])
    inter = max(0, xx-x)*max(0, yy-y)
    return inter / max(1, a[2]*a[3] + b[2]*b[3] - inter)


def nms(items, threshold=0.35):
    kept = []
    for item in sorted(items, key=lambda v: v['score'], reverse=True):
        if all(iou(item['box'], k['box']) < threshold for k in kept):
            kept.append(item)
    return kept


def contained(inner, outer, fraction=0.85):
    ix = max(0, min(inner[0]+inner[2], outer[0]+outer[2]) - max(inner[0], outer[0]))
    iy = max(0, min(inner[1]+inner[3], outer[1]+outer[3]) - max(inner[1], outer[1]))
    return ix*iy >= fraction*max(1, inner[2]*inner[3])


def drop_nested_people(items):
    """Remove a smaller box lying inside a larger one unless it is clearly the stronger detection."""
    kept = []
    for a in items:
        nested = any(b is not a and b['box'][2]*b['box'][3] > 1.4*a['box'][2]*a['box'][3]
                     and contained(a['box'], b['box']) and b['score'] >= .5*a['score']
                     for b in items)
        if not nested:
            kept.append(a)
    return kept


def detect_people(gray, threshold, min_height=0.07):
    """Classical gradient descriptors + bundled linear SVM; no neural model."""
    h, w = gray.shape
    if w < 64 or h < 128:
        return []
    scale = min(1., 800/max(w, h))
    small = cv2.resize(gray, None, fx=scale, fy=scale) if scale < 1 else gray
    if small.shape[1] < 64 or small.shape[0] < 128:
        return []
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    boxes, weights = hog.detectMultiScale(small, hitThreshold=threshold,
        winStride=(8, 8), padding=(8, 8), scale=1.05, useMeanshiftGrouping=False)
    out = []
    for box, score in zip(boxes, np.asarray(weights).reshape(-1)):
        x, y, bw, bh = [int(round(v/scale)) for v in box]
        x, y = max(0, x), max(0, y)
        bw, bh = min(bw, w-x), min(bh, h-y)
        # Upright pedestrians are roughly 1:2; reject flat or very tall regions and tiny boxes.
        if bw < 1 or bh < min_height*h or not 1.3 <= bh/bw <= 3.6:
            continue
        out.append(dict(box=[x, y, bw, bh],
                        score=float(score), label='person', method='HOG + linear SVM'))
    return nms(out)


def rectified_crop(gray, contour):
    rect = cv2.minAreaRect(contour)
    pts = cv2.boxPoints(rect).astype(np.float32)
    # Order corners around centroid; rotate to top-left starting point.
    center = pts.mean(axis=0)
    pts = pts[np.argsort(np.arctan2(pts[:, 1]-center[1], pts[:, 0]-center[0]))]
    pts = np.roll(pts, -int(np.argmin(pts.sum(axis=1))), axis=0)
    width = max(1, int(np.linalg.norm(pts[1]-pts[0])))
    height = max(1, int(np.linalg.norm(pts[2]-pts[1])))
    dst = np.float32([[0,0],[width-1,0],[width-1,height-1],[0,height-1]])
    crop = cv2.warpPerspective(gray, cv2.getPerspectiveTransform(pts, dst), (width, height))
    if height > width:
        crop = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
    return crop


def character_count(binary):
    h, w = binary.shape
    count, _, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    return sum(1 for x, y, bw, bh, area in stats[1:count]
               if .28*h < bh < .96*h and .01*w < bw < .19*w
               and area > .001*h*w and bh/max(1,bw) > 1.05)


def plate_colour_masks(bgr):
    """Masks for the plate backgrounds in use: yellow, white, green and red."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, sat, val = [c.astype(np.int16) for c in cv2.split(hsv)]
    masks = {
        'yellow': (h >= 15) & (h <= 35) & (sat >= 80) & (val >= 90),
        'white': (sat <= 45) & (val >= 150),
        'green': (h >= 38) & (h <= 90) & (sat >= 60) & (val >= 50),
        'red': ((h <= 8) | (h >= 172)) & (sat >= 110) & (val >= 70),
    }
    return {k: np.uint8(v)*255 for k, v in masks.items()}


def drop_containers(items):
    """A box enclosing another candidate is usually bumper/body, keep the tight plate."""
    return [a for a in items
            if not any(b is not a and a['box'][2]*a['box'][3] > 1.6*b['box'][2]*b['box'][3]
                       and contained(b['box'], a['box']) for b in items)]


def detect_plates(gray, edge_binary, cfg, detail_gray=None, color=None):
    """Propose regions from edge grouping, bright regions and plate-coloured regions.

    Candidate score is a heuristic, never a probability or OCR confidence.
    """
    h, w = gray.shape
    sx = cv2.convertScaleAbs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
    _, gx = cv2.threshold(sx, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    evidence = cv2.bitwise_or(gx, edge_binary)
    group = cv2.morphologyEx(evidence, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (17, 5)))
    group = cv2.morphologyEx(group, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
    _, bright = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, np.ones((3,7), np.uint8))
    colour_masks = plate_colour_masks(color) if color is not None else {}
    saturated = (cv2.cvtColor(color, cv2.COLOR_BGR2HSV)[:, :, 1] >= 110) if color is not None else None
    sources = [group, bright]
    # Close the dark characters inside each coloured plate so the plate becomes one region.
    close_k = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 7))
    for mask in colour_masks.values():
        sources.append(cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_k))
    contours = []
    for mask in sources:
        cs, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours.extend(cs)
    candidates = []
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:500]:
        x, y, bw, bh = cv2.boundingRect(contour)
        ratio, area = bw/max(1,bh), bw*bh
        if not (1.3 < ratio < 6.8 and bw >= 40 and bh >= 12
                and bw <= .45*w and .0005*w*h < area < .10*w*h):
            continue
        rectangularity = cv2.contourArea(contour)/area
        if rectangularity < .35:
            continue
        colour_name, fill = 'unknown', 0.
        if colour_masks:
            fills = {k: np.count_nonzero(m[y:y+bh, x:x+bw])/area for k, m in colour_masks.items()}
            colour_name, fill = max(fills.items(), key=lambda kv: kv[1])
            # Plate background must dominate the box (dark characters take the remainder).
            if fill < (.30 if colour_name == 'white' else .25):
                continue
            # Squarer two-line plates (motorcycles, three-wheelers) need a stronger colour match.
            if ratio < 1.8 and fill < .45:
                continue
            # White plates have dark characters on a colourless field; coloured signs fail this.
            if colour_name == 'white' and np.count_nonzero(saturated[y:y+bh, x:x+bw])/area > .15:
                continue
        crop = rectified_crop(detail_gray if detail_gray is not None else gray, contour)
        if min(crop.shape) < 8:
            continue
        crop = cv2.resize(crop, (max(1,int(crop.shape[1]*80/crop.shape[0])),80), interpolation=cv2.INTER_CUBIC)
        crop = cv2.createCLAHE(2., (4,4)).apply(crop)
        _, otsu = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        adaptive = cv2.adaptiveThreshold(crop, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY_INV, 25, 8)
        # Keep thin glyph strokes from disappearing into a smoothed local background.
        variants = [(mask, character_count(mask)) for mask in (otsu,255-otsu,adaptive,255-adaptive)]
        plausible = [(mask,count) for mask,count in variants if 3 <= count <= 14]
        binary, chars = max(plausible or variants, key=lambda pair: pair[1])
        density = np.count_nonzero(edge_binary[y:y+bh,x:x+bw])/area
        min_chars = 4 if colour_name == 'white' else 3
        if chars < min_chars or not .025 < density < .75:
            continue
        colour_score = min(1., fill/.6) * (.85 if colour_name == 'white' else 1.)
        if colour_masks:
            score = (.22*min(1, rectangularity) + .28*min(1,chars/7)
                     + .15*max(0,1-abs(ratio-3.8)/3.8) + .10*min(1,density/.16)
                     + .25*colour_score)
        else:
            score = (.30*min(1, rectangularity) + .35*min(1,chars/7)
                     + .20*max(0,1-abs(ratio-3.8)/3.8) + .15*min(1,density/.16))
        if score >= cfg.plate_min_score:
            candidates.append(dict(box=[x,y,bw,bh], score=float(score), label='plate-'+colour_name,
                colour=colour_name, colour_fill=float(fill),
                characters=int(chars), crop=crop, binary=binary,
                perimeter=float(cv2.arcLength(contour,True)), area=int(area),
                rectangularity=float(rectangularity), aspect_ratio=float(ratio)))
    return drop_containers(nms(candidates))[:12], group


def compute_motion(gray, ref, cfg):
    if ref.shape != gray.shape:
        raise ValueError('Reference and current image must have the same transformed dimensions.')
    delta = cv2.absdiff(cv2.GaussianBlur(gray,(5,5),0), cv2.GaussianBlur(ref,(5,5),0))
    _, mask = cv2.threshold(delta, cfg.motion_threshold, 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3),np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7,7),np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask,8)
    clean = np.zeros_like(mask)
    regions = []
    for idx in range(1,count):
        x,y,w,h,area = stats[idx]
        if area >= cfg.motion_min_area:
            clean[labels==idx] = 255
            regions.append(dict(box=[int(x),int(y),int(w),int(h)], label='change', score=1.))
    return clean, regions


def process(image, cfg=Settings(), reference=None):
    start = time.perf_counter()
    if cfg.kernel not in (3,5,7,9) or not 0 < cfg.gamma <= 3:
        raise ValueError('Invalid filter kernel or gamma.')
    if not 0 <= cfg.low < cfg.high <= 255:
        raise ValueError('Canny lower threshold must be smaller than upper threshold.')
    analysis, geometry = transform(image,cfg)
    original_gray = cv2.cvtColor(analysis,cv2.COLOR_BGR2GRAY)
    color, gray, filters = enhance(analysis,cfg)
    selected = filters[cfg.filter]
    maps = edges(selected,cfg)
    active = maps[cfg.edge]
    if cfg.edge == 'Canny':
        edge_binary = active.copy()
    else:
        _, edge_binary = cv2.threshold(active,cfg.edge_threshold,255,cv2.THRESH_BINARY)
    # Compare original and enhanced paths: aggressive night processing can weaken HOG gradients.
    people = drop_nested_people(nms(detect_people(original_gray,cfg.hog_threshold,cfg.person_min_height) +
                 detect_people(selected,cfg.hog_threshold,cfg.person_min_height))) if cfg.humans else []
    plates, group = detect_plates(selected,edge_binary,cfg,gray,color) if cfg.plates else ([],np.zeros_like(selected))
    person_mask = np.zeros_like(selected)
    plate_mask = np.zeros_like(selected)
    for collection, mask in ((people,person_mask),(plates,plate_mask)):
        for item in collection:
            x,y,w,h = item['box']
            mask[y:y+h,x:x+w] = 255
    # Context separation is box-based, not a pixel-accurate person segmentation.
    objects = cv2.bitwise_or(person_mask,plate_mask)
    context_edges = cv2.bitwise_and(edge_binary,cv2.bitwise_not(objects))
    object_edges = cv2.bitwise_and(edge_binary,objects)
    motion_mask, changes = np.zeros_like(selected), []
    if reference is not None:
        ref_crop,_ = transform(reference,cfg)
        # Raw grayscale differences avoid frame-wise CLAHE producing spurious changes.
        motion_mask,changes = compute_motion(original_gray,cv2.cvtColor(ref_crop,cv2.COLOR_BGR2GRAY),cfg)
    overlay = color.copy()
    for items, box_color in ((people,(100,230,80)),(plates,(20,190,255)),(changes,(255,190,60))):
        for item in items:
            x,y,w,h = item['box']
            cv2.rectangle(overlay,(x,y),(x+w-1,y+h-1),box_color,2)
            cv2.putText(overlay,item['label'],(x,max(14,y-6)),cv2.FONT_HERSHEY_SIMPLEX,.48,box_color,1,cv2.LINE_AA)
    metrics = dict(people=len(people),plate_candidates=len(plates),changed_regions=len(changes),
        changed_percent=float(np.count_nonzero(motion_mask)/motion_mask.size*100),
        brightness_before=float(original_gray.mean()),brightness_after=float(gray.mean()),
        contrast_before=float(original_gray.std()),contrast_after=float(gray.std()),
        edge_percent=float(np.count_nonzero(edge_binary)/edge_binary.size*100),
        processing_ms=round((time.perf_counter()-start)*1000,2))
    images = {'Original ROI':analysis,'Enhanced':color,'Grayscale':gray,
        **{'Filter - '+k:v for k,v in filters.items()},**maps,
        'Binary edges':edge_binary,'Plate grouping':group,'Object edges':object_edges,
        'Surrounding edges':context_edges,'Change mask':motion_mask,'Detections':overlay}
    return dict(images=images, people=people, plates=plates, changes=changes,
                metrics=metrics, settings=asdict(cfg), geometry=geometry,
                analysis_size=[analysis.shape[1],analysis.shape[0]])
