"""One-to-one bounding box matching at IoU >= threshold.

Annotations: {"filename.png":{"person":[[x,y,w,h]],"plate":[...]}}.
Coordinates refer to each result's analysis image, not the original upload.
"""
from .pipeline import iou


def validate_annotations(data, results):
    if not isinstance(data,dict):
        raise ValueError('Annotations must be a JSON object keyed by exact filename.')
    names = [name for name,_ in results]
    if len(names) != len(set(names)):
        raise ValueError('Evaluation needs unique image filenames.')
    if set(data) != set(names):
        raise ValueError('Annotation filenames must exactly match the processed images. Include empty lists for negative images.')
    for name,result in results:
        entry = data[name]
        if not isinstance(entry,dict) or not all(k in entry for k in ('person','plate')):
            raise ValueError(f'{name}: include both person and plate lists.')
        width,height = result['analysis_size']
        for label in ('person','plate'):
            if not isinstance(entry[label],list):
                raise ValueError(f'{name}: {label} must be a list of boxes.')
            for box in entry[label]:
                if not isinstance(box,list) or len(box)!=4 or any(type(v) not in (int,float) for v in box):
                    raise ValueError(f'{name}: boxes must contain four numeric values [x,y,width,height].')
                x,y,w,h = box
                if not (0<=x<width and 0<=y<height and w>0 and h>0 and x+w<=width and y+h<=height):
                    raise ValueError(f'{name}: box is outside the analysis image.')


def evaluate(results, annotations, threshold=.5):
    validate_annotations(annotations,results)
    rows=[]
    for label,key in [('person','people'),('plate','plates')]:
        tp=fp=fn=0
        error=0
        for name,result in results:
            gt=annotations[name][label]
            pred=sorted(result[key],key=lambda x:x['score'],reverse=True)
            matched=set()
            for item in pred:
                choices=[(iou(item['box'],box),idx) for idx,box in enumerate(gt) if idx not in matched]
                best=max(choices,default=(0,None))
                if best[0]>=threshold:
                    tp+=1
                    matched.add(best[1])
                else:
                    fp+=1
            fn+=len(gt)-len(matched)
            error+=abs(len(pred)-len(gt))
        precision=tp/(tp+fp) if tp+fp else 0.
        recall=tp/(tp+fn) if tp+fn else 0.
        rows.append(dict(class_name=label,tp=tp,fp=fp,fn=fn,precision=precision,
            recall=recall,f1=2*precision*recall/(precision+recall) if precision+recall else 0.,
            count_mae=error/len(results) if results else 0.,images=len(results),iou_threshold=threshold))
    return rows
