import csv
import io
import json
import re
import zipfile
import cv2
import numpy as np


def png(image):
    ok, encoded = cv2.imencode('.png', image)
    if not ok:
        raise ValueError('Could not encode image.')
    return encoded.tobytes()


def serializable(result):
    def clean(value):
        if isinstance(value, np.ndarray):
            return None
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {k:clean(v) for k,v in value.items() if k not in {'images','crop','binary'}}
        if isinstance(value, (list,tuple)):
            return [clean(v) for v in value]
        return value
    return clean(result)


def summary_csv(rows):
    stream = io.StringIO(newline='')
    keys = sorted(set().union(*(row.keys() for row in rows))) if rows else ['filename']
    writer = csv.DictWriter(stream, fieldnames=keys)
    writer.writeheader()
    for row in rows:
        # Spreadsheet formula injection protection for uploaded filenames/text.
        writer.writerow({k: ("'"+v if isinstance(v,str) and v.startswith(('=','+','-','@')) else v)
                         for k,v in row.items()})
    return stream.getvalue().encode('utf-8-sig')


def export_zip(results, errors=None):
    buffer = io.BytesIO()
    rows = []
    with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED) as archive:
        for idx,(name,result) in enumerate(results):
            stem = re.sub(r'[^A-Za-z0-9_-]','_',name.rsplit('.',1)[0])[:70] or 'image'
            folder = f'{idx+1:03d}_{stem}'
            for title,image in result['images'].items():
                archive.writestr(f'{folder}/{title.replace(" ","_")}.png',png(image))
            for j,plate in enumerate(result['plates']):
                archive.writestr(f'{folder}/plate_{j+1}_crop.png',png(plate['crop']))
                archive.writestr(f'{folder}/plate_{j+1}_binary.png',png(plate['binary']))
            archive.writestr(f'{folder}/analysis.json',json.dumps(serializable(result),indent=2))
            rows.append(dict(filename=name,**result['metrics']))
        archive.writestr('summary.csv',summary_csv(rows))
        if errors:
            archive.writestr('errors.json',json.dumps(errors,indent=2))
    return buffer.getvalue()
