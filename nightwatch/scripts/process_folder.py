"""Offline folder batch processing using the same core as the browser app."""
from pathlib import Path
import argparse
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.pipeline import Settings,decode_image,process
from src.exporting import export_zip


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('folder',type=Path)
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/batch.zip')
    args=parser.parse_args()
    if not args.folder.is_dir(): parser.error('Input must be an existing folder.')
    results=[]; errors=[]
    for path in sorted(args.folder.rglob('*')):
        if path.suffix.lower() not in {'.jpg','.jpeg','.png','.bmp','.tif','.tiff','.webp'}: continue
        name=str(path.relative_to(args.folder))
        try:
            image,_=decode_image(path.read_bytes())
            results.append((name,process(image,Settings())))
        except Exception as exc:
            errors.append(dict(filename=name,error=str(exc)))
    if not results and not errors: parser.error('No supported images found.')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_bytes(export_zip(results,errors))
    print(f'{len(results)} processed, {len(errors)} errors. Saved {args.output}')

if __name__=='__main__': main()
