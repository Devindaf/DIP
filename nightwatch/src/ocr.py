"""Optional local Tesseract integration. Nothing is sent to a remote service."""
import shutil
import subprocess
import tempfile
from pathlib import Path
from .exporting import png


def available():
    return shutil.which('tesseract') is not None


def read_plate(image):
    executable=shutil.which('tesseract')
    if not executable:
        return {'text':'','status':'Tesseract is not installed; plate location and crops still work.'}
    with tempfile.TemporaryDirectory() as folder:
        path=Path(folder)/'plate.png'
        path.write_bytes(png(image))
        try:
            result=subprocess.run([executable,str(path),'stdout','--psm','7','-l','eng',
                '-c','tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-'],
                capture_output=True,text=True,timeout=12,check=False)
            if result.returncode:
                return {'text':'','status':'OCR failed; check that Tesseract English language data is installed.'}
            return {'text':result.stdout.strip(),'status':'OCR suggestion; verify against the crop.'}
        except (subprocess.TimeoutExpired,OSError):
            return {'text':'','status':'OCR timed out or could not start.'}
