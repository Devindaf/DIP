import io
import json
import zipfile
from dataclasses import replace
import cv2
import numpy as np
import pytest
from src.pipeline import Settings, decode_image, process, iou
from src.exporting import png, export_zip
from src.evaluation import evaluate
from scripts.make_demo import create


def test_plate_pipeline_finds_synthetic_region():
    image=create()
    result=process(image,Settings(humans=False))
    assert any(iou(p['box'],[270,285,266,68])>.55 for p in result['plates'])
    assert result['metrics']['brightness_after']>result['metrics']['brightness_before']
    assert all(v.shape[:2]==image.shape[:2] for v in result['images'].values())
    assert np.array_equal(cv2.bitwise_or(result['images']['Object edges'],result['images']['Surrounding edges']),result['images']['Binary edges'])


def test_blank_has_no_targets_and_export_is_complete():
    result=process(np.full((240,320,3),30,np.uint8))
    assert not result['people'] and not result['plates']
    with zipfile.ZipFile(io.BytesIO(export_zip([('blank.png',result)]))) as archive:
        assert 'summary.csv' in archive.namelist()
        data=json.loads(archive.read('001_blank/analysis.json'))
        assert data['settings']['gamma']==.65
        assert data['analysis_size']==[320,240]
        assert '001_blank/Detections.png' in archive.namelist()


def test_decode_metadata_invalid_and_16bit():
    gray=np.full((64,80),100,np.uint8)
    im,meta=decode_image(png(gray))
    assert im.shape==(64,80,3) and meta['channels']==1
    im,meta=decode_image(png(np.full((64,80),32768,np.uint16)))
    assert meta['bit_depth']==16 and im.dtype==np.uint8
    with pytest.raises(ValueError): decode_image(b'not an image')
    with pytest.raises(ValueError): decode_image(png(np.zeros((10,10),np.uint8)))


def test_change_detection_and_reference_validation():
    base=np.full((240,320,3),35,np.uint8)
    image=base.copy(); image[50:180,100:170]=150
    result=process(image,Settings(humans=False,plates=False),base)
    assert result['metrics']['changed_regions']==1
    assert 10<result['metrics']['changed_percent']<15
    with pytest.raises(ValueError): process(image,Settings(),np.zeros((80,80,3),np.uint8))


@pytest.mark.parametrize('edge',['Canny','Sobel','Laplacian of Gaussian'])
def test_edges_and_rotation_roi(edge):
    image=create()
    result=process(image,Settings(humans=False,edge=edge,rotation=90,roi=(10,10,90,90)))
    assert result['images']['Binary edges'].dtype==np.uint8
    assert set(np.unique(result['images']['Binary edges'])) <= {0,255}
    assert result['geometry']['rotation']==90
    with pytest.raises(ValueError): process(image,replace(Settings(),roi=(90,0,10,100)))


def test_small_roi_hog_does_not_crash():
    result=process(np.zeros((32,32,3),np.uint8),Settings())
    assert result['people']==[]


def test_evaluation_one_to_one_and_validation():
    r={'analysis_size':[200,200], 'people':[{'box':[10,10,20,40],'score':2.},
       {'box':[10,10,20,40],'score':1.}], 'plates':[]}
    gt={'x.png':{'person':[[10,10,20,40]],'plate':[]}}
    scores=evaluate([('x.png',r)],gt)
    assert scores[0]['tp']==1 and scores[0]['fp']==1 and scores[0]['recall']==1.
    assert scores[0]['precision']==.5
    with pytest.raises(ValueError): evaluate([('x.png',r)],{})
    with pytest.raises(ValueError): evaluate([('x.png',r)],{'x.png':{'person':[[190,0,30,30]],'plate':[]}})


def test_real_development_hog_smoke():
    from pathlib import Path
    path=Path(__file__).resolve().parents[1]/'dataset/development/opencv_basketball1.png'
    if not path.exists(): pytest.skip('Optional upstream development fixture not included.')
    result=process(cv2.imread(str(path)),Settings(plates=False))
    # Approximate manual box for the visible upright person on the left.
    # This is a development smoke check, not a held-out accuracy claim.
    assert any(iou(p['box'],[5,70,180,400])>.5 for p in result['people'])


def test_ocr_missing_is_explicit(monkeypatch):
    from src import ocr
    monkeypatch.setattr(ocr.shutil,'which',lambda _:None)
    result=ocr.read_plate(np.zeros((40,120),np.uint8))
    assert result['text']=='' and 'not installed' in result['status']


def _plate_scene(fill, text):
    scene = np.full((480, 800, 3), 40, np.uint8)
    cv2.rectangle(scene, (100, 100), (700, 380), (60, 55, 50), -1)
    cv2.rectangle(scene, (300, 300), (480, 350), fill, -1)
    cv2.putText(scene, 'WP CAB 4521', (306, 338), cv2.FONT_HERSHEY_SIMPLEX, .9, text, 3, cv2.LINE_AA)
    return scene


@pytest.mark.parametrize('name,fill,text', [('yellow', (0, 200, 255), (10, 10, 10)),
                                            ('green', (60, 160, 40), (255, 255, 255)),
                                            ('red', (30, 30, 200), (255, 255, 255))])
def test_coloured_plates_are_found(name, fill, text):
    result = process(_plate_scene(fill, text), Settings(humans=False))
    assert any(p['colour'] == name and iou(p['box'], [300, 300, 180, 50]) > .6 for p in result['plates'])


def test_blue_sign_with_white_text_is_not_a_plate():
    scene = np.full((480, 800, 3), 40, np.uint8)
    cv2.rectangle(scene, (520, 120), (760, 190), (200, 80, 20), -1)
    cv2.putText(scene, 'HOTEL ROOMS', (530, 170), cv2.FONT_HERSHEY_SIMPLEX, .9, (255, 255, 255), 2, cv2.LINE_AA)
    assert not process(scene, Settings(humans=False))['plates']
