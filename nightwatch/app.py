from pathlib import Path
import io
import json
import zipfile
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from src.pipeline import Settings, decode_image, process
from src.exporting import png, export_zip, summary_csv
from src.evaluation import evaluate
from src.ocr import available, read_plate

st.set_page_config(page_title='NightWatch | Surveillance Lab',page_icon='◈',layout='wide')
st.markdown('''<style>
.stApp {background:#0b111b} .block-container {padding-top:2rem;max-width:1500px}
h1 {letter-spacing:-1.8px} [data-testid="stMetric"] {background:#141f2f;border:1px solid #27364a;padding:16px;border-radius:12px}
[data-testid="stSidebar"] {background:#101a28} .eyebrow {color:#60d8c4;letter-spacing:3px;font-size:12px;font-weight:700}
.hero-note {color:#a3b3c8;max-width:850px;font-size:17px} .stTabs [data-baseweb="tab-list"] {gap:22px}
</style>''',unsafe_allow_html=True)
st.markdown('<div class="eyebrow">SE3071 · SECURITY / SURVEILLANCE · DOMAIN 06</div>',unsafe_allow_html=True)
st.title('NightWatch Surveillance Lab')
st.markdown('<p class="hero-note">Bring the scene into focus. Enhance low-light frames, inspect edges, locate people and examine number-plate candidates using classical image processing.</p>',unsafe_allow_html=True)


def reset():
    for key in list(st.session_state):
        del st.session_state[key]

with st.sidebar:
    st.header('Processing controls')
    st.button('Reset workspace',on_click=reset,width='stretch')
    preset=st.selectbox('Starting preset',['Night / balanced','Very dark','Daylight'],key='preset')
    default={'Night / balanced':(.65,2.5),'Very dark':(.45,3.0),'Daylight':(1.,1.5)}[preset]
    # Changing the preset updates its two controls while preserving other choices.
    if st.session_state.get('_preset')!=preset:
        st.session_state['gamma']=default[0]
        st.session_state['clahe']=default[1]
        st.session_state['_preset']=preset
    gamma=st.slider('Gamma exponent (below 1 brightens)',.2,2.,step=.05,key='gamma')
    clahe=st.slider('CLAHE contrast limit',0.,5.,step=.25,key='clahe')
    illumination=st.checkbox('Correct uneven illumination',value=True)
    filter_name=st.selectbox('Denoising filter',['Gaussian','Median','Bilateral'])
    kernel=st.select_slider('Filter kernel',options=[3,5,7,9],value=5)
    sigma=st.slider('Gaussian sigma',.5,3.,1.2,.1)
    edge=st.selectbox('Active edge method',['Canny','Sobel','Laplacian of Gaussian'])
    low,high=st.slider('Canny thresholds',0,255,(45,130))
    edge_threshold=st.slider('Sobel / Laplacian binary threshold',1,255,35)
    with st.expander('Geometry and region of interest'):
        max_width=st.select_slider('Maximum image side (pixels)',options=[640,960,1280,1600],value=1280)
        rotation=st.selectbox('Rotate clockwise',[0,90,180,270])
        xr=st.slider('ROI horizontal range (%)',0,100,(0,100))
        yr=st.slider('ROI vertical range (%)',0,100,(0,100))
        st.caption('ROI coordinates apply after rotation and resizing. Detection boxes refer to the cropped analysis image.')
    with st.expander('Detection settings',expanded=True):
        humans=st.checkbox('Detect people (HOG + SVM)',value=True)
        hog=st.slider('Person SVM margin threshold',0.,2.,.4,.1)
        plates=st.checkbox('Locate number-plate candidates',value=True)
        plate_score=st.slider('Plate heuristic score cutoff',.2,.95,.42,.01)
        ocr=st.checkbox('Read plate text with optional Tesseract',value=False)
        if ocr and not available():
            st.info('Tesseract is not installed. You can still inspect plate crops.')
    with st.expander('Change detection'):
        motion_threshold=st.slider('Difference threshold',5,100,25)
        motion_area=st.number_input('Minimum changed area (pixels)',25,10000,150,25)

cfg=Settings(max_width=max_width,rotation=rotation,roi=(xr[0],yr[0],xr[1],yr[1]),
    gamma=gamma,clahe=clahe,illumination=illumination,filter=filter_name,kernel=kernel,sigma=sigma,
    edge=edge,low=low,high=high,edge_threshold=edge_threshold,humans=humans,hog_threshold=hog,
    plates=plates,plate_min_score=plate_score,motion_threshold=motion_threshold,motion_min_area=motion_area)


def show_image(image,caption=None):
    st.image(cv2.cvtColor(image,cv2.COLOR_BGR2RGB) if image.ndim==3 else image,
             caption=caption,width='stretch',clamp=True)


def read_inputs(files,zip_upload):
    entries=[]
    if files:
        entries.extend((f.name,f.getvalue()) for f in files)
    if zip_upload:
        try:
            with zipfile.ZipFile(io.BytesIO(zip_upload.getvalue())) as archive:
                infos=[i for i in archive.infolist() if not i.is_dir()]
                supported=[i for i in infos if Path(i.filename).suffix.lower() in {'.jpg','.jpeg','.png','.bmp','.tif','.tiff','.webp'}]
                if len(supported)>100 or sum(i.file_size for i in supported)>150*1024*1024:
                    raise ValueError('ZIP limit: 100 images and 150 MB uncompressed.')
                for info in supported:
                    if info.file_size>25*1024*1024:
                        raise ValueError(f'{info.filename} exceeds 25 MB.')
                    entries.append((info.filename,archive.read(info)))
                if not supported:
                    raise ValueError('ZIP contains no supported images.')
        except (zipfile.BadZipFile,RuntimeError,OSError,ValueError) as exc:
            raise ValueError(f'Cannot read image ZIP: {exc}') from exc
    if len(entries)>100 or sum(len(data) for _,data in entries)>150*1024*1024:
        raise ValueError('Batch limit: 100 images and 150 MB.')
    if len({name for name,_ in entries})!=len(entries):
        raise ValueError('Each image must have a unique filename; rename duplicates.')
    return entries

workspace,comparison,plates_tab,batch,evaluation,guide=st.tabs(
    ['01  Workspace','02  Pipeline & edges','03  Plate inspection','04  Batch & export','05  Evaluation','06  Method guide'])
with workspace:
    left,right=st.columns([1.4,1])
    with left:
        uploads=st.file_uploader('Upload an image or select multiple images',type=['jpg','jpeg','png','bmp','tif','tiff','webp'],accept_multiple_files=True)
        zipped=st.file_uploader('Or upload a ZIP of an image folder',type=['zip'])
    with right:
        camera=st.checkbox('Use a camera snapshot')
        shot=st.camera_input('Take a snapshot') if camera else None
        reference=st.file_uploader('Optional reference frame for change detection',type=['jpg','jpeg','png','bmp','webp'])
        demo=st.checkbox('Use included synthetic night-plate demo',value=not bool(uploads or zipped or shot))
    if st.button('Analyze images',type='primary',width='stretch'):
        # Clear previous output so errors never leave stale detections on screen.
        st.session_state.pop('results',None)
        st.session_state.pop('result_metas',None)
        try:
            entries=read_inputs(uploads,zipped)
            if shot:
                entries.append(('camera.png',shot.getvalue()))
            if not entries and demo:
                p=Path(__file__).parent/'dataset/synthetic/night_plate.png'
                entries=[(p.name,p.read_bytes())]
            if not entries:
                raise ValueError('Upload images or enable the included demo.')
            ref=None
            if reference:
                ref,_=decode_image(reference.getvalue())
            results=[]; errors=[]; metas={}
            bar=st.progress(0,text='Processing images…')
            with st.spinner('Enhancing and analyzing…'):
                for idx,(name,data) in enumerate(entries):
                    try:
                        image,meta=decode_image(data)
                        result=process(image,cfg,ref)
                        if ocr:
                            for plate in result['plates']:
                                plate['ocr']=read_plate(plate['binary'])
                        results.append((name,result)); metas[name]=meta
                    except (ValueError,cv2.error,OSError) as exc:
                        errors.append(dict(filename=name,error=str(exc)))
                    bar.progress((idx+1)/len(entries),text=f'Processed {idx+1} / {len(entries)}')
            st.session_state['results']=results
            st.session_state['result_metas']=metas
            st.session_state['errors']=errors
        except (ValueError,OSError) as exc:
            st.error(str(exc))
    results=st.session_state.get('results',[])
    for err in st.session_state.get('errors',[]):
        st.error(f"{err['filename']}: {err['error']}")
    if results:
        name=st.selectbox('Image to inspect',[n for n,_ in results])
        result=dict(results)[name]
        if result['settings']!=json.loads(json.dumps(dict(cfg.__dict__))):
            # Tuples are normalised to lists for an accurate comparison.
            if json.dumps(result['settings'],sort_keys=True)!=json.dumps(cfg.__dict__,sort_keys=True):
                st.warning('Controls changed. Click Analyze images to update all results.')
        cols=st.columns(4)
        for col,label,value in zip(cols,['People detected','Plate candidates','Edge coverage','Processing time'],
                [result['metrics']['people'],result['metrics']['plate_candidates'],f"{result['metrics']['edge_percent']:.1f}%",f"{result['metrics']['processing_ms']:.0f} ms"]):
            col.metric(label,value)
        a,b=st.columns(2)
        with a: show_image(result['images']['Original ROI'],'Original · transformed ROI')
        with b: show_image(result['images']['Detections'],'Enhanced · detected regions')
        st.caption('Green: HOG person detections · amber: plate candidates · blue: changed regions. Candidate scores are not probabilities.')
        with st.expander('Image metadata and coordinate system'):
            st.json(dict(source=st.session_state['result_metas'][name],geometry=result['geometry'],analysis_size=result['analysis_size']))
        st.download_button('Download this image analysis ZIP',export_zip([(name,result)]),file_name='nightwatch_image.zip',mime='application/zip')
    else:
        result=None
        st.info('Choose an image and click Analyze images. The included demo is synthetic and tests the plate pipeline; it is not real-world accuracy evidence.')

with comparison:
    if result:
        st.subheader('Follow the image through the pipeline')
        names=['Grayscale','Enhanced','Filter - Gaussian','Filter - Median','Filter - Bilateral','Canny','Sobel','Laplacian of Gaussian','Plate grouping']
        for offset in range(0,len(names),3):
            for col,title in zip(st.columns(3),names[offset:offset+3]):
                with col: show_image(result['images'][title],title)
        st.subheader('Object regions and the surrounding scene')
        for col,title in zip(st.columns(3),['Binary edges','Object edges','Surrounding edges']):
            with col: show_image(result['images'][title],title)
        st.caption('Object separation uses the detected bounding boxes. Missed objects remain in surrounding edges; this is not a silhouette segmentation.')
        st.dataframe(pd.DataFrame([result['metrics']]),width='stretch')
        if np.any(result['images']['Change mask']):
            show_image(result['images']['Change mask'],'Changed pixels relative to the reference')
        st.caption('Higher contrast or edge coverage does not by itself establish better recognition. Compare detection precision and recall on labelled images.')
    else: st.info('Analyze an image first.')

with plates_tab:
    if result:
        st.subheader('Examine number-plate regions')
        if not result['plates']:
            st.info('No plate candidates passed the filters. Crop closer, compare edge methods or reduce the score cutoff. A missed plate does not prove it is absent.')
        for idx,plate in enumerate(result['plates']):
            st.markdown(f"**Candidate {idx+1}** · heuristic score {plate['score']:.2f} · {plate['characters']} character-like components")
            a,b=st.columns(2)
            with a: show_image(plate['crop'],'Rectified and enlarged crop')
            with b: show_image(plate['binary'],'Selected character mask (Otsu / adaptive)')
            st.write('Box [x, y, width, height]:',plate['box'])
            if 'ocr' in plate:
                st.write('Suggested text:',plate['ocr']['text'] or '(none)')
                st.caption(plate['ocr']['status'])
            st.download_button('Download plate crop',png(plate['crop']),f'plate_{idx+1}.png','image/png',key=f'plate_{idx}')
        st.caption('Enhancement cannot restore characters absent from an underexposed, saturated, tiny or motion-blurred source.')
    else: st.info('Analyze an image first.')

with batch:
    st.subheader('Batch results and reproducible exports')
    st.write('Select several images, or ZIP an image folder and upload it in Workspace. All images use the same settings. Invalid files are logged and skipped.')
    if results:
        rows=[dict(filename=n,**r['metrics']) for n,r in results]
        st.dataframe(pd.DataFrame(rows),width='stretch')
        st.download_button('Download complete results ZIP',export_zip(results,st.session_state.get('errors')),file_name='nightwatch_batch.zip',mime='application/zip')
        st.download_button('Download summary CSV',summary_csv(rows),file_name='summary.csv',mime='text/csv')
        st.caption('ZIP contains every pipeline view, plate crops, settings, coordinate metadata, detections, summary.csv and any errors.')
    else: st.info('No batch results yet.')

with evaluation:
    st.subheader('Measure detection performance')
    st.write('Use at least 10 unseen, manually labelled images. Upload annotations for exactly the processed batch. Boxes must be drawn on Original ROI after its rotation, resize and crop.')
    st.code('{"scene.png": {"person": [[40, 20, 70, 180]], "plate": [[160, 230, 120, 35]]},\n "empty.png": {"person": [], "plate": []}}',language='json')
    if results:
        template={n:{'person':[],'plate':[]} for n,_ in results}
        st.download_button('Download blank annotation template',json.dumps(template,indent=2),'annotations.json','application/json')
        st.caption('Empty lists are assertions that no target exists. Replace them with manually measured boxes wherever targets are present.')
        # Original ROI with coordinate axes, allowing manual box annotation without another app.
        with st.expander('View image with annotation coordinates'):
            from PIL import Image,ImageDraw
            pick=st.selectbox('Annotation image',[n for n,_ in results],key='annotate')
            r=dict(results)[pick]
            canvas=Image.fromarray(cv2.cvtColor(r['images']['Original ROI'],cv2.COLOR_BGR2RGB))
            draw=ImageDraw.Draw(canvas)
            for x in range(0,canvas.width,50):
                draw.line((x,0,x,canvas.height),fill=(80,180,180),width=1)
                draw.text((x+2,2),str(x),fill=(255,255,0))
            for y in range(50,canvas.height,50):
                draw.line((0,y,canvas.width,y),fill=(80,180,180),width=1)
                draw.text((2,y+2),str(y),fill=(255,255,0))
            st.image(canvas,caption='Grid spacing: 50 analysis pixels',width='stretch')
        annotation=st.file_uploader('Upload completed ground-truth JSON',type=['json'])
        threshold=st.slider('Matching IoU threshold',.1,.9,.5,.05)
        if st.button('Evaluate labelled batch'):
            try:
                if annotation is None:
                    raise ValueError('Upload completed annotations first.')
                scores=evaluate(results,json.loads(annotation.getvalue()),threshold)
                st.dataframe(pd.DataFrame(scores),width='stretch')
                st.download_button('Download evaluation CSV',summary_csv(scores),'evaluation.csv','text/csv')
                if len(results)<10: st.warning('This batch has fewer than the 10 unseen images required by the brief.')
            except (ValueError,TypeError,KeyError) as exc:
                st.error(f'Invalid annotations: {exc}')
    else: st.info('Process the evaluation images first.')

with guide:
    st.subheader('How the system makes its decisions')
    st.markdown('''1. **Validate and transform:** inspect real file contents, record metadata, rotate, resize and crop the ROI.
2. **Enhance:** use Lab luminance, divide by a smooth illumination estimate, apply gamma correction and CLAHE; convert to grayscale.
3. **Denoise:** compare Gaussian (weighted smoothing), median (impulse noise) and bilateral (edge-preserving) filters. The selected output feeds analysis.
4. **Extract edges:** Canny uses two thresholds; Sobel measures gradient magnitude; Gaussian + Laplacian measures a second derivative. The selected map feeds plate grouping and context separation.
5. **Locate targets:** gradient-based HOG + a bundled linear SVM proposes upright people. Sobel/selected edges, morphological closing, Otsu/adaptive segmentation, contours, rectangularity and character components propose plates.
6. **Inspect and measure:** non-maximum suppression removes overlapping boxes. Rectification straightens plate candidates; optional local OCR suggests text. A same-view reference enables thresholded difference regions.
7. **Evaluate and export:** match detections against manual boxes and export precision, recall, F1, count MAE and processing time.''')
    st.warning('This classical prototype can miss small, occluded or unusual poses and can misclassify fences, signs and text. Dark, blurred or glaring plates may remain unreadable. Change detection also responds to lighting or camera movement.')
    st.write('Build your own 30-image dataset including at least 10 student-captured images. Keep development and unseen images separate. Synthetic demos are only functional checks. Read README.md and docs/METHODOLOGY.md for setup and viva explanations.')
    st.caption('Local processing · no facial recognition · no cloud inference · no model downloads required')
