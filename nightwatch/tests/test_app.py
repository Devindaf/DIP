from pathlib import Path
from streamlit.testing.v1 import AppTest

APP=Path(__file__).resolve().parents[1]/'app.py'


def test_browser_demo_pipeline_and_reset():
    app=AppTest.from_file(str(APP),default_timeout=45).run()
    assert not app.exception
    analyze=next(b for b in app.button if b.label=='Analyze images')
    analyze.click().run()
    assert not app.exception
    assert len(app.session_state['results'])==1
    assert app.session_state['results'][0][1]['plates']
    assert any(m.label=='Plates' for m in app.metric)
    next(s for s in app.selectbox if s.label=='Active edge method').set_value('Laplacian of Gaussian').run()
    next(b for b in app.button if b.label=='Analyze images').click().run()
    assert not app.exception
    assert app.session_state['results'][0][1]['settings']['edge']=='Laplacian of Gaussian'
    next(b for b in app.button if b.label=='Reset workspace').click().run()
    assert not app.exception
    assert not app.session_state['results'] if 'results' in app.session_state else True
