import csv
import hashlib
import io
import json
import os
import re
import time
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import tifffile
from playwright.sync_api import sync_playwright

ROOT=Path(os.environ.get('ATLAS_ARTIFACTS','artifacts/step19/desktop'))
ROOT.mkdir(parents=True,exist_ok=True)
CHECKS=[]


def passed(name):
    CHECKS.append(name)
    print('PASS',name,flush=True)


def wait_plot(page,predicate,timeout=120):
    end=time.monotonic()+timeout
    while time.monotonic()<end:
        plot=page.locator('.js-plotly-plot:visible').first
        if plot.count() and plot.evaluate(predicate):return plot
        page.wait_for_timeout(200)
    raise AssertionError('Expected desktop plot state unavailable')


def wait_ready(page):
    page.wait_for_function("()=>Array.from(document.querySelectorAll('.output-area')).every(e=>{while(e){if(Number(getComputedStyle(e).opacity)<.99)return false;e=e.parentElement}return true})",timeout=60000)
    page.wait_for_timeout(300)


def download(page,label,directory):
    with page.expect_download(timeout=30000) as event:
        page.get_by_text(label,exact=True).filter(visible=True).click()
    result=event.value;path=Path(directory)/result.suggested_filename;result.save_as(path)
    assert path.stat().st_size>0
    return path


with sync_playwright() as playwright,TemporaryDirectory() as directory:
    browser=playwright.chromium.launch(executable_path='/home/ronaldo/.cache/ms-playwright/chromium-1243/chrome-linux64/chrome',args=['--no-sandbox'])
    page=browser.new_page(viewport={'width':1440,'height':1050},device_scale_factor=1)
    errors=[];page.on('pageerror',lambda error:errors.append(str(error)))
    page.goto(os.environ.get('ATLAS_URL','http://127.0.0.1:2734'),wait_until='domcontentloaded')
    page.get_by_role('tab',name='Drug Explorer',exact=True).wait_for(timeout=120000)
    page.get_by_role('tab',name='Drug Explorer',exact=True).click()
    plot=wait_plot(page,"e=>e.data.some(t=>t.name==='Measured viability')")
    prism_title=plot.evaluate('e=>e.layout.title.text')
    page.get_by_role('tab',name='Cell Explorer',exact=True).click()
    page.get_by_role('heading',name=re.compile(r'^Individual-cell atlas')).wait_for(timeout=120000)
    plot=wait_plot(page,"e=>e.layout.title.text.includes('Cell crops')")
    assert page.get_by_role('tab').all_text_contents()==['Discoveries','Drug Explorer','Cell Explorer','Validation']
    passed('Four tabs and retained PRISM view')
    table=download(page,'Download cell atlas table',directory)
    records=list(csv.DictReader(table.open()));by_id={row['cell_id']:row for row in records}
    assert len(by_id)==len(records)>0
    page.get_by_role('button',name='Cell analysis methods and exclusions',exact=True).click()
    manifest=json.loads(download(page,'Download cell-analysis provenance',directory).read_text())
    assert len(records)==manifest['cells']
    assert all(row['eligible']=='true' and row['border']=='false' and int(row['nuclear_count'])>=1 for row in records)
    assert all(float(row['area_px'])>=manifest['config']['core']['settings']['minimum_area_px'] for row in records)
    (ROOT/'cell-atlas-export.csv').write_bytes(table.read_bytes())
    (ROOT/'cell-atlas-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    passed('Actual cell counts, unique IDs, QC exclusions and provenance export')
    plot=wait_plot(page,"e=>e.layout.title.text.includes('Cell crops')")
    plot.scroll_into_view_if_needed()
    point=plot.evaluate("""e=>{const points=[];for(const t of e._fullData)if(t.customdata?.[0]?.length>1)for(let i=0;i<t.x.length;i++)points.push({id:t.customdata[i][0],compound:t.customdata[i][1],x:e._fullLayout.xaxis.l2p(t.x[i])+e._fullLayout.xaxis._offset,y:e._fullLayout.yaxis.l2p(t.y[i])+e._fullLayout.yaxis._offset});let chosen=points[0],best=-1;for(const p of points.filter(p=>p.compound==='cytochalasin B').slice(0,120)){let distance=Infinity;for(const q of points)if(q.id!==p.id)distance=Math.min(distance,(p.x-q.x)**2+(p.y-q.y)**2);if(distance>best){best=distance;chosen=p}}return chosen}""")
    box=plot.bounding_box();page.mouse.click(box['x']+point['x'],box['y']+point['y'])
    row=by_id[point['id']]
    cache_paths=[Path(row['mask_file']),Path(row['crop_archive']),Path(row['mask_file']).parent/'embeddings.npz']
    cache_hashes={str(path):hashlib.file_digest(path.open('rb'),'sha256').hexdigest() for path in cache_paths}
    heading=f"Cell {row['mask_label']} · well {row['well_id']} · {row['compound']}"
    page.get_by_role('heading',name=heading,exact=True).wait_for(timeout=60000)
    plot=wait_plot(page,"e=>e.data.some(t=>t.name==='Selected cell')")
    assert plot.evaluate("e=>e._fullData.find(t=>t.name==='Selected cell').customdata[0][0]")==row['cell_id']
    assert page.locator('select:visible').nth(0).input_value()==row['compound']
    assert float(page.locator('select:visible').nth(1).input_value())==float(row['dose_um'])
    page.get_by_role('heading',name=heading,exact=True).evaluate("e=>e.scrollIntoView({block:'start'})")
    page.wait_for_timeout(300)
    assert page.locator('img:visible').count()==2
    assert page.locator('img:visible').evaluate_all('(images)=>images.every(image=>image.naturalWidth>0&&image.alt.length>0)')
    wait_ready(page)
    page.screenshot(path=str(ROOT/'cell-viewer-desktop.png'))
    passed('Native UMAP mouse click synchronizes compound, dose, field and highlighted cell')
    raw=download(page,'Download raw cell channels',directory)
    exported=tifffile.imread(raw)
    metadata=list(csv.DictReader(Path('data/bbbc021/metadata.csv').open()))
    field=next(item for item in metadata if item['Image_Metadata_Plate_DAPI']+'/'+item['Image_FileName_DAPI']==row['image_field_id'])
    padding=manifest['config']['core']['settings']['crop_padding_px']
    y0,x0=max(0,int(row['min_y'])-padding),max(0,int(row['min_x'])-padding)
    y1,x1=min(1024,int(row['max_y'])+padding),min(1280,int(row['max_x'])+padding)
    with zipfile.ZipFile('data/bbbc021/plate.zip') as archive:
        expected=np.stack([tifffile.imread(archive.open(field['Image_Metadata_Plate_DAPI']+'/'+field['Image_FileName_'+channel]))[y0:y1,x0:x1] for channel in ['DAPI','Tubulin','Actin']])
    np.testing.assert_array_equal(exported,expected)
    assert exported.dtype==np.uint16
    with np.load(row['mask_file'],allow_pickle=False) as masks:yy,xx=np.nonzero(masks['labels'][1]==int(row['mask_label']))
    assert len(xx)==int(row['area_px'])
    np.testing.assert_allclose([float(row['x']),float(row['y'])],[xx.mean(),yy.mean()])
    passed('Downloaded crop preserves original uint16 channels; mask area and coordinates reconcile')
    candidates=[item for item in records if item['image_field_id']==row['image_field_id'] and item['cell_id']!=row['cell_id']]
    target=candidates[0]
    page.locator('.atlas-controls button:visible').first.click()
    page.get_by_placeholder('Search...',exact=True).filter(visible=True).fill(f"Cell {target['mask_label']} ·")
    page.get_by_role('option').filter(has_text=f"Cell {target['mask_label']} ·").first.click()
    page.get_by_role('heading',name=f"Cell {target['mask_label']} · well {target['well_id']} · {target['compound']}",exact=True).wait_for(timeout=60000)
    plot=wait_plot(page,"e=>e._fullData.find(t=>t.name==='Selected cell')?.customdata[0][0]==="+json.dumps(target['cell_id']))
    passed('Searchable individual-cell selector updates overlay and UMAP marker')
    for color in ['Dose','Cell area','Nuclear fraction','Compound']:
        page.locator('select:visible').last.select_option(label=color)
        _title={'Dose':'μM','Cell area':'px²','Nuclear fraction':'Fraction'}.get(color)
        predicate="e=>e.data.filter(t=>t.marker?.showscale).length===0&&e.data.some(t=>t.name==='DMSO')" if color=='Compound' else "e=>e.data.some(t=>t.marker?.showscale&&t.marker.colorbar?.title?.text==="+json.dumps(_title)+")"
        plot=wait_plot(page,predicate)
        assert plot.evaluate('e=>e.getBoundingClientRect().height')==600
    wait_ready(page)
    plot.scroll_into_view_if_needed();plot.screenshot(path=str(ROOT/'single-cell-umap-desktop.png'))
    passed('All four cell-color controls and desktop map geometry')
    page.get_by_role('heading',name='Dose-dependent morphology',exact=True).evaluate("e=>e.scrollIntoView({block:'start'})")
    wait_ready(page)
    page.screenshot(path=str(ROOT/'dose-morphology-desktop.png'))
    wells=list(csv.DictReader(download(page,'Download well summaries',directory).open()))
    assert all(item['compound']==target['compound'] for item in wells)
    assert len({(item['plate_id'],item['well_id']) for item in wells})==len(wells)
    page.get_by_role('button',name='Sampled wells and cell counts',exact=True).click()
    passed('Dose distributions, descriptive profile differences and well-level export')
    dose=page.locator('select:visible').nth(1)
    next_dose=next(option.get_attribute('value') for option in dose.locator('option').all() if option.get_attribute('value')!=dose.input_value())
    dose.select_option(next_dose)
    deadline=time.monotonic()+45
    while time.monotonic()<deadline:
        plot=wait_plot(page,"e=>e.layout.title.text.includes('Cell crops')")
        actual=plot.evaluate("e=>e._fullData.find(t=>t.name==='Selected condition').x.length")
        expected_count=sum(item['compound']==target['compound'] and float(item['dose_um'])==float(next_dose) for item in records)
        if actual==expected_count:break
        page.wait_for_timeout(200)
    assert actual==expected_count
    passed('Dose selector updates the matching cell population')
    field_select=page.locator('select:visible').nth(2)
    field_select.select_option(field_select.locator('option').nth(1).get_attribute('value'))
    page.get_by_role('heading',name='No analyzed cell in this field',exact=True).wait_for(timeout=60000)
    assert not page.get_by_text('Download raw cell channels',exact=True).filter(visible=True).count()
    page.get_by_role('button',name='Image-field analysis',exact=True).click()
    page.get_by_role('heading',name='Image-field morphology map',exact=True).wait_for(timeout=30000)
    assert page.locator('img:visible').count()==2
    passed('Unanalyzed field clears stale cell output; original microscopy and image-level map remain available')
    page.get_by_role('button',name='Image-field analysis',exact=True).click()
    field_select.select_option(field_select.locator('option').first.get_attribute('value'))
    page.get_by_role('tab',name='Validation',exact=True).click()
    page.get_by_role('heading',name='Cross-dataset comparison unavailable',exact=True).wait_for(timeout=60000)
    page.get_by_role('tab',name='Discoveries',exact=True).click()
    page.get_by_role('tab',name='Cell Explorer',exact=True).click()
    wait_plot(page,"e=>e.layout.title.text.includes('Cell crops')")
    page.get_by_role('tab',name='Drug Explorer',exact=True).click()
    plot=wait_plot(page,"e=>e.data.some(t=>t.name==='Measured viability')")
    assert plot.evaluate('e=>e.layout.title.text')==prism_title
    passed('Tab persistence and strict PRISM/BBBC021 separation')
    assert cache_hashes=={str(path):hashlib.file_digest(path.open('rb'),'sha256').hexdigest() for path in cache_paths}
    passed('Cell navigation leaves masks, crops and embeddings unchanged')
    page.get_by_role('tab',name='Cell Explorer',exact=True).click()
    wait_plot(page,"e=>e.layout.title.text.includes('Cell crops')")
    page.get_by_role('heading',name='Phenotypic Drug-Response Atlas',exact=True).scroll_into_view_if_needed()
    wait_ready(page)
    page.screenshot(path=str(ROOT/'cell-explorer-desktop.png'))
    assert not page.evaluate('document.documentElement.scrollWidth>innerWidth')
    assert not errors,errors
    passed('Desktop rendering and no browser JavaScript errors')
    (ROOT/'browser-results.json').write_text(json.dumps(dict(status='passed',checks=CHECKS,viewport=[1440,1050],mobile_validation='Not performed; excluded by user',cell_count=len(records),sampled_fields=manifest['fields'],javascript_errors=errors),indent=2)+'\n')
    browser.close()
