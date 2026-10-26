import csv
import io
import json
import re
import time
from pathlib import Path
from tempfile import TemporaryDirectory

from playwright.sync_api import sync_playwright

ROOT = Path('artifacts/step18')
ROOT.mkdir(parents=True,exist_ok=True)
checks=[]


def passed(name):
    checks.append(name)
    print('PASS',name,flush=True)


def wait_plot(page, predicate, timeout=180):
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        plot=page.locator('.js-plotly-plot:visible').first
        if plot.count() and plot.evaluate(predicate):
            return plot
        page.wait_for_timeout(250)
    raise AssertionError('Plot did not reach expected state; active tabs: '+str(page.locator('[role=tab][aria-selected=true]').all_text_contents()))


def choose_search(page,index,query,target):
    page.locator('.atlas-controls button:visible').nth(index).click()
    page.get_by_placeholder('Search...',exact=True).filter(visible=True).fill(query)
    page.get_by_role('option').filter(has_text=target).first.click()


def download(page,label,directory):
    with page.expect_download(timeout=30000) as event:
        page.get_by_text(label,exact=True).filter(visible=True).click()
    result=event.value
    path=Path(directory)/result.suggested_filename
    result.save_as(path)
    assert path.stat().st_size>0
    return path


with sync_playwright() as playwright,TemporaryDirectory() as directory:
    browser=playwright.chromium.launch(executable_path='/home/ronaldo/.cache/ms-playwright/chromium-1243/chrome-linux64/chrome',args=['--no-sandbox'])
    page=browser.new_page(viewport={'width':1440,'height':1050},device_scale_factor=1)
    errors=[]
    page.on('pageerror',lambda error:errors.append(str(error)))
    page.goto('http://127.0.0.1:2724',wait_until='domcontentloaded')
    page.get_by_role('tab',name='Discoveries',exact=True).wait_for(timeout=90000)
    page.locator('tbody tr').first.wait_for(timeout=90000)
    assert page.get_by_role('tab').all_text_contents()==['Discoveries','Drug Explorer','Cell Explorer','Validation']
    passed('Four public tabs load')
    shortlist=download(page,'Download shortlist',directory)
    assert len(list(csv.DictReader(shortlist.open())))==100
    full=download(page,'Download full catalog',directory)
    assert len(list(csv.DictReader(full.open())))==51763
    passed('Shortlist and full catalog downloads')
    page.locator('tbody tr').first.get_by_role('checkbox').click()
    page.get_by_role('tab',name='Drug Explorer',exact=True).click()
    wait_plot(page,"e=>e.layout.title.text.includes('XL888')")
    passed('Candidate selection updates compound, cell and screen')
    choose_search(page,0,'bortezomib','BRD-K88510285-001-17-8')
    wait_plot(page,"e=>e.layout.title.text.includes('bortezomib')")
    page.locator('select:visible').first.wait_for()
    deadline=time.monotonic()+45
    while page.locator('select:visible').first.input_value()!='MTS010' and time.monotonic()<deadline:
        page.wait_for_timeout(250)
    assert page.locator('select:visible').first.input_value()=='MTS010'
    passed('Compound identifier updates source screen')
    choose_search(page,0,'bortezomib','BRD-K88510285-001-11-1')
    choose_search(page,1,'A375','A375')
    plot=wait_plot(page,"e=>e.layout.title.text.includes('A375')")
    assert page.locator('select:visible').first.input_value()=='HTS002'
    passed('Searchable compound and cell selectors')
    page.get_by_role('button',name='Add TabPFN prediction',exact=True).click()
    plot=wait_plot(page,"e=>e.data.some(t=>t.name.includes('TabPFN'))")
    predicted=plot.evaluate("e=>Array.from(e._fullData.find(t=>t.name.includes('TabPFN')).y)")
    assert len(predicted)==120 and all(isinstance(value,(int,float)) for value in predicted)
    passed('Real configured TabPFN prediction')
    cursor=page.locator('select:visible').last
    selected=cursor.locator('option').first.get_attribute('value')
    cursor.select_option(selected)
    plot=wait_plot(page,"e=>e.data.some(t=>t.name.includes('TabPFN'))&&e.layout.annotations[0].text.includes('0.000')")
    assert plot.evaluate("e=>Array.from(e._fullData.find(t=>t.name.includes('TabPFN')).y)")==predicted
    passed('Cursor updates annotation and preserves prediction')
    observed=download(page,'Download observations',directory)
    rows=list(csv.DictReader(observed.open()))
    assert {row['broad_id'] for row in rows}=={'BRD-K88510285-001-11-1'}
    assert {row['screen_id'] for row in rows}=={'HTS002'}
    assert len({row['column_name'] for row in rows})==len(rows)
    download(page,'Download interactive figure',directory)
    plot.hover()
    with page.expect_download(timeout=30000) as event:
        page.locator('[data-title*="Download plot"]').filter(visible=True).first.click()
    image=event.value
    image.save_as(Path(directory)/image.suggested_filename)
    assert '<svg' in (Path(directory)/image.suggested_filename).read_text()[:1000]
    passed('Current observations, HTML and SVG figure downloads')
    plot.hover()
    page.locator('[data-title="Zoom in"]').filter(visible=True).first.click()
    page.locator('[data-title="Reset axes"]').filter(visible=True).first.click()
    page.get_by_role('button',name='Fit interpretation and uncertainty').click()
    page.get_by_text('Confidence intervals are unavailable',exact=False).filter(visible=True).wait_for()
    passed('Plot zoom/reset and uncertainty disclosure')
    page.get_by_role('tab',name='Cell Explorer',exact=True).click()
    page.locator('select:visible').nth(0).select_option(label='nocodazole')
    page.get_by_role('heading',name='nocodazole at',exact=False).wait_for(timeout=60000)
    dose=page.locator('select:visible').nth(1)
    previous_dose=dose.input_value()
    selected_dose=dose.locator('option').first.get_attribute('value')
    previous_fields=page.locator('select:visible').nth(2).locator('option').evaluate_all('(options)=>options.map(o=>o.value)')
    dose.select_option(selected_dose)
    page.get_by_role('heading',name=f'nocodazole at {float(selected_dose):g} μM',exact=True).wait_for(timeout=60000)
    field=page.locator('select:visible').nth(2)
    deadline=time.monotonic()+30
    while previous_dose!=selected_dose and field.locator('option').evaluate_all('(options)=>options.map(o=>o.value)')==previous_fields and time.monotonic()<deadline:
        page.wait_for_timeout(100)
    option=field.locator('option').nth(1)
    field_label=option.inner_text()
    field.select_option(option.get_attribute('value'))
    well,number=re.search(r'Well (\w+) · field (\d+)',field_label).groups()
    page.get_by_text(f'Well {well}, field {number}',exact=True).filter(visible=True).wait_for(timeout=30000)
    page.wait_for_timeout(300)
    assert page.locator('img:visible').count()==2
    assert page.locator('img:visible').evaluate_all('(images)=>images.every(image=>image.naturalWidth>0&&image.alt.length>0)')
    page.get_by_role('heading',name='Embedding map unavailable').wait_for()
    passed('Imaging compound, dose, field, controls and honest map state')
    page.get_by_role('button',name='Channels and map interpretation').click()
    passed('Microscopy interpretation disclosure')
    page.get_by_role('tab',name='Drug Explorer',exact=True).click()
    wait_plot(page,"e=>e.layout.title.text.includes('A375')&&e.data.some(t=>t.name.includes('TabPFN'))")
    passed('Imaging selection does not alter PRISM condition')
    page.get_by_role('tab',name='Validation',exact=True).click()
    wait_plot(page,'e=>e.data[0].type==="bar"')
    page.get_by_role('button',name='Evaluate this condition',exact=True).click()
    page.get_by_text('Current condition',exact=True).filter(visible=True).wait_for(timeout=180000)
    plot=wait_plot(page,'e=>e.data[0].type==="bar"')
    evaluated=download(page,'Download predictions and failures',directory)
    evaluation_rows=list(csv.DictReader(evaluated.open()))
    assert {row['model'] for row in evaluation_rows}=={'Hill LL4','Mean baseline','Segmented','TabPFN'}
    assert all(row['curve_id'].startswith('BRD-K88510285-001-11-1/') for row in evaluation_rows)
    passed('Live concentration benchmark and matching export')
    page.get_by_role('button',name='Model coverage and error',exact=True).click()
    page.get_by_role('button',name='Evaluation scope',exact=True).click()
    page.get_by_role('button',name='Methods and citations',exact=True).click()
    method=download(page,'Download methods and citations',directory)
    assert method.read_bytes()==Path('METHODS.md').read_bytes()
    passed('Coverage, methods, scope and citation download')
    for width,height,device in [(1440,1050,'desktop'),(390,844,'mobile')]:
        page.set_viewport_size({'width':width,'height':height})
        for tab in ['Discoveries','Drug Explorer','Cell Explorer','Validation']:
            page.get_by_role('tab',name=tab,exact=True).click()
            if tab in ['Discoveries','Drug Explorer','Validation']:
                page.locator('.js-plotly-plot:visible').first.wait_for(timeout=30000)
            if tab=='Drug Explorer':
                assert page.locator('.js-plotly-plot:visible').first.evaluate("e=>e._fullLayout.xaxis.type==='log'&&e._fullLayout.xaxis.title.text==='Concentration (μM)'&&e._fullLayout.yaxis.title.text==='Relative viability'")
            page.wait_for_timeout(300)
            page.get_by_role('heading',name='Phenotypic Drug-Response Atlas',exact=True).scroll_into_view_if_needed()
            page.evaluate("document.querySelectorAll('*').forEach(e=>{if(e.scrollTop)e.scrollTop=0})")
            assert not page.evaluate('document.documentElement.scrollWidth>innerWidth')
            page.screenshot(path=str(ROOT/(tab.lower().replace(' ','-')+'-'+device+'.png')),full_page=True)
            if tab in ['Drug Explorer','Validation']:
                page.locator('.js-plotly-plot:visible').first.screenshot(path=str(ROOT/(tab.lower().replace(' ','-')+'-figure-'+device+'.png')))
            if tab=='Discoveries':
                page.locator('tbody tr').first.scroll_into_view_if_needed()
                page.screenshot(path=str(ROOT/('ranked-candidates-'+device+'.png')),full_page=True)
        passed(device+' navigation, rendering and no horizontal overflow')
    assert not errors,errors
    passed('No browser JavaScript errors')
    (ROOT/'browser-results.json').write_text(json.dumps({'status':'passed','checks':checks,'javascript_errors':errors,'viewports':[[1440,1050],[390,844]]},indent=2)+'\n')
    browser.close()
