import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen


PRISM = {
    'secondary-screen-logfold-change.csv': 'responses.csv',
    'secondary-screen-replicate-treatment-info.csv': 'treatments.csv',
    'secondary-screen-dose-response-curve-parameters.csv': 'curves.csv',
    'secondary-screen-cell-line-info.csv': 'cell_lines.csv',
    'secondary-screen-pooling-info.csv': 'pooling.csv',
}
BBBC = {
    'BBBC021_v1_image.csv': 'metadata.csv',
    'BBBC021_v1_compound.csv': 'compounds.csv',
    'BBBC021_v1_moa.csv': 'mechanisms.csv',
    'BBBC021_v1_images_Week1_22123.zip': 'plate.zip',
}


def download(url, path):
    if path.exists():
        return 'existing file; acquisition identity unverified'
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.part')
    subprocess.run(['curl','-fL','--retry','5','-C','-','-o',str(temp),url],check=True)
    temp.rename(path)
    return 'downloaded'


def digest(path, algorithm='sha256'):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,algorithm).hexdigest()


def main():
    parser=argparse.ArgumentParser(description='Explicitly download PRISM and one BBBC021 plate without overwriting existing files')
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]/'data')
    args=parser.parse_args()
    article=json.load(urlopen('https://api.figshare.com/v2/articles/9393293'))
    records=[]
    for source,target in PRISM.items():
        record=next(f for f in article['files'] if f['name']==source)
        path=args.root/'prism'/target
        status=download(record['download_url'],path)
        records.append(dict(dataset='PRISM',source_filename=source,source_url=record['download_url'],
                            article_id=9393293,article_version=article.get('version'),file_id=record['id'],
                            path=str(path),sha256=digest(path),upstream_md5=record.get('computed_md5'),
                            upstream_checksum_matches=digest(path,'md5')==record.get('computed_md5'),status=status))
    for source,target in BBBC.items():
        url='https://data.broadinstitute.org/bbbc/BBBC021/'+source
        path=args.root/'bbbc021'/target
        status=download(url,path)
        records.append(dict(dataset='BBBC021',version=1,source_url=url,path=str(path),sha256=digest(path),status=status))
    output=args.root/('acquisition-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.json')
    output.write_text(json.dumps(dict(checked_at=datetime.now(timezone.utc).isoformat(),files=records),indent=2)+'\n')
    print(output)


if __name__=='__main__':
    main()
