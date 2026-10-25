
import json
import subprocess
from pathlib import Path
from urllib.request import urlopen

root = Path(__file__).resolve().parents[1] / "data"

def download(url, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    temp = path.with_name(path.name + ".part")
    subprocess.run(
        ["curl", "-fL", "--retry", "5", "-C", "-", "-o", str(temp), url],
        check=True,
    )
    temp.rename(path)

files = json.load(urlopen("https://api.figshare.com/v2/articles/9393293"))["files"]
prism = {
    "secondary-screen-logfold-change.csv": "responses.csv",
    "secondary-screen-replicate-treatment-info.csv": "treatments.csv",
    "secondary-screen-dose-response-curve-parameters.csv": "curves.csv",
    "secondary-screen-cell-line-info.csv": "cell_lines.csv",
    "secondary-screen-pooling-info.csv": "pooling.csv",
}

for source, target in prism.items():c
    url = next(f["download_url"] for f in files if f["name"] == source)
    download(url, root / "prism" / target)

base = "https://data.broadinstitute.org/bbbc/BBBC021/"
bbbc = {
    "BBBC021_v1_image.csv": "metadata.csv",
    "BBBC021_v1_compound.csv": "compounds.csv",
    "BBBC021_v1_moa.csv": "mechanisms.csv",
    "BBBC021_v1_images_Week1_22123.zip": "plate.zip",
}

for source, target in bbbc.items():
    download(base + source, root / "bbbc021" / target)

print("Downloads complete.")
