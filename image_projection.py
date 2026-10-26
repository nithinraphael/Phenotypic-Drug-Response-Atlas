import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from expression import Error, Ok

from analysis import embeddings, microscopy, provenance


def load_projection(metadata, root='data/bbbc021'):
    config_path = Path(root) / 'projection_config.json'
    if not config_path.is_file():
        return Error('No verified projection configured. Cell-DINO weights and channel compatibility are unresolved.')
    config = json.loads(config_path.read_text())
    sidecar = Path(config['cache_path']).with_suffix('.json')
    if not sidecar.is_file():
        return Error('Projection manifest missing')
    manifest = json.loads(sidecar.read_text())
    verified = provenance.verify_manifest(manifest, list(manifest.get('inputs', {})), config['analysis_config'])
    if verified.is_error():
        return verified
    return embeddings.configured_projection(metadata, root)


def generate(config_path, device='cpu', batch_size=2):
    import torch
    import torchvision.transforms as T
    import umap

    config_path = Path(config_path)
    cache = Path('data/bbbc021/dino_baseline_projection.npz')
    if cache.exists() or Path('data/bbbc021/projection_config.json').exists():
        return Error('Projection already exists; preserve it and choose an explicit new run before regenerating')
    settings = json.loads(config_path.read_text())
    for path, expected in settings['verified_files'].items():
        if not Path(path).is_file() or provenance.sha256(path) != expected:
            return Error(f'Verified model file missing or changed: {path}')
    if device == 'cuda' and not torch.cuda.is_available():
        return Error('Requested CUDA device unavailable; use --device cpu explicitly')
    metadata_result = microscopy.load_metadata()
    if metadata_result.is_error():
        return metadata_result
    metadata = metadata_result.ok.sort('image_field_id')
    torch.manual_seed(42)
    torch.set_num_threads(4)
    model = torch.hub.load(settings['source_directory'], 'dino_vits8', source='local', pretrained=False)
    model.load_state_dict(torch.load(settings['checkpoint_path'], map_location='cpu', weights_only=True), strict=True)
    model.eval().to(device)
    preprocessing = dict(channels=['Actin', 'Tubulin', 'DAPI'], contrast_percentiles=[1, 99.5],
                         display_rgb_uint8=True, resize_short_edge=256, center_crop=224,
                         interpolation='bilinear', antialias=True,
                         mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = T.Compose([T.ToTensor(), T.Resize(256, antialias=True), T.CenterCrop(224),
                           T.Normalize(preprocessing['mean'], preprocessing['std'])])
    vectors = []
    rows = metadata.to_dicts()
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            images = [microscopy.load_microscopy_rgb('data/bbbc021/plate.zip', row) for row in rows[start:start+batch_size]]
            errors = [result.error for result in images if result.is_error()]
            if errors:
                return Error('; '.join(errors))
            batch = torch.stack([transform(result.ok) for result in images]).to(device)
            vectors.append(model(batch).cpu().numpy())
            print(f'Embedded {min(start+batch_size, len(rows))}/{len(rows)} fields', flush=True)
    vectors = np.concatenate(vectors)
    if vectors.shape != (metadata.height, 384) or not np.all(np.isfinite(vectors)):
        return Error('Invalid DINO embedding output')
    normalized = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    coordinates = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42, n_jobs=1).fit_transform(normalized)
    np.savez_compressed(cache, image_field_id=np.array(metadata['image_field_id'].to_list()), embeddings=vectors, coordinates=coordinates)
    analysis_config = dict(model_id='DINO ViT-S/8 ImageNet baseline', checkpoint_hash=provenance.sha256(settings['checkpoint_path']),
                           preprocessing_hash=hashlib.sha256(json.dumps(preprocessing, sort_keys=True).encode()).hexdigest(),
                           preprocessing=preprocessing, embedding_dimension=384, umap_version=provenance.versions()['umap-learn'],
                           random_seed=42, n_neighbors=15, min_dist=0.1, metric='cosine', input_normalization='L2',
                           n_fields=metadata.height, device=device, precision='float32', batch_size=batch_size,
                           weights_source=settings['weights_source'], architecture_source=settings['architecture_source'],
                           interpretation='Image-field embeddings of contrast-adjusted fluorescence composites; exploratory transfer from ImageNet, not Cell-DINO or single-cell measurements.')
    inputs = ['data/bbbc021/metadata.csv', 'data/bbbc021/plate.zip', str(cache), str(config_path),
              'image_projection.py', *settings['verified_files']]
    provenance.write_json(cache.with_suffix('.json'), provenance.manifest('DINO image-field projection', inputs, analysis_config))
    provenance.write_json('data/bbbc021/projection_config.json', dict(checkpoint_path=settings['checkpoint_path'], cache_path=str(cache), analysis_config=analysis_config))
    return Ok(dict(fields=metadata.height, embedding_dimension=384, cache=str(cache)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/dino_baseline.json')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--batch-size', type=int, choices=[1, 2], default=2)
    args = parser.parse_args()
    result = generate(args.config, args.device, args.batch_size)
    print(result.ok if result.is_ok() else result.error)
    raise SystemExit(0 if result.is_ok() else 1)
