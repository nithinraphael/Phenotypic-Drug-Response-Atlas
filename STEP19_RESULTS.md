# Step 19 results

## Executed analysis

Pretrained inference retained 5,857 cell objects from 60 wells, one deterministically selected field per well (60 of 240 available fields, seven compounds). No model training. PRISM and existing image-field analysis remain separate and unchanged.

InstanSeg's initial cell predictions fragmented the control cytoplasm. Cellpose-SAM 4.0.8 therefore supplied cell masks uniformly across the atlas; InstanSeg model 0.1.1 supplied nuclear predictions. Generic ImageNet DINO ViT-S/8 supplied 384-dimensional crop embeddings. These are not Cell-DINO embeddings.

Total segmented objects: 8,647; retained: 5,857; excluded: 2,790. Edge-touching objects: 2,143; without nuclear support: 1,171. Exclusion reasons overlap. All original masks and excluded records remain cached.

GPU execution: 2859.10 seconds (47.65 minutes); peak process RSS 2,656,059,392 bytes (2.47 GiB); peak allocated GPU memory 2,472,572,416 bytes (2.30 GiB), GTX 1650. Allocation excludes driver and other-process memory. Separate real 256×256 CPU smoke inference produced 20 cell and 23 nuclear objects in 48.80 seconds, peak process RSS 3,306,061,824 bytes.

## Data verification

Manifest SHA256: `e905e3a48971962671b7263b65db8a79eeb09f8de85efc47a104e557febcdc8c`. Full hashes and software versions: `data/bbbc021/single_cells/atlas-v1/manifest.json`.

Verified unique stable cell IDs, image/compound/dose/well metadata, mask areas, raw uint16 channel preservation, 384-dimensional normalized vectors and two-dimensional finite coordinates. Compared 180 cached crops against original source pixels. Recalculated all well summaries and adjacent-dose embedding differences. UMAP coordinates join by cell ID, including shuffled-key tests.

Retained disconnected masks: 258 (4.40%). These predictions remain visible with a warning. This is an integrity count, not an accuracy score.

## Interface

Cell Explorer includes a shared single-cell UMAP, compound/dose/area/nuclear-fraction colors, linked field and crop viewers, searchable cell selection, QC counts, per-dose area distributions, adjacent-dose profile differences, raw TIFF and table exports, and provenance. Existing image-field analysis remains in its disclosure. Public navigation never invokes segmentation or embedding inference.

## Limitations

No matched manual reference masks: IoU/Dice accuracy unavailable. Zenodo archive identifiers could not be mapped traceably to this local plate; no substituted masks. CellProfiler baseline was not executed. Optical pixel calibration is unavailable; areas/coordinates use pixels. Only one field per well was processed. All supplied metadata identify replicate 1; biological independence is unverified. Statistics are descriptive with equal field/well weighting; no cell-level inferential p-values, confidence intervals or confirmed transitions. Generic DINO transfer and UMAP remain exploratory. Excluding boundary/non-nucleated objects can bias populations, particularly damaged cells. No temporal trajectories or viability claims derive from these cells.

Desktop-only verification; mobile implementation and validation excluded as requested. Test results are recorded in TEST_RESULTS.md and artifacts/step19.

## Final verification

83 automated tests passed (0 failed, 0 skipped), 12 desktop browser groups passed, raw-data integrity validation passed, marimo CLI/default-rule checks passed and the complete notebook executed with exit 0. Final screenshots were inspected directly. A separate finish reviewer could not execute because of its usage limit; no independent visual-review approval is claimed.
