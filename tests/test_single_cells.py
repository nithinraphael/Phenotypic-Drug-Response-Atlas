import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
import polars as pl
import tifffile

from analysis import provenance
from public_ui import selected_image_field
from single_cells import crop_cell,describe_cells,load_atlas,population_summary,read_crop
from single_cell_ui import cell_figure,crop_tiff


def image_fixture():
    channels=np.arange(3*20*20,dtype=np.uint16).reshape(3,20,20)
    cells=np.zeros((20,20),dtype=np.int32)
    cells[4:10,4:10]=1;cells[0:4,12:18]=2;cells[12:18,4:10]=3
    nuclei=np.zeros_like(cells)
    nuclei[5:8,5:8]=1;nuclei[1:3,13:15]=2
    row=dict(image_field_id='synthetic/field',Image_Metadata_Compound='synthetic test compound',Image_Metadata_Concentration=1.,
             Image_Metadata_Plate_DAPI='synthetic',Image_Metadata_Well_DAPI='A01',Replicate=1)
    return channels,cells,nuclei,row


class SingleCellTests(unittest.TestCase):
    def test_identity_coordinates_and_exclusions(self):
        image,mask,nuclei,row=image_fixture()
        cells=describe_cells(mask,nuclei,image,row).ok
        self.assertEqual(cells['cell_id'].to_list(),['synthetic/field::cell=1','synthetic/field::cell=2','synthetic/field::cell=3'])
        first=cells.row(0,named=True)
        self.assertEqual((first['x'],first['y'],first['area_px']),(6.5,6.5,36))
        self.assertEqual(first['nuclear_count'],1)
        self.assertEqual(cells['eligible'].to_list(),[True,False,False])
        self.assertIn('boundary',cells['exclusion_reason'][1])
        self.assertIn('nucleus',cells['exclusion_reason'][2])

    def test_minor_nuclear_overlap_does_not_create_another_nucleus(self):
        image,mask,nuclei,row=image_fixture()
        nuclei[6:14,9:14]=3
        cells=describe_cells(mask,nuclei,image,row).ok
        self.assertEqual(cells['nuclear_count'][0],1)

    def test_empty_invalid_and_noninteger_masks_fail_honestly(self):
        image,mask,nuclei,row=image_fixture()
        for bad in (np.zeros_like(mask),mask.astype(float),-mask,mask[:10]):
            self.assertTrue(describe_cells(bad,nuclei,image,row).is_error())

    def test_raw_crops_preserve_all_channels_and_spatial_bounds(self):
        image,mask,nuclei,row=image_fixture()
        first=describe_cells(mask,nuclei,image,row).ok.row(0,named=True)
        crop,cell_mask,bounds=crop_cell(image,mask,first,padding=2)
        self.assertEqual(bounds,(2,2,12,12))
        np.testing.assert_array_equal(crop,image[:,2:12,2:12])
        self.assertEqual(int(cell_mask.sum()),36)
        self.assertEqual(crop.dtype,np.uint16)

    def test_population_profiles_weight_fields_and_wells_not_cell_counts(self):
        cells=pl.DataFrame(dict(compound=['synthetic']*5,dose_um=[1.,1.,1.,1.,2.],plate_id=['P']*5,
            well_id=['A']*4+['B'],replicate=[1]*5,image_field_id=['F1','F2','F2','F2','F3'],
            area_px=[10,20,30,40,50],nuclear_fraction=[.2]*5))
        vectors=np.array([[1.,0.],[0.,1.],[0.,1.],[0.,1.],[1.,0.]])
        wells,changes=population_summary(cells,vectors).ok
        self.assertEqual(wells['cells'].to_list(),[4,1])
        self.assertEqual(wells['fields'].to_list(),[2,1])
        self.assertEqual(changes['lower_wells'][0],1)
        self.assertAlmostEqual(changes['cosine_change'][0],1-1/np.sqrt(2))
        self.assertFalse(any('p_value' in c or 'confidence' in c for c in changes.columns))
        self.assertTrue(population_summary(cells,vectors[:2]).is_error())

    def test_raw_crop_export_contains_original_uint16_channels(self):
        channels,mask,nuclei,row=image_fixture()
        cell=describe_cells(mask,nuclei,channels,row).ok.row(0,named=True)
        crop,selected,bounds=crop_cell(channels,mask,cell)
        with tempfile.TemporaryDirectory() as directory:
            archive=Path(directory)/'crops.zip';buffer=io.BytesIO()
            np.savez_compressed(buffer,channels=crop,mask=selected)
            with zipfile.ZipFile(archive,'w') as z:z.writestr('1.npz',buffer.getvalue())
            cell['crop_archive']=str(archive)
            np.testing.assert_array_equal(read_crop(cell)[0],crop)
            exported=crop_tiff(cell)
            np.testing.assert_array_equal(tifffile.imread(io.BytesIO(exported)),crop)

    def test_shuffled_embedding_identifiers_join_to_correct_cells(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            raw=root/'input';raw.write_text('synthetic test fixture')
            pl.DataFrame(dict(cell_id=['A','B'])).write_parquet(root/'cells.parquet')
            for file in ('fields','wells','changes'):pl.DataFrame(dict(test=['synthetic'])).write_parquet(root/f'{file}.parquet')
            np.savez(root/'embeddings.npz',cell_id=np.array(['B','A']),embeddings=np.ones((2,384),dtype=np.float32),coordinates=np.array([[20.,21.],[10.,11.]],dtype=np.float32))
            outputs={p.name:provenance.sha256(p) for p in root.iterdir() if p.name!='input'}
            provenance.write_json(root/'manifest.json',provenance.manifest('Synthetic cache test',[raw],{},outputs=outputs))
            result=load_atlas(root)
            self.assertTrue(result.is_ok())
            self.assertEqual(result.ok['cells']['umap_1'].to_list(),[10.,20.])
            (root/'embeddings.npz').write_bytes(b'changed')
            self.assertTrue(load_atlas(root).is_error())

    def test_cell_plot_click_uses_stable_ids_without_temporal_lines(self):
        cells=pl.DataFrame(dict(cell_id=['A','B'],compound=['synthetic','vehicle'],dose_um=[1.,0.],well_id=['A','B'],mask_label=[1,2],
            area_px=[100,200],nuclear_fraction=[.2,.3],umap_1=[1.,2.],umap_2=[3.,4.]))
        for color in ('Compound','Dose','Cell area','Nuclear fraction'):
            figure=cell_figure(cells,'synthetic',1.,'A',color)
            self.assertTrue(all(trace.mode=='markers' for trace in figure.data))
            self.assertEqual(selected_image_field([dict(curveNumber=0,pointIndex=0)],figure),figure.data[0].customdata[0][0])
            self.assertEqual(figure.layout.height,600)

    def test_missing_atlas_does_not_invent_cells(self):
        with tempfile.TemporaryDirectory() as directory:self.assertTrue(load_atlas(directory).is_error())
