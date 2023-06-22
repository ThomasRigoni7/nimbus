# run comparisons
python src/compare_segmentations.py predictions/fpn_5/32TNS data/labels/cloudless_exolabs_water/32TNS/10m.npz -out_dir prediction_differences/32TNS/
python src/compare_segmentations.py predictions/fpn_5/13TDE data/labels/cloudless_exolabs_water/13TDE/10m.npz -out_dir prediction_differences/13TDE/
python src/compare_segmentations.py predictions/fpn_5/07VEH data/labels/cloudless_exolabs_water/07VEH/10m.npz -out_dir prediction_differences/07VEH/
python src/compare_segmentations.py predictions/fpn_5/32VMP data/labels/cloudless_exolabs_water/32VMP/10m.npz -out_dir prediction_differences/32VMP/