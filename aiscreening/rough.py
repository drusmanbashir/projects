# %%
from label_analysis.utils.utils.helpers import crop_center, get_labels
import ipdb
from label_analysis.markups import MarkupFromLabelmap
from registration.groupreg import (
    apply_tfm_file,
    compound_to_np,
    create_vector,
    store_compound_img,
)

from utilz.stringz import info_from_filename

tr = ipdb.set_trace

import pandas as pd
import ast
from det3d.monai.data.dataset import Dataset
from utilz.helpers import pbar
from det3d.monai.transforms.compose import Compose
from det3d.det3d.monai.transforms.croppad.dictionary import CropForegroundd, ResizeWithPadOrCropd
from det3d.det3d.monai.transforms.intensity.dictionary import MaskIntensityd
from det3d.det3d.monai.transforms.io.dictionary import SaveImaged
from det3d.det3d.monai.transforms.spatial.dictionary import Spacingd
from det3d.det3d.monai.transforms.utility.dictionary import (
    EnsureChannelFirstd,
    SqueezeDimd,
    Transposed,
)
import numpy as np
from pathlib import Path
import SimpleITK as sitk
from torch.utils.data import DataLoader

from fran.transforms.imageio import LoadSITKd
from utilz.fileio import load_json, maybe_makedirs, save_json
from utilz.helpers import find_matching_fn
from utilz.imageviewers import view_sitk, ImageMaskViewer
import ast
from functools import reduce
import sys
import shutil
from label_analysis.geometry import LabelMapGeometry
from label_analysis.merge import MergeLabelMaps

from label_analysis.overlap_old import BatchScorer, ScorerAdvanced, ScorerFiles
from label_analysis.remap import RemapFromMarkup


if __name__ == "__main__":
    fldr = Path("/s/datasets_bkp/litqsmall/lms")
    dfs =[]
    fns = list(fldr.glob("*"))
    for fn in pbar(fns):
        cid = info_from_filename(fn.name, True)['case_id']
        lm = sitk.ReadImage(str(fn))
        L = LabelMapGeometry(lm,ignore_labels =[1])
        df = L.nbrhoods
        df = df.assign(case_id = cid)
        dfs.append(df)
# %%
df = pd.concat(dfs)
df.to_csv("litqsmall_lms.csv",index=False)
