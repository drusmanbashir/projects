# %%
from label_analysis.helpers import crop_center, get_labels
import ipdb
from label_analysis.markups import MarkupFromLabelmap
from registration.groupreg import (
    apply_tfm_file,
    compound_to_np,
    create_vector,
    store_compound_img,
)

from fran.utils.string import info_from_filename

tr = ipdb.set_trace

import pandas as pd
import ast
from monai.data.dataset import Dataset
from fran.utils.helpers import pbar
from monai.transforms.compose import Compose
from monai.transforms.croppad.dictionary import CropForegroundd, ResizeWithPadOrCropd
from monai.transforms.intensity.dictionary import MaskIntensityd
from monai.transforms.io.dictionary import SaveImaged
from monai.transforms.spatial.dictionary import Spacingd
from monai.transforms.utility.dictionary import (
    EnsureChannelFirstd,
    SqueezeDimd,
    Transposed,
)
import numpy as np
from pathlib import Path
import SimpleITK as sitk
from torch.utils.data import DataLoader

from fran.transforms.imageio import LoadSITKd
from fran.utils.fileio import load_json, maybe_makedirs, save_json
from fran.utils.helpers import find_matching_fn
from fran.utils.imageviewers import view_sitk, ImageMaskViewer
import ast
from functools import reduce
import sys
import shutil
from label_analysis.geometry import LabelMapGeometry
from label_analysis.merge import MergeLabelMaps

from label_analysis.overlap import BatchScorer, ScorerAdvanced, ScorerFiles
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
