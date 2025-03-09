# %%
import ast
import shutil
import time
from pathlib import Path
from label_analysis.helpers import to_binary
from xnat.object_oriented import *
from dicom_utils.capestart_related import collate_nii_foldertree

from dicom_utils.helpers import dcm_segmentation
from label_analysis.utils import fix_slicer_labelmap, get_metadata, thicken_nii
from xnat.object_oriented import *
from utilz.fileio import maybe_makedirs
from utilz.helpers import find_matching_fn


if __name__ == "__main__":

    parent = Path("/s/xnat_shadow/tcianode/")
    fldr_imgs = parent/("images")
    fldr_lms = parent/("lms")
    
    fns_lm = list(fldr_lms.glob("*"))
    fns_img = list(fldr_imgs.glob("*"))
# %%
    dicis = []
    for fn_lm in pbar( fns_lm[25:]):
        fn_img = find_matching_fn(fn_lm,fns_img)

        img= sitk.ReadImage(str(fn_img))
        sz_img = img.GetSize()
        lm = sitk.ReadImage(str(fn_lm))
        sz = lm.GetSize()
        sp = lm.GetSpacing()
        labs = get_labels(lm)
        lm_bin =to_binary(lm)
        fil1 = sitk.LabelShapeStatisticsImageFilter()
        fil1.Execute(lm_bin)
        bb = fil1.GetBoundingBox(1)
        dici = {'fn': fn_lm, 'size_lm':sz, 'size_img':sz_img, 'spacing':sp, 'bbox':bb ,'num_lesions': len(labs)}
        dicis.append(dici)

# %%
    df = pd.DataFrame(dicis)
    df.to_csv("lminfo.csv")

# %%
    df2 = pd.read_csv("lminfo.csv")
    df2['shape_out'] = None

    ind = 0
# %%
    for ind in range(len(df2)):
        row = df2.iloc[ind]
        sp = row['spacing']
        bbox =row.bbox
        try:
            bbox = ast.literal_eval(bbox)
            sp = ast.literal_eval(sp)
        except:
            pass
        bbox = bbox[3:]
        sp_target = .8,.8,1.5
        factor = [a/b for a,b in zip(sp,sp_target)]
        shape_out = [int(a*b) for a,b in zip(sp_target,bbox)]
        df2.at[ind,'shape_out'] = shape_out
# %%
    df2.to_csv("lminfo.csv")


