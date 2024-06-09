# %%
import ast
import shutil
import time
from pathlib import Path
from label_analysis.helpers import relabel, to_binary, to_cc, to_int
from xnat.object_oriented import *
from dicom_utils.capestart_related import collate_nii_foldertree

from dicom_utils.helpers import dcm_segmentation
from label_analysis.utils import fix_slicer_labelmap, get_metadata, thicken_nii
from xnat.object_oriented import *
from fran.utils.fileio import maybe_makedirs
from fran.utils.helpers import find_matching_fn


if __name__ == "__main__":

# %%
    lm_fn = "/s/fran_storage/predictions/litsmc/LITS-933/crc_CRC125_20170708_ABDOMEN.nii.gz"
    lm1 = sitk.ReadImage(lm_fn)
    sp = lm1.GetSpacing()
    out_spacing = [1,1,1]
    sz= lm1.GetSize()
    factor = [a/b for a,b in zip(sp,out_spacing)]
    outsize = [int(a*b) for a,b in zip(factor,sz)]
# %%
    res = sitk.ResampleImageFilter()
    res.SetInterpolator(sitk.sitkNearestNeighbor)
    res.SetTransform(sitk.Transform())
    res.SetSize(outsize)
    res.SetOutputDirection(lm1.GetDirection())
    res.SetOutputOrigin(lm1.GetOrigin())
    res.SetDefaultPixelValue(lm1.GetPixelIDValue())
    res.SetOutputSpacing(out_spacing)
# %%

    lm2 = res.Execute(lm1)
    get_labels(lm2)
    view_sitk(lm2,lm1,'mm')
# %%


    lm_lesions = relabel(lm2,{1:0})
    lm_lesions = to_binary(lm_lesions)
    lm_lesions = to_cc(lm_lesions)
    get_labels(lm_lesions)

    
# %%
    lm_liver = to_binary(lm2)
    fil= sitk.BinaryErodeImageFilter()
    fil.SetForegroundValue(1)
    fil.SetKernelRadius(10)


# %%
    lm_core = fil.Execute(lm_liver)
    lm_shell = lm_liver-lm_core
    lm_shell = to_int(lm_shell)
    lm_lesions = to_int(lm_lesions)
    all = lm_lesions+lm_shell

    view_sitk(lm2,lm_lesions,'mm')
# %%
    sitk.WriteImage(lm1,"org.nii.gz")
    sitk.WriteImage(lm2,"org_shrunk_10.nii.gz")
    sitk.WriteImage(lm2,"res.nii.gz")
    sitk.WriteImage(lm2,"res_shrunk10.nii.gz")
    sitk.WriteImage(all,"shell_lesions.nii.gz")

# %%
    view_sitk(lm1,lm2,dtypes='mm')
    lm_liver = lm2-lm1


    sub.SetInput1(lm2)
    sub.SetInput2(lm1)
    
# %%
