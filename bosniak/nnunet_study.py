# %%
from utilz.helpers import *
from utilz.fileio import *


# %%
if __name__ == "__main__":
    fldr ="/s/nnUNet_results/trained_models/3d_cascade_fullres/Task135_KiTS2021/nnUNetTrainerV2CascadeFullRes__nnUNetPlansv2.1"

    fl = Path(fldr)/"plans.pkl"

    d = load_dict(fl)
# %%
    d.keys()
    hires = Path("/home/ub/datasets/preprocessed/nnunet/Task500_kits21/nnUNetData_plans_v2.1_stage1/")
    files = list(hires.glob("*np*"))
    outfldr = Path("/home/ub/datasets/preprocessed/nnunet/Task500_kits21/nnUNetData_plans_v2.1_stage1_nifti/") 
# %%
    maybe_makedirs(
    outfldr
    )

# %%


    args = [[f,  outfldr/(f.name.split(".")[0]+".nii.gz") ,True] for f in files]
    out =    multiprocess_multiarg(np_to_ni,args,debug=False)
# %%
pp(print(globals().keys()))
