
# %%
import pandas as pd
import collections
from fran.utils.helpers import *
from fran.utils.fileio import *


main_folder = Path('/home/ub/bosniak2')
cases_fldr =main_folder/('cases')

def remove_unwanted_items_from_list(listi,excluded):
    listi2=[]
    for im in listi:
        if not any(e in str(im) for e in excluded):
            listi2.append(im)
    return listi2



# %%
cases = list(cases_fldr.glob("*"))
masks = list(cases_fldr.rglob("*label*"))
excluded = ['ctbl']
masks= remove_unwanted_items_from_list(masks,excluded)
masks.sort()
# %%
# %% [markdown]
## Catching duplicates
# %%

cases  = [im.parent for im in masks]
S.LP.nbrhoods([ca for ca, count in collections.Counter(cases).items() if count >1])
# %%
cases_done = [aa.parent.parent for aa in masks]
print("---Cases completed ---{}".format(len(cases_done)))
S.LP.nbrhoods(cases_done)

remaining = set(cases).difference(set(cases_done))

print("\n---Cases remaining --- {}".format(len(remaining)))
S.LP.nbrhoods(remaining)
len(remaining)
# %%
S.LP.nbrhoods(masks)
# %%
# %% [markdown]
## Image files
# %%
excluded = ['seg','Segment','pre']
imgs = list(cases_fldr.rglob("*.nrrd"))
imgs = remove_unwanted_items_from_list(imgs,excluded)
len(imgs)
# %% [markdown]
## Catching duplicates
# %%

cases  = [im.parent for im in imgs]
S.LP.nbrhoods([ca for ca, count in collections.Counter(cases).items() if count >1])
# %%
# %% [markdown]
## Going through each case folder and moving img and mask (renamed by case) to a new folder
# %%
# %%
if __name__ == "__main__"
# %% [markdown]
## Storing names
# %%
# %%
if __name__ == "__main__":
    fldr = Path("/home/ub/bosniak/imagesTr")
    files = list(fldr.glob("*nrrd"))
    n = 0
    def tmp (fil,overwrite=False):
        nm = fil.name
        outnm =nm.split(".")[0]+"_0000.nii.gz"
        pr =Path("/home/ub/bosniak/nifti") 
        outname = pr/(outnm)
        if not outname.exists() or overwrite==True:
            print(outname)
            im = sitk.ReadImage(str(fil))
            sitk.WriteImage(im,str(outname))
        else: print("exists")

    
    from fran.utils.helpers import multiprocess_multiarg
    args = [[f] for f in files]
    out =    multiprocess_multiarg(tmp,args,debug=False)


# %%

# %%

    fldr = Path("/home/ub/bosniak/nifti_nnunet_output")
    files = list(fldr.glob("*npy"))
    args = [[f,True] for f in files]
    out =    multiprocess_multiarg(np_to_ni,args,debug=False)
# %%
    n=0 
    fldr_im = Path("/home/ub/bosniak/imagesTr")
    mask_fn  =files[n]
    mask_np = np.load(mask_fn)
    im_fn = fldr_im/Path(mask_fn.name.split(".")[0]+".nrrd")
    im_np = sitk.ReadImage(str(im_fn))
    im_np = sitk.GetArrayFromImage(im_np)
# %%
    mask_sc= np.argmax(mask_np,axis=0)
    mask= mask_sc.transpose(1,2,0)
# %%
    ImageMaskViewer([im_np,mask])
# %%
# %% [markdown]
## Resize image files to match nnunet mask sizes
# %%

# %%
    import torch.nn.functional as F
    def nnUNet_mask_to_pairs(mask_fn,overwrite=False):
        idi = mask_fn.name.split(".")[0]
        out_mask_fn = mask_fn.parent/(idi+"_nokid_mask.nii.gz")
        out_img_fn= mask_fn.parent/(idi+"_nokid_img.nii.gz")
        if not any([out_mask_fn.exists(),out_img_fn.exists()]) or overwrite==True:
            mask = np.load(mask_fn)
            img = mask[0]
            mask_final = mask[1:].astype(np.uint8)
            mask_final = np.argmax(mask_final,0)
            mask_final = mask_final.astype(np.uint8)

            img = img.transpose(1,2,0)
            mask_final = mask_final.transpose(1,2,0)

            img_sitk = sitk.GetImageFromArray(img)
            mask_sitk = sitk.GetImageFromArray(mask_final)
            mask_sitk.SetOrigin(img_sitk.GetOrigin())
            sitk.WriteImage(mask_sitk,str(out_mask_fn))
            sitk.WriteImage(img_sitk,str(out_img_fn))
        else: print("Files exists. Skipping")
# %%

    fldr = Path("/home/ub/bosniak/nifti_nnunet_output")
    files = list(fldr.glob("*npy"))
    args = [[f,True] for f in files]
    multiprocess_multiarg(nnUNet_mask_to_pairs,args)
#
