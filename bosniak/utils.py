# %$
import os,sys,cc3d
sys.path.append('/home/ub/Dropbox/code/')
import SimpleITK as sitk
from global_utils.imageviewers import *
from global_utils.fileio import *
from pathlib import Path


# %%

# %%
# %% [markdown]
## Copying relevant nrrd files into local drive from external HDD
# %%

# %%
from distutils.dir_util import copy_tree
import shutil
fold = Path('/media/ub/UB1/bosniak')
files_list =[]
# accepted_desc_match= ["venous","pv","90sec","json"]
accepted_desc_match= ["UB*label"]
for desc in accepted_desc_match:
    a2 = list(fold.rglob(("*"+desc+"*")))
    files_list+=a2
# %%
len(files_list)
# %%
for a in files_list:
    neo =Path(str(a).replace("/media/ub/UB1", "/home/ub"))
    if neo.exists():
        print("Skipping file {0} exists".format(str(neo)))
    else:
        try:
            os.makedirs(neo.parent)
        except FileExistsError:
            print("Folder exists")
            pass
        print("---------Copying file {0}".format(str(neo)))
        shutil.copy(a,neo)
# %%



