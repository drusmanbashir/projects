# %%
import ray
import ast
import shutil
import time
from pathlib import Path
from label_analysis.geometry import LabelMapGeometry
from label_analysis.helpers import empty_img, np_to_native, relabel, to_binary, to_cc, to_int
from numpy import nan_to_num
from xnat.object_oriented import *
from dicom_utils.capestart_related import collate_nii_foldertree

from dicom_utils.helpers import dcm_segmentation
from label_analysis.utils import fix_slicer_labelmap, get_metadata, thicken_nii
from xnat.object_oriented import *
from fran.utils.fileio import maybe_makedirs, np_to_ni
from fran.utils.helpers import chunks, find_matching_fn

class Onions():
    def __init__(self,lm, shell_dia:int =10 ):
        '''
        margin: int millimeters
        '''
        if isinstance(lm,Union[str|Path]):
            lm = sitk.ReadImage(str(lm))
        self.lm = lm
        self.lm_bin = to_binary(lm)
        self.out_spacing = [1,1,1]
        self.shell_dia =shell_dia# 10 pixel deep erosion# aka 10mm of isotropic image
        self.metadata(lm)
        self.create_resamplers()
        self.create_erode_filter()
        self.lm_iso = self.res_iso.Execute(self.lm_bin)
        self.create_onions()
        
    def create_onions(self):
        # i.e., create onions
        self.core = self.eroder.Execute(self.lm_iso)
        self.core = self.res_rev.Execute(self.core)
        self.shell = self.lm_bin- self.core
        self.core = self.res_rev.Execute(self.core)

    def create_erode_filter(self):
        self.eroder= sitk.BinaryErodeImageFilter()
        self.eroder.SetForegroundValue(1)
        self.eroder.SetKernelRadius(self.shell_dia)



    def create_resamplers(self):
        self.resampler_isotropic()
        self.resampler_revert()

    def metadata(self,lm):
        self.spacing= lm.GetSpacing()
        self.size = lm.GetSize()
        self.direction = lm.GetDirection()
        self.origin = lm.GetOrigin()
        self.default_pixel = lm.GetPixelIDValue()
        self.factor = [a/b for a,b in zip(self.spacing,self.out_spacing)]
        self.size_isotropic = [int(a*b) for a,b in zip(self.factor,self.size)]

    def resampler_isotropic(self):
        self.res_iso = sitk.ResampleImageFilter()
        self.res_iso.SetInterpolator(sitk.sitkNearestNeighbor)
        self.res_iso.SetTransform(sitk.Transform())
        self.res_iso.SetSize(self.size_isotropic)
        self.res_iso.SetOutputDirection(self.direction)
        self.res_iso.SetOutputOrigin(self.origin)
        self.res_iso.SetDefaultPixelValue(self.default_pixel)
        self.res_iso.SetOutputSpacing(self.out_spacing)

    def resampler_revert(self):
        self.res_rev = sitk.ResampleImageFilter()
        self.res_rev.SetInterpolator(sitk.sitkNearestNeighbor)
        self.res_rev.SetTransform(sitk.Transform())
        self.res_rev.SetSize(self.size)
        self.res_rev.SetOutputDirection(self.direction)
        self.res_rev.SetOutputOrigin(self.origin)
        self.res_rev.SetDefaultPixelValue(self.default_pixel)
        self.res_rev.SetOutputSpacing(self.spacing)


class MiniDFProcessor():
    def __init__(self,lm_fldr, shell_dia=10):
        store_attr()

    def process_minidf(self,mini ):
        gt_fns = list(self.lm_fldr.glob("*"))
        cents = mini.gt_cent
        lm_fn = mini.gt_fn.tolist()[0]

        if not Path(lm_fn).exists():
            lm_fn = [fn for fn in gt_fns if cid in fn.name][0]
        lm =  sitk.ReadImage(str(lm_fn))
        R = Onions(lm,shell_dia=self.shell_dia)
        origin = R.origin
        spacing = R.spacing
        out_spacing= R.out_spacing
        direction =R.direction
        direction = np.array([direction[0],direction[4],direction[8]])
        if not np.all(direction>0):
            print("Error while processing: ",lm_fn)
            tr() # # need to fix vector
        scale = [a/b for a,b in zip(out_spacing,spacing)]
        try:
            locations =[]
            centres=[]
            for cent in cents:
                    if isinstance(cent,Union[float,int]):
                        location = None
                    else:
                            cent = ast.literal_eval(cent)
                            lm2 = empty_img(lm)
                            vec = []
                            for or_,cen in zip(origin,cent):
                                vec_ = cen-or_
                                vec.append(vec_)
                            vec_arr = [int(vec_*scal) for vec_,scal in zip(vec,scale)]
                            lm2.SetPixel(*vec_arr,1)
                            mask = sitk.MaskImageFilter()
                            lm2_masked = mask.Execute(R.core,lm2)
                            lm2_masked_arr = sitk.GetArrayFromImage(lm2_masked)
                            if lm2_masked_arr.max()==0:
                                location = "core"
                            else:
                                location = "shell"
                    centres.append(vec_arr)
                    locations.append(location)
            mini = mini.assign(location=locations)
            return mini
        except :
                print("Error processing ",lm_fn)



@ray.remote(num_cpus=6, num_gpus=0)
class ClusterDFProcessor(MiniDFProcessor):

    def __init__(self,chunk_mini_df,lm_fldr, shell_dia=10):
        store_attr()
    def process(self):
        results = []
        for mini in pbar(self.chunk_mini_df):
            mini_locs = self.process_minidf(mini )
            results.append(mini_locs)
        return results
# %%

if __name__ == "__main__":

# %%
#SECTION:--------------------putting 'core' or 'shell' info into results dataframe --------------------------------------------------------------------------------------
    fn_df = Path("/s/fran_storage/predictions/litsmc/LITS-933_fixed_mc/results/results_thresh0mm_results.xlsx")
    df =    pd.read_excel(fn_df)
    cids =df.case_id.unique().tolist()

# %%
    fldr_lms = Path("/s/xnat_shadow/crc/lms")
    gt_fns = list(fldr_lms.glob("*"))

# %%


    mini_dfs = []
# %%
    mini_dfs =[]
    for cid in cids:
        mini = df.loc[df.case_id==cid]
        mini_dfs.append(mini)
    chunks_mini_dfs= list(chunks(mini_dfs,8))
    df['location20'] = None
# %%

    actors = [ClusterDFProcessor.remote(chunk_mini_df, fldr_lms,20) for chunk_mini_df in chunks_mini_dfs]
    results = ray.get([actor.process.remote() for actor in actors ])
# %%
    minis=[]
    for res in results:
        mini_df = pd.concat(res,axis=0)
        minis.append(mini_df)
    df_all = pd.concat(minis,axis=0)
# %%
# %%
    for res in results:
        for resi in res:
            df.loc[resi.index,'location10']=resi
# %%
    df_all.to_excel(fn_df.str_replace("mm_results","mm_results2"),index=False)
# %%
    for cid in pbar(cids[120:]):
        cid = cids[0]
        mini = df.loc[df.case_id==cid]
        cents = mini.gt_cent
        lm_fn = mini.gt_fn.tolist()[0]
        if not Path(lm_fn).exists():
            lm_fn = [fn for fn in gt_fns if cid in fn.name][0]


        lm = sitk.ReadImage(str(lm_fn))
        R = Onions(lm)
        origin = R.origin
        spacing = R.spacing
        out_spacing= R.out_spacing
        direction =R.direction
        direction = np.array([direction[0],direction[4],direction[8]])
        if not np.all(direction>0):
            tr() # # need to fix vector

        scale = [a/b for a,b in zip(out_spacing,spacing)]
        locations =[]
        for cent in cents:
            if isinstance(cent,Union[float,int]):
                location = None
            else:
                cent = ast.literal_eval(cent)
                lm2 = empty_img(lm)
                vec = []
                for or_,cen in zip(origin,cent):
                    vec_ = cen-or_
                    vec.append(vec_)
                vec_arr = [int(vec_*scal) for vec_,scal in zip(vec,scale)]
                lm2.SetPixel(*vec_arr,1)
                mask = sitk.MaskImageFilter()
                lm2_masked = mask.Execute(R.shell,lm2)
                lm2_masked_arr = sitk.GetArrayFromImage(lm2_masked)
                if lm2_masked_arr.max()==0:
                    location = "shell"
                else:
                    location = "core"
            locations.append(location)
        df.loc[mini.index,'location']= locations
# %%
        
    df.to_csv(fn_df,index=False)



# %%
    L2  = LabelMapGeometry(lm2)
    L2.nbrhoods
    L.nbrhoods
    get_labels(lm2)
# %%



    cent = cents.iloc[0]
    if not isinstance(cent,float):
        cent = ast.literal_eval(cent)

# %%


    fldr_gt= Path("/s/xnat_shadow/crc/lms")
    lm_fn = "/s/fran_storage/predictions/litsmc/LITS-933/crc_CRC125_20170708_ABDOMEN.nii.gz"

# %%
    L1 = LabelMapGeometry(R.lm, [1])
    L1.nbrhoods
    bboxes = L1.nbrhoods.bbox
# %%
    im = empty_img(R.lm_bin)
# %%
    for bbox in bboxes:
        bbox_start= bbox[:3]
        offset =[b/2 for b in  bbox[3:]]
        # offset =[b for b in  bbox[3:]]
        cent = [int(start+off) for start,off in zip(bbox_start,offset)]


        im.SetPixel(*cent,2)

# %%

    L2 = LabelMapGeometry(im)

    L2.nbrhoods
# %%
    sitk.WriteImage(R.core,"core.nii.gz")
    sitk.WriteImage(R.shell,"shell.nii.gz")
    sitk.WriteImage(R.lm_bin,"lm.nii.gz")
# %%

    lm = R.res_iso.Execute(R.lm_bin)
    lm.GetSize()
    lm2 = R.res_rev.Execute(lm)
    lm2.GetSize()
    view_sitk(R.lm_bin,lm2)


# %%
    gt_fn = find_matching_fn(lm_fn , fldr_gt)
    lm1 = sitk.ReadImage(lm_fn)
    sp = lm.GetSpacing()
    self.out_spacing = [1,1,1]
    sz= lm.GetSize()
    factor = [a/b for a,b in zip(sp,self.out_spacing)]
    outsize = [int(a*b) for a,b in zip(factor,sz)]
# %%
    res = sitk.ResampleImageFilter()
    res.SetInterpolator(sitk.sitkNearestNeighbor)
    res.SetTransform(sitk.Transform())
    res.SetSize(outsize)
    res.SetOutputDirection(lm1.GetDirection())
    res.SetOutputOrigin(lm1.GetOrigin())
    res.SetDefaultPixelValue(lm1.GetPixelIDValue())
    res.SetOutputSpacing(self.out_spacing)
# %%

    lm2 = res.Execute(lm1)
    get_labels(lm2)
    view_sitk(lm2,lm1,'mm')
# %%


    lm_lesions = relabel(lm2,{1:0})
    lm_lesions = to_binary(lm_lesions)
    lm_lesions = to_cc(lm_lesions)
    get_labels(lm_lesions)

    lm_liver = to_binary(lm2)

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
