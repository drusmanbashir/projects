# %%
from label_analysis.markups import MarkupFromLabelmap, MarkupMultipleFiles
import ray
import ast
from pathlib import Path
from label_analysis.geometry import LabelMapGeometry
from label_analysis.helpers import crop_center, empty_img, get_lm_boundingbox, np_to_native, relabel, to_binary, to_cc, to_int
from xnat.object_oriented import *

from label_analysis.utils import align_sitk_imgs
from xnat.object_oriented import *
from fran.utils.fileio import maybe_makedirs, np_to_ni, save_json
from fran.utils.helpers import chunks, find_matching_fn
from fran.utils.string import find_file
from projects.aiscreening.main import add_liver,  apply_tfms_all, compile_tfmd_files,  crop_center_resample, infer_slice_from_str, infer_str_from_slice

def make_mini_dfs(df):
    #list of dfs, one df per case_id
    cids = df.case_id.unique().tolist()
    mini_dfs =[]
    for cid in cids:
        mini = df.loc[df.case_id==cid]
        mini_dfs.append(mini)
    return mini_dfs

def mini_dfs_to_dots(minis,gt_fldr, outfldr , mode,dot_value=2):
    modes = ["shell","core"]
    assert mode in modes, "Select from {}".format(modes)
    for mini in pbar(minis):
        if len(mini)>0:
            fn = mini.gt_fn.iloc[0]
            fn_gt = find_matching_fn(fn,gt_fldr,use_cid=True)
            out_fn  = outfldr/fn_gt.name
            # lm = relabel(lm,{1:0})
            # sitk.WriteImage(lm,str(fn_gt))
            if fn_gt:
                lm = sitk.ReadImage(str(fn_gt))
                lm = empty_img(lm)
                try:
                    D = DotImage(mini.gt_cent,lm,dot_value)
                    print("Lesions in {0}: {1}".format(fn_gt,len(D.cents)))
                    D.put_dots()
                except:
                    tr()
                    # error in minidf . this xtion has bugs
                    O = Onions(lm,shell_dia=10)
                    if mode == "shell":
                        lm_masked = O.mask_shell(lm)
                    else:
                        lm_masked = O.mask_core(lm)
                    L = LabelMapGeometry(lm_masked)
                    cents = L.nbrhoods.cent
                    D = DotImage(cents,lm,dot_value)
                    D.put_dots()
                print("Writing to ",out_fn)
                sitk.WriteImage(D.lm,str(out_fn))
            else:
                print("No file: {}. Skipping".format(fn_gt))



def phy_to_arr_coords(coords,spacing):
    # physical coords eg centre to corresponding array slice
    out_spacing = [1,1,1]
    scale = [a/b for a,b in zip(out_spacing,spacing)]
    coord_arr = [int(vec_*scal) for vec_,scal in zip(coords,scale)]
    return coord_arr

def distance_vector(origin,loc):
 
    vec = []
    for or_,cen in zip(origin,loc):
        vec_ = cen-or_
        vec.append(vec_)
    return vec



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
        self.compute_volumes()
        self.mfil= sitk.MaskImageFilter()
        
    def create_onions(self):
        # i.e., create onions
        self.core = self.eroder.Execute(self.lm_iso)
        self.core = self.res_rev.Execute(self.core)
        self.shell = self.lm_bin- self.core
        self.core = self.res_rev.Execute(self.core)

    def compute_volumes(self):
        fil = sitk.LabelShapeStatisticsImageFilter()
        fil.Execute(self.core)
        self.vol_core = fil.GetPhysicalSize(1)*1e-3

        fil.Execute(self.shell)
        self.vol_shell = fil.GetPhysicalSize(1)*1e-3

    def create_erode_filter(self):
        self.eroder= sitk.BinaryErodeImageFilter()
        self.eroder.SetForegroundValue(1)
        self.eroder.SetKernelRadius(self.shell_dia)


    def mask_core(self, lm):
        lm_masked = self.mfil.Execute(self.core,lm)
        return lm_masked

    def mask_shell(self,lm):
        lm_masked = self.mfil.Execute(self.shell,lm)
        return lm_masked

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


class DotImage():
    def __init__(self,centres, lm,dot_value=2,size='large'):
        sizes = ['small','large']
        assert size in sizes, "Choose one of {}".format(sizes)
        self.size = size

        self.dot_value = dot_value
        self.cents = []
        for cent in centres:
            if isinstance(cent,str):
                cent = ast.literal_eval(cent)
            self.cents.append(cent)
        self.lm = lm
        self.lm = relabel(self.lm,{1:0,2:0,3:0})  #make it empty image 
        self.spacing = self.lm.GetSpacing()
        self.origin= self.lm.GetOrigin()

    def put_dots(self):
        for cent in self.cents:
            vec = distance_vector(self.origin,cent)
            vec_arr = phy_to_arr_coords(vec,self.spacing)
            if self.size == "small":
                self.lm.SetPixel(*vec_arr,self.dot_value)
            else:
                vec_arrs = self.contiguous(vec_arr)
                for vec_arr in vec_arrs:
                    self.lm.SetPixel(*vec_arr,self.dot_value)
        # sitk.WriteImage(lm,str(out_fn))

    def contiguous(self,vec_arr):
        # 8 pixel cube is created around vec_arr
        xm = vec_arr[0]-1
        xp = vec_arr[0]+1
        ym = vec_arr[1]-1
        yp = vec_arr[1]+1
        zm = vec_arr[2]-1
        zp = vec_arr[2]+1

        a = [xm,vec_arr[1],vec_arr[2]]
        b = [xm,yp,vec_arr[2]]
        c = [vec_arr[0],yp,vec_arr[2]]

        d = [vec_arr[0],vec_arr[1],zp]
        e = [xm,vec_arr[1],zp]
        f = [xm,yp,zp]
        g = [vec_arr[0],yp,zp]
        return [vec_arr,a,b,c,d,e,f,g]




class MiniDFProcessor():
    def __init__(self,lm_fldr, shell_dia=10):
        self.lm_fns= list(lm_fldr.glob("*"))
        store_attr(but='lm_fldr')

    def process_minidf(self,mini ):
        cents = mini.gt_cent
        cid = mini.case_id.iloc[0]
        if not isinstance(cid,str):
            cid = cid.item()
        lm_fn = find_file(cid,self.lm_fns)
        try:
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
            locations =[]
            centres=[]
            for cent in cents:

                    if isinstance(cent,Union[float,int]):
                        location = None
                        vec_arr = None
                    else:
                            cent = ast.literal_eval(cent)
                            lm2 = empty_img(lm)
                            vec = distance_vector(origin,cent)
                            vec_arr = [int(vec_*scal) for vec_,scal in zip(vec,scale)]
                            lm2.SetPixel(*vec_arr,1)
                        
                            lm2_masked = R.mask_core(lm2)
                            lm2_masked_arr = sitk.GetArrayFromImage(lm2_masked)
                            if lm2_masked_arr.max()>0:
                                location = "core"
                            else:
                                location = "shell"
                    centres.append(vec_arr)
                    locations.append(location)
            mini = mini.assign(location=locations)
            mini = mini.assign(vol_shell = R.vol_shell)
            mini = mini.assign(vol_core= R.vol_core)
            return mini
        except:
            print("Error Processing ",lm_fn)



@ray.remote(num_cpus=6, num_gpus=0)
class ClusterDFProcessor(MiniDFProcessor):

    def __init__(self,chunk_mini_df,lm_fldr, shell_dia=10):
        super().__init__(lm_fldr,shell_dia)
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
#SECTION:-------------------- SETUP-------------------------------------------------------------------------------------
    outshape = [288, 224, 64]
    outspacing = [1,1,3]
    fldr_main = Path("/s/xnat_shadow/crc/registration_output/june")
    preds_fldr = Path("/s/fran_storage/predictions/litsmc/LITS-933")
    fldr_core = Path("/s/xnat_shadow/crc/registration_output/june/core_all")
    fldr_shell = Path("/s/xnat_shadow/crc/registration_output/june/shell_all")
    fldr_shell_liver=Path("/s/xnat_shadow/crc/registration_output/june/shell_all_liver")
    fldr_core_liver = Path("/s/xnat_shadow/crc/registration_output/june/core_all_liver")
    fldr_core_crop = Path("/s/xnat_shadow/crc/registration_output/june/core_all_crop")
    fldr_shell_crop = Path("/s/xnat_shadow/crc/registration_output/june/shell_crop_all_crop")
    fn_df = Path("/s/fran_storage/predictions/litsmc/LITS-933_fixed_mc/results/results_thresh1mm.xlsx")
    pred_fldr =Path("/s/fran_storage/predictions/litsmc/LITS-933")
    gt_fldr =Path("/s/xnat_shadow/crc/lms")
    gt_fns = list(gt_fldr.glob("*"))
    fldr_core_tfmd = Path("/s/xnat_shadow/crc/registration_output/june/tfmd_core/")
    fldr_shell_tfmd = Path("/s/xnat_shadow/crc/registration_output/june/tfmd_shell/")
    fldr_core_merged = fldr_main/("core_merged")
    fldr_shell_merged = fldr_main/("shell_merged")
    fldrs = [fldr_core,fldr_shell, fldr_core_crop, fldr_shell_crop,fldr_core_tfmd,fldr_core_merged,fldr_core_liver,fldr_shell_liver,fldr_shell_tfmd,fldr_shell_merged]
    maybe_makedirs(fldrs)
# %%
#SECTION:-------------------- CREATING dataframe with locs--------------------------------------------------------------------------------------

    df =    pd.read_excel(fn_df)
    cids =df.case_id.dropna().unique().tolist()
    shell_dia =10
    dia_str  = str(shell_dia)
    mini_dfs = []
# %%
    mini_dfs =[]
    for cid in cids:
        mini = df.loc[df.case_id==cid]
        if len(mini)==0:
            tr()
        mini_dfs.append(mini)
    chunks_mini_dfs= list(chunks(mini_dfs,8))
# %%

    actors = [ClusterDFProcessor.remote(chunk_mini_df, pred_fldr ,shell_dia) for chunk_mini_df in chunks_mini_dfs]
    results = ray.get([actor.process.remote() for actor in actors ])
# %%
    minis=[]
    for res in results:
        mini_df = pd.concat(res,axis=0)
        minis.append(mini_df)
    df_all = pd.concat(minis,axis=0)
    df_all.to_excel(fn_df.str_replace("mm","mm_onions{}".format(dia_str)),index=False)
# %%
#SECTION:-------------------- DOTS ------------------------------------

    df_fn = "/s/fran_storage/predictions/litsmc/LITS-933_fixed_mc/results/results_thresh0mm_results10.xlsx"
    df = pd.read_excel(df_fn)
    df_core = df.loc[df.location=="core"]
    df_shell = df.loc[df.location=="shell"]


    cor_f = Path("/s/xnat_shadow/crc/lms_corrupt")
    fns_cor = list(cor_f.glob("*"))
    cps = list(Path("/s/xnat_shadow/crc/lms_bkp/").glob("*"))

    mini_core = make_mini_dfs(df_core)
    mini_shell = make_mini_dfs(df_shell)
# %%
    mini_dfs_to_dots(minis=mini_shell,gt_fldr=gt_fldr, outfldr = fldr_shell ,mode="shell")
    mini_dfs_to_dots(minis=mini_core,gt_fldr=gt_fldr, outfldr = fldr_core,mode="core")

# %%
    cid = "crc_CRC164"
    minc = [df for df in mini_core if df.case_id.iloc[0]==cid][0]
    mins = [df for df in mini_shell if df.case_id.iloc[0]==cid][0]


# %%
#SECTION:-------------------- add liver  and crop--------------------------------------------------------------------------------------
# %%
    add_liver(fldr_shell,preds_fldr,fldr_shell,True)
    add_liver(fldr_core,preds_fldr,fldr_core,True)

# %%
    crop_center_resample(fldr_core, fldr_core_crop, outspacing, outshape)
    crop_center_resample(fldr_shell, fldr_shell_crop, outspacing, outshape)

# %%
#SECTION:-------------------- Apply tfm--------------------------------------------------------------------------------------

    apply_tfms_all(fldr_core_crop, output_folder=fldr_core_tfmd)
    apply_tfms_all(fldr_shell_crop, output_folder=fldr_shell_tfmd)

# %%
#SECTION:-------------------- Compile tfmd files--------------------------------------------------------------------------------------

    fls_core = list(fldr_core_tfmd.glob("*"))
    fls_shell = list(fldr_shell_tfmd.glob("*"))
    compile_tfmd_files(fls_core, fldr_core_merged, outspacing)
    compile_tfmd_files(fls_shell, fldr_shell_merged, outspacing)

# %%

    fn_core = fldr_core_tfmd
    lm = sitk.ReadImage(str(fn_core))
    M = MarkupFromLabelmap([],0,'liver','red')
    mup = M.process(lm)

    save_json(mup,"lesions.json")



# %%
    fn_shell= fldr_shell_merged/("lesions.nii.gz")
    lm_shell = sitk.ReadImage(str(fn_shell))
    M = MarkupFromLabelmap([],0,'liver','yellow')
    mup2 = M.process(lm_shell)

    save_json(mup2,"lesions_shell.json")
# %%
# %%
#SECTION:-------------------- TROUBLESHOOTING--------------------------------------------------------------------------------------
#
#     mini = minc
#     outfldr = fldr_core
#     fn = mini.gt_fn.iloc[0]
#     fn_gt = find_matching_fn(fn,gt_fldr,use_cid=True)
#     out_fn  = outfldr/fn_gt.name
#         lm = sitk.ReadImage(str(fn_gt))
#         lm = empty_img(lm)
#         try:
#             dot_value = 2
#             D = DotImage(mini.gt_cent,lm,dot_value)
#             print("Lesions in {0}: {1}".format(fn_gt,len(D.cents)))
#             D.put_dots()
# # %%
#   for cent in D.cents:
#   cent = D.cents[0]
#             vec = distance_vector(D.origin,cent)
#             vec_arr = phy_to_arr_coords(vec,D.spacing)
#
#             D.lm.SetPixel(*vec_arr,D.dot_value)
# # %%
# %%

    fn = "/s/xnat_shadow/crc/registration_output/june/tfmd_shell/crc_CRC164_20180213_Abdomen3p0I30f3.nrrd"
    fns = list(fldr_shell_tfmd.glob("*"))

    M = MarkupMultipleFiles([1],0,'liver','red')

    M.process(fns,"lesions_all.json")



    fns_core = list(fldr_core_tfmd.glob("*"))
    M2 = MarkupMultipleFiles([1],0,'liver','green')
    M2.process(fns_core,"lesions_all_core.json")
# %%


    cps = mup2['markups'][0]['controlPoints']
    mups['markups'][0]['controlPoints'].extend(cps)
    save_json(mups,"lesions_all.json")


    len(mups['markups'][0]['controlPoints'])

#SECTION:-------------------- OLD--------------------------------------------------------------------------------------

    for mini_ in make_mini_dfs:
        M = MiniDFProcessor(pred_fldr)
        a= M.process_minidf(mini)
# %%
    lm_preds = list(pred_fldr.glob("*.gz"))
    for cid in pbar(cids[120:]):
        mini = df.loc[df.case_id==cid]
        cents = mini.gt_cent
        lm_fn=  find_file(cid,lm_preds)
        gt_fn = find_file(cid,gt_fns)
        lm_liver = sitk.ReadImage(str(lm_fn))
        lm_liver = relabel(lm_liver,{2:1,3:1})
        R = Onions(lm_liver, 10)
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
# %%
#SECTION:-------------------- ERROR FiX--------------------------------------------------------------------------------------

    fldr_c = Path("/s/xnat_shadow/crc_lms_only/lms_corrupt")
    fldr_shell=  Path("/s/xnat_shadow/crc_lms_only/lms_new")
    fldr_crop = Path("/s/xnat_shadow/crc/cropped/lms")
    fns_c = list(fldr_c.glob("*"))
    fn = "/s/xnat_shadow/crc/lms_some_missing/crc_CRC089_20180509_ABDOMEN.nii.gz"
    lm= sitk.ReadImage(fn)
    get_labels(lm)

# %%
    for fn_c in fns_c:
        fn_out = fldr_shell/fn_c.name
        fn_liver = find_matching_fn(fn_c,Path("/s/fran_storage/predictions/litsmc/LITS-935"),True)
        fn_gt_cr = find_matching_fn(fn_c,fldr_crop,True)
        fn_cropped =find_matching_fn(fn_c,fldr_crop,True)
        lm_cropped = sitk.ReadImage(str(fn_cropped))
        lm_cropped = crop_center(lm_cropped)
        lm_lesions = relabel(lm_cropped,{1:0})
        lesions_array = sitk.GetArrayFromImage(lm_lesions)
        lm_lesions.GetSize()
        bb_o1,sz1 = get_lm_boundingbox(lm_cropped)
        lm_cropped.GetSize()
        lm_liver = sitk.ReadImage(str(fn_liver))
        bb_o,sz = get_lm_boundingbox(lm_liver)
        
        bb_o = bb_o[2],bb_o[1],bb_o[0]
        sz = sz[2],sz[1],sz[0]

        sls = [slice(a,a+b) for a,b in zip(bb_o,lesions_array.shape)]



        lm_neo = empty_img(lm_liver)
        arr = sitk.GetArrayFromImage(lm_neo)
        arr[sls[0],sls[1],sls[2]] = lesions_array
        lm_out  = sitk.GetImageFromArray(arr)
        lm_final = align_sitk_imgs(lm_out,lm_liver)
        sitk.WriteImage(lm_final,str(fn_out))

        



# %%
