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
from label_analysis.utils import is_sitk_file


sys.path += ["/home/ub/code"]
import itertools as il
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
import six
from label_analysis.helpers import *

from fran.transforms.totensor import ToTensorT
from fran.utils.fileio import maybe_makedirs
from fran.utils.helpers import *
from fran.utils.imageviewers import *
from fran.utils.string import (
    find_file,
    info_from_filename,
    match_filenames,
    replace_extension,
    strip_extension,
    strip_slicer_strings,
)

np.set_printoptions(linewidth=250)
np.set_printoptions(formatter={"float": lambda x: "{0:0.2f}".format(x)})


slcs = [
    slice(0, 50),
    slice(50, 100),
    slice(100, 150),
    slice(150, 200),
    slice(200, 250),
    slice(240, 267),
]


def apply_tfm_folder(tfm_fn, input_fldr, output_fldr, slc):
    """
    input_fldr: untransformed files are here
    output_flder: store transformed files here
    """

    maybe_makedirs(output_fldr)
    tfm_fn = str(tfm_fn)
    df = pd.read_csv("/home/ub/code/registration/fnames.csv")
    fn_lms = df["fnames"]
    fn_lms2 = fn_lms[slc].tolist()
    fn_lms2 = [Path(fn) for fn in fn_lms2]
    fn_missed = list(input_fldr.glob("*"))
    fn_missed = [fn for fn in fn_missed if is_sitk_file(fn)]
    fn_missed2 = []
    for fn in fn_lms2:
        fn_missed_ = find_matching_fn(fn, fn_missed, use_cid=True)
        if not fn_missed_:
            fn_missed_ = (
                fn  # there are some extra files in the cropped resampled folder.
            )
        fn_missed2.append(fn_missed_)

    lms_missed2 = [sitk.ReadImage(str(i)) for i in fn_missed2]
    # tfm_fn= "/s/xnat_shadow/crc/registration_output/TransformParameters.{0}.txt".format(tfm_suffix)
    # out_f=Path("/s/xnat_shadow/crc/registration_output/lms_missed_0_50")
    im_lesions = apply_tfm_file(tfm_fn, lms_missed2, is_label=True)
    store_compound_img(im_lesions, out_fldr=output_fldr, fnames=fn_missed2)


def _collate_folders_common_prefix(folder_prefix):
    # folder prefix without trailing underscore e.g., "/s/xnat_shadow/crc/registration_output/lms_all"
    folders = []
    parent_folder = Path("/s/xnat_shadow/crc/registration_output/")
    folder_prefix = folder_prefix + "_"
    for slc in slcs:
        tfm_suffix = str(slc.start) + "_" + str(slc.stop)
        folder_name = folder_prefix + tfm_suffix
        folder_name_full = parent_folder / folder_name
        folders.append(folder_name_full)
    return folders


def collate_files_common_prefix(prefix):

    fldrs_all = _collate_folders_common_prefix(prefix)
    fls_all = []
    for fldr in fldrs_all:
        fls_ = list(fldr.glob("*"))
        fls_all.extend(fls_)
    fls_all = list(set(fls_all))
    return fls_all


def _infer_slice_from_str(string):
    start, end = string.split("_")
    start, end = int(start), int(end)
    slc = slice(start, end)
    return slc


def apply_tfms_all(untfmd_fldr, output_folder_prefix):
    folders = _collate_folders_common_prefix(output_folder_prefix)
    pat = r"\d+_\d+"
    for fldr in folders:
        tfm_suffix = re.search(pat, fldr.name)[0]
        slc = _infer_slice_from_str(tfm_suffix)
        tfm_fn = Path(
            "/s/xnat_shadow/crc/registration_output/TransformParameters.{0}.txt".format(
                tfm_suffix
            )
        )
        assert tfm_fn.exists(), "File not found".format(tfm_fn)
        apply_tfm_folder(
            tfm_fn=tfm_fn, input_fldr=untfmd_fldr, output_fldr=fldr, slc=slc
        )


def add_liver(lesions_fldr, liver_fldr, output_fldr, overwrite=False):
    ms_fns = lesions_fldr.glob("*")
    ms_fns = [fn for fn in ms_fns if is_sitk_file(fn)]
    liver_fns = list(liver_fldr.glob("*"))
    liver_fns = [fn for fn in liver_fns if is_sitk_file(fn)]
    for ms_fn in pbar(ms_fns):
        liver_fn = find_matching_fn(ms_fn, liver_fns, use_cid=True)
        output_fname = output_fldr / (ms_fn.name)
        if overwrite == True or not output_fname.exists():
            MergeLiver = MergeLabelMaps(
                liver_fn,
                ms_fn,
                output_fname=output_fname,
                remapping1={2: 1, 3: 1},
                remapping2={1: 99},
            )
            MergeLiver.process()
            MergeLiver.write_output()


def crop_center_resample(in_fldr, out_fldr, outspacing, outshape):
    # outspacing = [1,1,3]
    # outshape = [288,224,64]
    # out_fldr = Path("/s/xnat_shadow/crc/cropped_resampled_missed_subcm")
    # out_lms_fldr = out_fldr / ("lms")
    # maybe_makedirs(out_lms_fldr)
    fn_lms = list(in_fldr.glob("*.*"))
    pairs = []
    for lm_fn in fn_lms:
        # lm_fn = find_matching_fn(img_fn, fn_lms)
        dici = {"label": str(lm_fn)}
        pairs.append(dici)
    keys = ["label"]
    L = LoadSITKd(keys=keys)
    E = EnsureChannelFirstd(keys=keys, channel_dim="no_channel")
    ScL = Spacingd(keys=keys, pixdim=outspacing, mode="nearest")
    C = CropForegroundd(keys=keys, source_key="label", select_fn=lambda lm: lm > 0)
    Res = ResizeWithPadOrCropd(keys=keys, spatial_size=outshape)
    Sq = SqueezeDimd(keys=keys)
    T = Transposed(keys=keys, indices=[2, 1, 0])
    all_ = Compose([L, E, ScL, C, Res, Sq, T])
    ds = Dataset(data=pairs, transform=all_)
    for dici in pbar(ds):
        lm_tfmd = dici["label"]
        l = sitk.GetImageFromArray(lm_tfmd)
        l.SetSpacing(outspacing)
        fn = lm_tfmd.meta["filename_or_obj"]
        fn = Path(fn)
        print("Processing", fn)
        fn_lm_out = out_fldr / fn.name
        # sitk.WriteImage(i,str(fn_img_out))
        sitk.WriteImage(l, str(fn_lm_out))


class Markups:
    def __init__(self, outfldr, markup_shape=None):
        cups = sitk.ReadImage(
            "/s/xnat_shadow/crc/registration_output/lms_missed_50_100/merged_3cups.nrrd"
        )
        self.shell = relabel(cups, {2: 0})
        self.core = relabel(cups, {3: 0})
        self.outfldr = Path(outfldr)
        self.markup_shape = markup_shape

    def remove_liver(self, lm):
        lm = relabel(lm, {1: 0})
        lm = to_binary(lm)
        return lm

    def process(self, lm_all_fn, lm_det_fn):

        self.case_filename = lm_all_fn
        lm_all = sitk.ReadImage(str(lm_all_fn))
        lm_det = sitk.ReadImage(str(lm_det_fn))

        lm_all_core, lm_all_shell, counts_all = self.process_lm(lm_all)
        lm_det_core, lm_det_shell, counts_det = self.process_lm(lm_det)
        self.dici = {
            "all_counts_core": counts_all[0],
            "all_counts_shell": counts_all[1],
            "det_counts_core": counts_det[0],
            "det_counts_shell": counts_det[1],
        }

        self.markups_all_core = self.create_markups(lm_all_core, "red", "all_core")
        self.markups_all_shell = self.create_markups(
            lm_all_shell, "yellow", "all_shell"
        )
        self.markups_det_core = self.create_markups(lm_det_core, "blue", "det_core")
        self.markups_det_shell = self.create_markups(lm_det_shell, "green", "det_shell")

    def create_json_fn(self, lm_fn, suffix):
        lm_fn_name = strip_extension(lm_fn.name)
        lm_fn_name = "_".join([lm_fn_name, suffix])
        lm_fn_name = lm_fn_name + ".json"
        fn_out = self.outfldr / lm_fn_name
        return fn_out

    def process_lm(self, lm):
        lm = self.remove_liver(lm)
        lm_core, lm_shell = self.split_lm(lm)
        counts = self.get_core_shell_counts(lm_core, lm_shell)
        return lm_core, lm_shell, counts

    def split_lm(self, lm):
        lm_core = sitk.Mask(lm, self.shell, outsideValue=0, maskingValue=3)
        lm_shell = sitk.Mask(lm, self.core, outsideValue=0, maskingValue=2)
        return lm_core, lm_shell

    def get_core_shell_counts(self, lm_core, lm_shell):
        LC = LabelMapGeometry(lm_core)
        count_core = len(LC)

        LS = LabelMapGeometry(lm_shell)
        count_shell = len(LS)
        return count_core, count_shell

    def create_markups(self, lm, color, fn_suffix):
        M = MarkupFromLabelmap([], 0, "auto", color)
        a = M.process(lm)
        if self.markup_shape is not None:
            a["markups"][0]["display"]["glyphType"] = self.markup_shape
        fn = self.create_json_fn(self.case_filename, fn_suffix)
        save_json(a, fn)
        return a


def merge_markups(fns):

    mups_full = load_json(fn)
    header = mups_full["@schema"]
    mups_base = mups_full["markups"]
    mups_base[0]["display"]["glyphSize"] = 5.0
    mups_base[0]["display"]["glyphScale"] = 1.0
    for fn in fns[1:]:
        mups_tmp = load_json(fn)
        header = mups_tmp["@schema"]
        mups = mups_tmp["markups"]
        cps = mups[0]["controlPoints"]
        mups_base[0]["controlPoints"].extend(cps)
    dici = {"@schema": header, "markups": mups_base}
    return dici


def compile_tfmd_files(fns, outfldr, outspacing):

    excludes = ["lesions", "liver", "merged", "react"]
    for exclude in excludes:
        fns = [fn for fn in fns if exclude not in fn.name]
    reference_fldr = Path("/s/xnat_shadow/crc/cropped_resampled_missed_subcm/lms/")
    ref_files = list(reference_fldr.glob("*"))
    fns_final = []
    for fn in fns:
        fn2 = find_matching_fn(fn, ref_files)
        if fn2:
            fn_already = find_matching_fn(fn, fns_final)
            if not fn_already:
                fns_final.append(fn)
    fns_final = set(fns_final)
    print("Total files to compile", len(fns_final))
    lms = [sitk.ReadImage(str(i)) for i in fns_final]
    lms_noliver = [relabel(lm, {1: 0}) for lm in lms]
    lms_nl = create_vector(lms_noliver)
    lms_ar = compound_to_np(lms_nl)
    lms_le_ar = np.sum(lms_ar, 0)

    lms_liver = [to_binary(lm) for lm in lms]
    lms_l = create_vector(lms_liver)

    lms_li_ar = compound_to_np(lms_l)
    lms_li_ar = np.mean(lms_li_ar, 0)
    lms_li_ar[lms_li_ar > 0.02] = 1

    lms_le = sitk.GetImageFromArray(lms_le_ar)
    lms_le.SetSpacing(outspacing)
    sitk.WriteImage(lms_le, str(outfldr / "lesions.nii.gz"))

    lms_li = sitk.GetImageFromArray(lms_li_ar)
    lms_li.SetSpacing(outspacing)
    sitk.WriteImage(lms_li, str(outfldr / "liver.nii.gz"))

    lms_merged_ar = lms_li_ar + lms_le_ar
    lms_merged = sitk.GetImageFromArray(lms_merged_ar)
    lms_merged.SetSpacing(outspacing)
    print("Writing merged.nii.gz")
    sitk.WriteImage(lms_merged, str(outfldr / "merged.nii.gz"))


# %%
if __name__ == "__main__":
    # SECTION:-------------------- SETUP --------------------------------------------------------------------------------------
    outspacing = [1, 1, 3]
    outshape = [288, 224, 64]

    parent = Path("/s/xnat_shadow/crc/registration_output/")

    mup_fldr = parent / ("markups")
    msb_fldr = Path(
        "/s/fran_storage/predictions/litsmc/LITS-933_fixed_mc/missed_subcm_binary/"
    )
    dsb_fldr = Path(
        "/s/fran_storage/predictions/litsmc/LITS-933_fixed_mc/detected_subcm_binary/"
    )
    ds_fldr = Path(
        "/s/fran_storage/predictions/litsmc/LITS-933_fixed_mc/detected_subcm/"
    )
    as_fldr = Path("/s/fran_storage/predictions/litsmc/LITS-933_fixed_mc/all_subcm/")
    ms_fldr = Path("/s/fran_storage/predictions/litsmc/LITS-933_fixed_mc/missed_subcm")
    maybe_makedirs([ms_fldr, msb_fldr, dsb_fldr, ds_fldr, as_fldr])
    preds_fldr = Path("/s/fran_storage/predictions/litsmc/LITS-933")
    msl_fldr = Path(
        "/s/fran_storage/predictions/litsmc/LITS-933_fixed_mc/missed_subcm_with_liver"
    )
    asl_fldr = Path(
        "/s/fran_storage/predictions/litsmc/LITS-933_fixed_mc/all_subcm_with_liver"
    )
    dsl_fldr = Path(
        "/s/fran_storage/predictions/litsmc/LITS-933_fixed_mc/detected_subcm_with_liver"
    )
    maybe_makedirs([msl_fldr, asl_fldr, dsl_fldr])
    mslc_fldr = Path("/s/xnat_shadow/crc/cropped_resampled_missed_subcm")
    aslc_fldr = Path("/s/xnat_shadow/crc/cropped_resampled_all_subcm")
    imgs_fldr = mslc_fldr / ("images")
    mslc_lms_fldr = mslc_fldr / ("lms")
    dslc_fldr = Path("/s/xnat_shadow/crc/cropped_resampled_detected_subcm")
    dslc_lms_fldr = dslc_fldr / ("lms")
    maybe_makedirs([aslc_fldr, mslc_lms_fldr, dslc_fldr, dslc_lms_fldr])

# %%
# %%
    # NOTE: Missed lesions collate into: a.All sub-cm, b.missed subcm c.detected subcm

    gt_fldr = Path("/s/xnat_shadow/crc/lms")
    gt_fns = list(gt_fldr.glob("*"))
    gt_fns = [fn for fn in gt_fns if is_sitk_file(fn)]

    imgs_fldr = Path("/s/xnat_shadow/crc/completed/images")

    results_df = pd.read_excel(
        "/s/fran_storage/predictions/litsmc/LITS-933_fixed_mc/results/results_thresh0mm.xlsx"
    )

# %%
    # lm_fn = [fn for fn in gt_fns if cid in fn.name][0]
    for fn_ms in pbar(gt_fns[15:]):
        fn_ms = ms_fldr / fn_ms.name
        fn_as = as_fldr / fn_ms.name
        fn_ds = ds_fldr / fn_ms.name
        cid = info_from_filename(fn_ms.name, full_caseid=True)["case_id"]
        sub_df = results_df[results_df["case_id"] == cid]
        sub_df = sub_df[sub_df["fk"] > 0]

        lm = sitk.ReadImage(str(fn_ms))
        L = LabelMapGeometry(lm)
        if L.is_empty():
            shutil.copy(fn_ms, str(fn_ms))
            shutil.copy(fn_ms, str(fn_ds))
            shutil.copy(fn_ms, str(fn_as))
        else:
            excluded = L.nbrhoods[L.nbrhoods["length"] > 10]
            excluded2 = excluded["label_cc"].tolist()
            remapping_exc = {x: 0 for x in excluded2}
            L.lm_cc = relabel(L.lm_cc, remapping_exc)
            sitk.WriteImage(L.lm_cc, str(fn_as))

            all_subcm = sub_df[sub_df["gt_length"] <= 10]
            if len(all_subcm) == 0:
                sitk.WriteImage(L.lm_cc, str(fn_ds))
                sitk.WriteImage(L.lm_cc, str(fn_ms))
            else:
                missed = sub_df[sub_df["dsc"].isna()]
                missed = missed[missed["gt_length"] <= 10]
                cents_missed = missed["gt_cent"].tolist()
                cents_missed = [ast.literal_eval(c) for c in cents_missed]
                if len(missed) > 0:
                    missed_nbr = L.nbrhoods[L.nbrhoods["cent"].isin(cents_missed)]
                    missed_labs = missed_nbr["label_cc"].tolist()
                    remapping_detected = {x: 0 for x in missed_labs}

                    detected_labs = L.nbrhoods[
                        ~L.nbrhoods["label_cc"].isin(missed_labs)
                    ]
                    detected_labs = detected_labs["label_cc"].tolist()
                    remapping_missed = {x: 0 for x in detected_labs}

                    lm_missed = relabel(L.lm_cc, remapping_missed)
                    sitk.WriteImage(lm_missed, str(fn_ms))

                    lm_missed_binary = to_binary(lm_missed)
                    sitk.WriteImage(lm_missed_binary, str(msb_fldr / fn_ms.name))

                    lm_detected = relabel(L.lm_cc, remapping_detected)
                    sitk.WriteImage(lm_detected, str(ds_fldr / fn_ms.name))
                    lm_detected_binary = to_binary(lm_detected)
                    sitk.WriteImage(lm_detected_binary, str(dsb_fldr / fn_ms.name))
                else:  # if no lesions were missed we should have an empty 'missed lesions' lm
                    # removing all labels
                    sitk.WriteImage(L.lm_cc, str(fn_ds))
                    remapping = {x: 0 for x in L.labels}
                    L.lm_cc = relabel(L.lm_cc, remapping)
                    sitk.WriteImage(L.lm_cc, str(ms_fldr / fn_ms.name))
                    # else:
                #     tr()
    # SECTION:-------------------- Add liver  --------------------------------------------------------------------------------------

    add_liver(ms_fldr, preds_fldr, msl_fldr)
    add_liver(as_fldr, preds_fldr, asl_fldr, True)
    add_liver(ds_fldr, preds_fldr, dsl_fldr, True)

# %%
    # SECTION:-------------------- CROP CENTER AND RESAMPLE- ---------------------

    crop_center_resample(asl_fldr, aslc_fldr, outspacing, outshape)
    crop_center_resample(dsl_fldr, dslc_fldr, outspacing, outshape)
    crop_center_resample(msl_fldr, mslc_fldr, outspacing, outshape)

# %%
    # SECTION:-------------------- Apply tfms iteratively (5 tfms)--------------------------------------------------------------------------------------'

    apply_tfms_all(aslc_fldr, output_folder_prefix="lms_all")
    apply_tfms_all(dslc_fldr, output_folder_prefix="lms_ds")
    apply_tfms_all(mslc_fldr, output_folder_prefix="lms_ms")
    # apply_tfm_folder(tfm_fn,mslc_lms_fldr,out_f_ms,slc)

    # compile_tfmd_files(out_f_ms)
# %%
# %%
    # SECTION:--------------------Super merge ALL merged files (1 merged file per tfm) --------------------------

    outfldr_missed = Path("/s/xnat_shadow/crc/registration_output/lms_ms_allfiles")
    outfldr_all = Path("/s/xnat_shadow/crc/registration_output/lms_all_allfiles")
    outfldr_detected = Path("/s/xnat_shadow/crc/registration_output/lms_ds_allfiles")
    maybe_makedirs([outfldr_missed, outfldr_all, outfldr_detected])

    fls_ms = collate_files_common_prefix("lms_ms")
    fls_all = collate_files_common_prefix("lms_all")
    fls_det = collate_files_common_prefix("lms_ds")

# %%
    compile_tfmd_files(fls_ms, outfldr_missed, outspacing)

    compile_tfmd_files(fls_all, outfldr_all, outspacing)

    compile_tfmd_files(fls_det, outfldr_detected, outspacing)
# %%
    # SECTION:--------------------CREATE CORE VERSUS SHELL MARKUPS -----------------------------------------

# %%

    dicis = []
    for fn_det in fls_det:
        cid = info_from_filename(fn_det.name, full_caseid=True)["case_id"]
        fn_all = [fn for fn in fls_all if fn.name == fn_det.name][0]
        M = Markups(outfldr=mup_fldr, markup_shape="Sphere3D")
        M.process(fn_all, fn_det)
        M.dici["case_id"] = cid
        dicis.append(M.dici)

# %%
# %%
    df = pd.DataFrame(dicis)
    print(df)
    df.to_csv(parent / ("mups.csv"), index=False)

# %%
    mup_jsons = list(mup_fldr.glob("*"))
    jsons_all_core = [fn for fn in mup_jsons if "all_core" in fn.name]
    merged_all_core = merge_markups(jsons_all_core)
    save_json(merged_all_core, mup_fldr / ("all_core.json"))

    jsons_all_shell = [fn for fn in mup_jsons if "all_shell" in fn.name]
    merged_all_shell = merge_markups(jsons_all_shell)
    save_json(merged_all_shell, mup_fldr / ("all_shell.json"))

    jsons_det_core = [fn for fn in mup_jsons if "det_core" in fn.name]
    merged_det_core = merge_markups(jsons_det_core)
    save_json(merged_det_core, mup_fldr / ("det_core.json"))

    jsons_det_shell = [fn for fn in mup_jsons if "det_shell" in fn.name]
    merged_det_shell = merge_markups(jsons_det_shell)
    save_json(merged_det_shell, mup_fldr / ("det_shell.json"))
# %%
    fns = [fn for fn in mup_jsons if cid in fn.name]
    fn_all_core = [fn for fn in fns if "all_core" in fn.name][0]
    fn_all_shell = [fn for fn in fns if "all_shell" in fn.name][0]
    fn_det_shell = [fn for fn in fns if "det_shell" in fn.name][0]
    fn_det_core = [fn for fn in fns if "det_core" in fn.name][0]
    mups_all_core = load_dict(fn_all_core)
    header = mups_all_core["@schema"]
    mups = mups_all_core["markups"]
    mups[0]["display"]["glyphSize"] = 5.0
    mups[0]["display"]["glyphScale"] = 1.0
    mups_all.append(mups)

# %%

# %%
    # SECTION:-------------------- Batch scoring dsc--------------------------------------------------------------------------------------
    lesion_masks_folder = Path("/s/xnat_shadow/crc/wxh/masks_manual_todo")
    lm_final_fldr = Path("/s/xnat_shadow/crc/wxh/masks_manual_final/")
    lesion_masks_folder2 = Path("/s/xnat_shadow/crc/srn/cases_with_findings/masks")
    marksups_fldr = Path("/s/xnat_shadow/crc/wxh/markups/")

    fnames_lab = list(lesion_masks_folder.glob("*"))
    fnames_final = list(lm_final_fldr.glob("*"))
    set(fnames_lab).difference(fnames_final)
    msl_fldr = lesion_masks_folder.parent / ("masks_manual_final")
    maybe_makedirs(msl_fldr)
# %%
    fnames_json = list(marksups_fldr.glob("*"))
    for fn_j in fnames_json:
        cid = info_from_filename(fn_j.name)["case_id"]
        fn_ms = [fn for fn in fnames_lab if cid in fn.name]

        fn_ms = fn_ms[0]
        lm_fn_out = msl_fldr / (fn_ms.name)
        R = RemapFromMarkup(organ_label=None)
        R.process(fn_ms, lm_fn_out, fn_j)

# %%
    # fnames_lab = list(lesion_masks_folder.glob("*"))
# %%
    imgs_fldr = Path("/s/xnat_shadow/crc/wxh/images/")
    imgs = list(imgs_fldr.glob("*"))
    preds_fldr = Path("/s/xnat_shadow/crc/srn/cases_with_findings/preds_fixed")
    lm_final = list(lm_final_fldr.glob("*"))

    pending = []
    for img_fn in imgs:
        case_id = info_from_filename(img_fn.name)["case_id"]
        lm_done = [fn for fn in lm_final if case_id in fn.name]
        if len(lm_done) == 0:
            done = False
        else:
            done = True
        pending.append(not done)

# %%
    imgs_pending = list(il.compress(imgs, pending))
    # img = imgs_pending[0]
    # case_id = info_from_filename(img.name)['case_id']
    # case_id = "crc_"+case_id
    # row = df.loc[df.case_id==case_id]
    # colnames = list(df.columns)+["disparity"]
    df2 = pd.DataFrame(columns=df.columns)
    df2["disparity"] = 0

# %%

    fnames_lab = list(lesion_masks_folder.glob("*"))
    R = RemapFromMarkup(organ_label=None)
    # for idx in range(len(df)):
    for idx, img in enumerate(imgs):
        case_id = info_from_filename(img.name)["case_id"]
        case_id = "crc_" + case_id
        row = df.loc[df.case_id == case_id]
        # idx = 0
        # print("---",idx)
        # row = df.loc[idx]
        row2 = row.copy()
        row2["disparity"] = 0
        # row2.columns = colnames
        print(row.labels)

        case_id = row.case_id.item()
        lab = row.labels.item()

        if lab == "exclude" or lab == "done":
            print("Case excluded/ done:  ", case_id)
            pass
        else:
            fn_lm = [fn for fn in fnames_lab if case_id in fn.name]
            if len(fn_lm) != 1:
                tr()
            else:
                fn_lm = fn_lm[0]

            fn_out = msl_fldr / (fn_lm.name)

            if fn_out.exists():
                print("File exists: ", fn_out)
                pass

            else:
                lm = sitk.ReadImage(fn_lm)
                lg = LabelMapGeometry(lm, ignore_labels=[])
                if lab == "normal":
                    if len(lg) != 0:
                        row2["disparity"] = 1
                        print(lg.nbrhoods.label)
                        if all(lg.nbrhoods.label == 2):
                            print("Labels are benign")
                            row2.labels = "benign"
                            shutil.copy(fn_lm, fn_out)
                        else:
                            tr()
                            remapper = {1: 2}
                            lm = relabel(lm, remapper)
                            sitk.WriteImage(lm, fn_out)
                    else:
                        row2["disparity"] = 0
                        shutil.copy(fn_lm, fn_out)
                    df2.loc[idx] = row2.iloc[0]

                elif lab == "benign":
                    if all(lg.nbrhoods.label == 2):
                        shutil.copy(fn_lm, fn_out)

                    elif all(lg.nbrhoods.label == 1):
                        remapper = {1: 2}
                        lm = relabel(lm, remapper)
                        sitk.WriteImage(lm, fn_out)
                    elif len(lg) == 0:
                        row2.labels = "normal"
                        row2.disparity = 1
                        shutil.copy(fn_lm, fn_out)
                    else:

                        tr()

                    df2.loc[idx] = row2.iloc[0]
                elif lab == "done":
                    tr()

                    remapper = {1: 2, 2: 3}
                    lm = relabel(lm, remapper)
                    sitk.WriteImage(lm, fn_out)

                elif lab == "mets":
                    if all(lg.nbrhoods.label == 3):
                        shutil.copy(fn_lm, fn_out)
                        df2.loc[idx] = row2.iloc[0]
                    elif all(lg.nbrhoods.label == 1):
                        remapper = {1: 3}
                        lm = relabel(lm, remapper)
                        sitk.WriteImage(lm, fn_out)

                    else:
                        tr()
                elif lab == "json" or lab == "markup":
                    fn_js = [fn for fn in fnames_json if case_id in fn.name]
                    if len(fn_js) == 1:
                        R.process(fn_ms, lm_fn_out, fn_js[0])
                    else:
                        tr()

# %%
# %%
    exc_pat = "_\d\.nii"
    preds_fldr = Path(
        "/s/fran_storage/predictions/litsmc/LITS-787_LITS-810_LITS-811_fixed_mc/"
    )
    for fn in preds_fldr.glob("*"):
        # fn = find_file("CRC018",preds_fldr)

        if fn.is_dir() or re.search(exc_pat, fn.name):
            print("Skipping ", fn)
        else:
            F = FixMulticlass_CC(fn, 3, overwrite=True)

# %%
    gt_fldr = Path("/s/xnat_shadow/crc/lms/")
    imgs_fldr = Path("/s/xnat_shadow/crc/images")

    gt_fns = list(gt_fldr.glob("*"))


# %%
    do_radiomics = False

# %%
    B = BatchScorer(
        gt_fns,
        imgs_fldr=imgs_fldr,
        preds_fldr=preds_fldr,
        debug=False,
        do_radiomics=False,
    )  # ,output_fldr=Path("/s/fran_storage/predictions/litsmc/LITS-787_mod/results"))
    B.process()
# %%
    df = pd.read_csv(B.output_fn)
    excluded = list(pd.unique(df["gt_fn"].dropna()))
# %%
    cids = np.array([info_from_filename(fn.name)["case_id"] for fn in gt_fns])
    news = []
    dups = []
    for id in cids:
        if id not in news:
            news.append(id)
        else:
            dups.append(id)

# %%
    case_subid = "CRC234"
    gt_fn = find_file(case_subid, gt_fns)
    lm = sitk.ReadImage(str(gt_fn))
    L = LabelMapGeometry(lm)

    sitk.WriteImage(L.lm_cc, "tmp.nii.gz")


    pred_fn = find_file(case_subid, preds_fldr)
# %%
    do_radiomics = False
    S = ScorerAdvanced(
        gt_fn,
        pred_fn,
        img_fn=None,
        ignore_labels_gt=[],
        ignore_labels_pred=[1],
        save_matrices=False,
        do_radiomics=do_radiomics,
        dusting_threshold=0,
    )
    df = S.process()
# %%

    predicted_masks_folder = Path(
        "/s/xnat_shadow/crc/srn/cases_with_findings/preds_fixed"
    )
    predicted_masks_folder = Path("/s/xnat_shadow/crc/srn/cases_with_findings/preds")
    lesion_masks_folder = Path(
        "/s/xnat_shadow/crc/srn/cases_with_findings/masks_lesions_are_label1"
    )
    lesion_masks_folder = Path(
        "/s/xnat_shadow/crc/srn/cases_with_findings/masks_no_liver/"
    )
# %%
    mapping = {1: 2}
    for fn_ms in lesion_masks_folder.glob("*"):
        lm = sitk.ReadImage(fn_ms)
        lm = to_label(lm)
        lm = sitk.ChangeLabelLabelMap(lm, mapping)
        lm = to_int(lm)
        sitk.WriteImage(lm, fn_ms)

# %%

    mask_fns = list(lesion_masks_folder.glob("*"))
    pred_fns = list(predicted_masks_folder.glob("*"))
# %%
    msl_fldr = Path(
        "/s/xnat_shadow/crc/srn/cases_with_findings/masks_final_with_liver/"
    )
    for mask_fn in mask_fns:
        # mask_fn = mask_fns[0]
        output_fn = msl_fldr / (mask_fn.name)
        if not output_fn.exists():
            pred_fn = find_matching_fn(mask_fn, pred_fns)
            M = MergeLabelMaps(pred_fn, mask_fn, output_fn)

            M.process()
        else:
            print("File exists: ", output_fn)
# %%
    out_fns = []
    imgs_fldr = Path("/s/xnat_shadow/crc/srn/cases_with_findings/images_done/")
    img_fns = list(imgs_fldr.glob("*"))
    for mask_fn in msl_fldr.glob("*"):
        out_fns.append(find_matching_fn(mask_fn, img_fns))

# %%
    predicted_masks_folder = Path(
        "/s/xnat_shadow/crc/srn/cases_with_findings/preds_fixed"
    )

    pred_fns = list(predicted_masks_folder.glob("*"))
# %%
    for pred_fn in pred_fns:
        fn_out = msl_fldr / (pred_fn.name)
        if fn_out.exists():
            print("File exists skipping", fn_out)
        else:
            case_id = info_from_filename(pred_fn.name)["case_id"]
            case_id = "crc_" + case_id
            row = df.loc[df.case_id == case_id]
            lab = row.labels.item()
            if lab == "benign":
                labels_expected = [1, 2]
            elif lab == "mets":
                labels_expected = [1, 3]
            else:
                labels_expected = [1, 2, 3]

            lm = sitk.ReadImage(pred_fn)
            labels = get_labels(lm)
            if labels == labels_expected:
                print("files concur", pred_fn)
                shutil.copy(pred_fn, fn_out)
            else:
                remapping = {2: 3, 3: 2}
                tr()
                lm = relabel(lm, remapping)
                sitk.WriteImage(lm, fn_out)

# %%
    fn = "/s/xnat_shadow/crc/wxh/masks_manual_final/crc_CRC159_20161122_ABDOMEN-Segment_1-label.nrrd"
    lm = sitk.ReadImage(fn)

    lg = LabelMapGeometry(lm, ignore_labels=[])
# %%

    fnames_lab = list(lesion_masks_folder.glob("*"))
    fnames_lab = [fn.name for fn in fnames_lab]
    fnames_final = list(lm_final_fldr.glob("*"))
    fnames_final = [fn.name for fn in fnames_final]
    pending_m = list(set(fnames_lab).difference(fnames_final))

    pp(pending_m)
# %%
