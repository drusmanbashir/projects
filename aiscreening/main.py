from typing import Any, Dict, List, Optional, Tuple
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
from utilz.helpers import pbar
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
from utilz.fileio import load_json, maybe_makedirs, save_json
from utilz.helpers import find_matching_fn
from utilz.imageviewers import view_sitk, ImageMaskViewer
import ast
from functools import reduce
import sys
import shutil
from label_analysis.geometry import LabelMapGeometry
from label_analysis.merge import MergeLabelMaps

from label_analysis.overlap import BatchScorer, ScorerAdvanced, ScorerFiles
from label_analysis.remap import RemapFromMarkup


sys.path += ["/home/ub/code"]
import itertools as il
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
import six
from label_analysis.helpers import *

from fran.transforms.totensor import ToTensorT
from utilz.fileio import maybe_makedirs
from utilz.helpers import *
from utilz.imageviewers import *
from utilz.string import (
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


def apply_tfm_folder(
    tfm_fn: Union[str, Path],
    input_fldr: Union[str, Path],
    output_fldr: Union[str, Path],
    slc: slice,
    is_label: bool = True,
) -> None:
    """
    Applies transformations to a folder of images using a specified transform file.

    Parameters:
    -----------
    tfm_fn : Union[str, Path]
        File path to the transformation parameters.
    input_fldr : Union[str, Path]
        Directory of untransformed files.
    output_fldr : Union[str, Path]
        Directory to store transformed files.
    slc : slice
        Slice object to select specific files.
    is_label : bool
        Flag to indicate if the data is label data. Default is True.
    """
    maybe_makedirs(output_fldr)
    tfm_fn = str(tfm_fn)
    df = pd.read_csv("/home/ub/code/registration/fnames.csv")
    fn_lms = df["fnames"]
    fn_lms2 = fn_lms[slc].tolist()
    fn_lms2 = [Path(fn) for fn in fn_lms2]
    fns = list(input_fldr.glob("*"))
    fns = [fn for fn in fns if is_sitk_file(fn)]
    tmplt_lm = sitk.ReadImage(str(fns[0]))
    fn_all = []
    for fn in fn_lms2:
        fn_missed_ = find_matching_fn(fn, fns, use_cid=True)
        fn_all.append(fn_missed_)

    lms = []
    for fn in fn_all:
        if fn is None:
            lm = empty_img(tmplt_lm)
        else:
            lm = sitk.ReadImage(str(fn))
        lms.append(lm)
    im_lesions = apply_tfm_file(tfm_fn, lms, is_label=is_label)
    store_compound_img(im_lesions, out_fldr=output_fldr, fnames=fn_all)


def folder_names_common_prefix(folder_prefix: str) -> List[Path]:
    """
    Generates a list of folder paths with a common prefix.

    Parameters:
    -----------
    folder_prefix : str
        Folder prefix without trailing underscore (e.g., "/s/xnat_shadow/crc/registration_output/lms_all").

    Returns:
    --------
    List[Path]
        List of generated folder paths.
    """
    folders = []
    parent_folder = Path("/s/xnat_shadow/crc/registration_output/")
    folder_prefix = folder_prefix + "_"
    for slc in slcs:
        tfm_suffix = str(slc.start) + "_" + str(slc.stop)
        folder_name = folder_prefix + tfm_suffix
        folder_name_full = parent_folder / folder_name
        folders.append(folder_name_full)
    return folders


def collate_files_common_prefix(prefix: str) -> List[Path]:
    """
    Collates files from folders with a common prefix.

    Parameters:
    -----------
    prefix : str
        Common prefix for folder names.

    Returns:
    --------
    List[Path]
        List of file paths collated from folders with the common prefix.
    """
    fldrs_all = folder_names_common_prefix(prefix)
    fls_all = []
    for fldr in fldrs_all:
        fls_ = list(fldr.glob("*"))
        fls_all.extend(fls_)
    fls_all = list(set(fls_all))
    return fls_all


def infer_slice_from_str(string: str) -> slice:
    """
    Infers a slice object from a string.

    Parameters:
    -----------
    string : str
        String representation of the slice (e.g., "10_20").

    Returns:
    --------
    slice
        Slice object inferred from the string.
    """
    start, end = map(int, string.split("_"))
    return slice(start, end)


def infer_str_from_slice(slc: slice) -> str:
    """
    Infers a string representation from a slice object.

    Parameters:
    -----------
    slc : slice
        Slice object.

    Returns:
    --------
    str
        String representation of the slice (e.g., "10_20").
    """
    return f"{slc.start}_{slc.stop}"


def apply_tfms_all(
    untfmd_fldr: Union[str, Path], output_folder: Union[str, Path]
) -> None:
    """
    Applies transformations to all specified slices in a folder.

    Parameters:
    -----------
    untfmd_fldr : Union[str, Path]
        Folder containing untransformed files.
    output_folder : Union[str, Path]
        Folder where the transformed files will be stored.
    """
    for slc in slcs:
        tfm_suffix = infer_str_from_slice(slc)

        tfm_fn = Path(
            f"/s/xnat_shadow/crc/registration_output/TransformParameters.{tfm_suffix}.txt"
        )
        assert tfm_fn.exists(), f"File not found: {tfm_fn}"
        apply_tfm_folder(
            tfm_fn=tfm_fn, input_fldr=untfmd_fldr, output_fldr=output_folder, slc=slc
        )


def add_liver(
    lesions_fldr: Union[str, Path],
    liver_fldr: Union[str, Path],
    output_fldr: Union[str, Path],
    overwrite: bool = False,
) -> None:
    """
    Adds liver segmentation to lesion segmentations and stores the results.

    Parameters:
    -----------
    lesions_fldr : Union[str, Path]
        Folder containing lesion segmentations.
    liver_fldr : Union[str, Path]
        Folder containing liver segmentations.
    output_fldr : Union[str, Path]
        Folder where the output will be stored.
    overwrite : bool
        Flag to overwrite existing files. Default is False.
    """
    ms_fns = [fn for fn in lesions_fldr.glob("*") if is_sitk_file(fn)]
    liver_fns = [fn for fn in liver_fldr.glob("*") if is_sitk_file(fn)]
    for ms_fn in pbar(ms_fns):
        liver_fn = find_matching_fn(ms_fn, liver_fns, use_cid=True)
        output_fname = output_fldr / ms_fn.name
        if overwrite or not output_fname.exists():
            MergeLiver = MergeLabelMaps(
                liver_fn,
                ms_fn,
                output_fname=output_fname,
                remapping1={2: 1, 3: 1},
                remapping2={1: 99},
            )
            MergeLiver.process()
            MergeLiver.write_output()


def crop_center_resample(
    in_fldr: Union[str, Path],
    out_fldr: Union[str, Path],
    outspacing: List[float],
    outshape: List[int],
    mode: str = "nearest",
) -> None:
    """
    Crops, centers, and resamples images in a folder.

    Parameters:
    -----------
    in_fldr : Union[str, Path]
        Input folder containing the images.
    out_fldr : Union[str, Path]
        Output folder where the processed images will be saved.
    outspacing : List[float]
        Desired output spacing.
    outshape : List[int]
        Desired output shape.
    mode : str
        Interpolation mode ('nearest', 'linear', etc.). Default is 'nearest'.
    """
    fn_lms = list(in_fldr.glob("*.*"))
    pairs = [{"label": str(lm_fn)} for lm_fn in fn_lms]
    keys = ["label"]
    L = LoadSITKd(keys=keys)
    E = EnsureChannelFirstd(keys=keys, channel_dim="no_channel")
    ScL = Spacingd(keys=keys, pixdim=outspacing, mode=mode)
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
        sitk.WriteImage(l, str(fn_lm_out))


class Markups:
    """
    This class loads a NIfTI image with core (label 2) and shell (label 3) regions, uses these regions to mask input labels, and processes them
    to separate core and shell parts. It also creates visual markup representations for these segments.

    Attributes:
    -----------
    shell : SimpleITK.Image
        Remapped image where only the shell region is retained.
    core : SimpleITK.Image
        Remapped image where only the core region is retained.
    outfldr : Path
        Directory where output files will be stored.
    markup_shape : Optional[str]
        Shape of the markup glyphs.

    Methods:
    --------
    remove_liver(lm)
        Removes the liver region from the label image.
    process(lm_all_fn, lm_det_fn)
        Processes two label images to extract core and shell regions, and creates markup representations.
    create_json_fn(lm_fn, suffix)
        Creates a JSON filename based on input filename and suffix.
    process_lm(lm)
        Processes a label image to remove liver and split into core and shell segments.
    split_lm(lm)
        Splits the label image into core and shell parts.
    get_core_shell_counts(lm_core, lm_shell)
        Gets the count of core and shell regions.
    create_markups(lm, color, fn_suffix)
        Creates markup representations for a label region and saves it to a JSON file.

    Examples:
    ---------
    >>> markups = Markups(outfldr='/output/folder')
    >>> markups.process('/path/to/lm_all.nii', '/path/to/lm_det.nii')
    """

    def __init__(self, outfldr: Union[str, Path], markup_shape: Optional[str] = None):
        """
        Initializes the Markups class with the given output folder and optional markup shape.

        Parameters:
        -----------
        outfldr : Union[str, Path]
            The directory where output files will be saved.
        markup_shape : Optional[str]
            Shape of the markup glyphs. Default is None.
        """
        cups = sitk.ReadImage(
            "/s/xnat_shadow/crc/registration_output/lms_missed_50_100/merged_3cups.nrrd"
        )
        self.shell = relabel(cups, {2: 0})
        self.core = relabel(cups, {3: 0})
        self.outfldr = Path(outfldr)
        self.markup_shape = markup_shape

    def remove_liver(self, lm: sitk.Image) -> sitk.Image:
        """
        Removes the liver region (label 1) from the label image.

        Parameters:
        -----------
        lm : sitk.Image
            The input label image.

        Returns:
        --------
        sitk.Image
            Binary image with liver region removed.
        """
        lm = relabel(lm, {1: 0})
        lm = to_binary(lm)
        return lm

    def process(self, lm_all_fn: Union[str, Path], lm_det_fn: Union[str, Path]) -> None:
        """
        Processes two label images to extract core and shell regions, and creates markup representations.

        Parameters:
        -----------
        lm_all_fn : Union[str, Path]
            File path to the 'all' label image.
        lm_det_fn : Union[str, Path]
            File path to the 'detected' label image.
        """
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

    def create_json_fn(self, lm_fn: Union[str, Path], suffix: str) -> Path:
        """
        Creates a JSON filename based on the input filename and suffix.

        Parameters:
        -----------
        lm_fn : Union[str, Path]
            The input filename.
        suffix : str
            The suffix to add to the filename.

        Returns:
        --------
        Path
            Full path to the output JSON file.
        """
        lm_fn_name = strip_extension(lm_fn.name)
        lm_fn_name = "_".join([lm_fn_name, suffix])
        lm_fn_name = lm_fn_name + ".json"
        fn_out = self.outfldr / lm_fn_name
        return fn_out

    def process_lm(
        self, lm: sitk.Image
    ) -> Tuple[sitk.Image, sitk.Image, Tuple[int, int]]:
        """
        Processes a label image to remove liver and split into core and shell segments.

        Parameters:
        -----------
        lm : sitk.Image
            The input label image.

        Returns:
        --------
        lm_core : sitk.Image
            Core region image.
        lm_shell : sitk.Image
            Shell region image.
        counts : tuple
            A tuple containing counts of core and shell regions.
        """
        lm = self.remove_liver(lm)
        lm_core, lm_shell = self.split_lm(lm)
        counts = self.get_core_shell_counts(lm_core, lm_shell)
        return lm_core, lm_shell, counts

    def split_lm(self, lm: sitk.Image) -> Tuple[sitk.Image, sitk.Image]:
        """
        Splits the label image into core and shell parts.

        Parameters:
        -----------
        lm : sitk.Image
            The input label image.

        Returns:
        --------
        lm_core : sitk.Image
            Core region image.
        lm_shell : sitk.Image
            Shell region image.
        """
        lm_core = sitk.Mask(lm, self.shell, outsideValue=0, maskingValue=3)
        lm_shell = sitk.Mask(lm, self.core, outsideValue=0, maskingValue=2)
        return lm_core, lm_shell

    def get_core_shell_counts(
        self, lm_core: sitk.Image, lm_shell: sitk.Image
    ) -> Tuple[int, int]:
        """
        Gets the count of core and shell regions.

        Parameters:
        -----------
        lm_core : sitk.Image
            Core region image.
        lm_shell : sitk.Image
            Shell region image.

        Returns:
        --------
        tuple
            A tuple containing counts of core and shell regions.
        """
        LC = LabelMapGeometry(lm_core)
        count_core = len(LC)

        LS = LabelMapGeometry(lm_shell)
        count_shell = len(LS)
        return count_core, count_shell

    def create_markups(
        self, lm: sitk.Image, color: str, fn_suffix: str
    ) -> Dict[str, Any]:
        """
        Creates markup representations for a label region and saves it to a JSON file.

        Parameters:
        -----------
        lm : sitk.Image
            The input label image.
        color : str
            Color of the markup glyphs.
        fn_suffix : str
            Suffix to add to the JSON filename.

        Returns:
        --------
        dict
            A dictionary containing markup data.
        """
        M = MarkupFromLabelmap([], 0, "auto", color)
        a = M.process(lm)
        if self.markup_shape is not None:
            a["markups"][0]["display"]["glyphType"] = self.markup_shape
        fn = self.create_json_fn(self.case_filename, fn_suffix)
        save_json(a, fn)
        return a


def merge_markups(fns: List[Union[str, Path]]) -> Dict[str, Any]:
    """
    Merges multiple markup JSON files into a single dictionary representation.

    Parameters:
    -----------
    fns : list of Union[str, Path]
        List of file paths to the JSON markup files.

    Returns:
    --------
    dict
        A dictionary containing merged markup data.
    """
    mups_full = load_json(fns[0])
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


def compile_tfmd_files(
    fns: List[Union[str, Path]],
    outfldr: Union[str, Path],
    outspacing: Tuple[float, float, float],
) -> None:
    """
    Compiles and processes transformation files, excluding specific categories, and writes the resulting images into the output folder.

    Parameters:
    -----------
    fns : list of Union[str, Path]
        List of file paths to the transformation files.
    outfldr : Union[str, Path]
        Output directory where the processed files will be saved.
    outspacing : tuple
        Desired spacing for the output images.
    """
    excludes = ["lesions", "liver", "merged", "react"]
    fns = [fn for fn in fns if all(exclude not in fn.name for exclude in excludes)]
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
    fn_lesions = str(outfldr / "lesions.nii.gz")
    print("Writing: ", fn_lesions)
    sitk.WriteImage(lms_le, fn_lesions)

    lms_li = sitk.GetImageFromArray(lms_li_ar)
    lms_li.SetSpacing(outspacing)
    fn_liver = str(outfldr / "liver.nii.gz")
    print("Writing: ", fn_liver)
    sitk.WriteImage(lms_li, fn_liver)

    lms_merged_ar = lms_li_ar + lms_le_ar
    lms_merged = sitk.GetImageFromArray(lms_merged_ar)
    lms_merged.SetSpacing(outspacing)
    fn_merged = str(outfldr / "merged.nii.gz")
    print("Writing: ", fn_merged)
    sitk.WriteImage(lms_merged, fn_merged)


# %%
if __name__ == "__main__":
# SECTION:-------------------- SETUP : Use the other file Onion which is more recent------------------------------------------------------------------------------------- <CR> <CR> <CR>

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
# SECTION:-------------------- Add liver  -------------------------------------------------------------------------------------- <CR> <CR> <CR>

    add_liver(ms_fldr, preds_fldr, msl_fldr)
    add_liver(as_fldr, preds_fldr, asl_fldr, True)
    add_liver(ds_fldr, preds_fldr, dsl_fldr, True)

# %%
# SECTION:-------------------- CROP CENTER AND RESAMPLE- --------------------- <CR> <CR> <CR>

    crop_center_resample(asl_fldr, aslc_fldr, outspacing, outshape)
    crop_center_resample(dsl_fldr, dslc_fldr, outspacing, outshape)
    crop_center_resample(msl_fldr, mslc_fldr, outspacing, outshape)

# %%
# SECTION:-------------------- Apply tfms iteratively (5 tfms)--------------------------------------------------------------------------------------' <CR> <CR> <CR>

    apply_tfms_all(aslc_fldr, output_folder_prefix="lms_all")
    apply_tfms_all(dslc_fldr, output_folder_prefix="lms_ds")
    apply_tfms_all(mslc_fldr, output_folder_prefix="lms_ms")
    # apply_tfm_folder(tfm_fn,mslc_lms_fldr,out_f_ms,slc)

    # compile_tfmd_files(out_f_ms)
# %%
# %%
# SECTION:--------------------Super merge ALL merged files (1 merged file per tfm) -------------------------- <CR> <CR> <CR>

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
# SECTION:--------------------CREATE CORE VERSUS SHELL MARKUPS ----------------------------------------- <CR> <CR> <CR>

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
# SECTION:-------------------- Batch scoring dsc-------------------------------------------------------------------------------------- <CR> <CR> <CR>
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
