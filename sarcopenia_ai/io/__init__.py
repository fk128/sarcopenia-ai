import os

import SimpleITK as sitk


def load_image(image_path):
    if image_path.endswith('.nii') or image_path.endswith('.nii.gz'):
        sitk_image = sitk.ReadImage(image_path)

    else:
        print("Reading Dicom directory:", image_path)
        reader = sitk.ImageSeriesReader()

        dicom_names = reader.GetGDCMSeriesFileNames(image_path)
        reader.SetFileNames(dicom_names)
        sitk_image = reader.Execute()

    image_name = os.path.basename(image_path).replace('.nii', '').replace('.gz', '')

    return sitk_image, image_name


def save_slice_as_dcm(image_sitk, slice_z, output_path, image_name):
    size = list(image_sitk.GetSize())
    size[2] = 0
    index = [0, 0, slice_z]
    Extractor = sitk.ExtractImageFilter()
    Extractor.SetSize(size)
    Extractor.SetIndex(index)
    sitk.WriteImage(Extractor.Execute(image_sitk),
                    os.path.join(output_path, '{}_{}_slice.dcm'.format(image_name, slice_z)))
