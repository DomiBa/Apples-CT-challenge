import imageio
import glob
import numpy as np
import SimpleITK as sitk
from scipy import ndimage, misc
import os

# Filepath to the challenge sinograms
sino_dir = r"E:\Apples-CT\projections_noisefree"

# Output path for the .nrrd files. These files will be used for the input of our network
out_dir = sino_dir + '_nrrd'

# Create out_dir if it does not exist yet
try:
    os.mkdir(out_dir)
except:
    pass


# Adapt the numbers in range(31101, 32207) to the corresponding filenumbers of the challenge data.
# The numbers given here covered all training sinograms (yes, the implementation is not very efficient/sophisticated ;))
for filenumber in range(31101, 32207):

    try:
        sino_volume = []
        for sino_path in glob.glob(sino_dir + "/*" +str(filenumber) + "*.tif"):
            sino_slice = imageio.imread(sino_path)

            # We rotate, flip and resample the sinograms
            sino_slice = np.rot90(np.array(sino_slice))
            sino_slice = np.flip(sino_slice, 1)
            sino_slice = ndimage.zoom(sino_slice, [1376/1377, 1])

            sino_volume.append(sino_slice)

        # Save all sinograms from a single apple as a 3D .nrrd file
        sino_volume = np.array(sino_volume)
        sino_out = sitk.GetImageFromArray(sino_volume)
        sitk.WriteImage(sino_out, os.path.join(out_dir, str(filenumber) + '.nrrd'))

    except:
        pass