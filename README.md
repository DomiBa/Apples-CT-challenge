# Apples-CT-challenge
Submission for the Apples-CT challenge

# Requirements:
- NVIDIA GPU with sufficient GPU memory (we used a Titan XP with 12GB memory, 8GB will most likely not suffice)
- Python 3.5
- Tensorflow 1.12
- CUDA v9.0

# Description:
In order to run the code, keras models must be downloaded. Since the files are too large to upload on Github, we uploaded them on Google Drive.

To reconstruct the data, 2 scripts need to be run. The first script converts the .tif sinograms into a 3D .nrrd file, which contains all sinograms of a single apple.
The .nrrd files will be saved in a separate directory. The .nrrd files are the input for the second script, which reconstructs the images using a trained tensorflow network.

We have uploaded 2 networks, one for 50-view sinograms and one for 25-view sinograms.

# How-To:
1. Download the trained networks from:
https://drive.google.com/file/d/1ZwW3AbdqZOAHrJm9gKBDgxIPUEBzZpD5/view?usp=sharing

2. Create .nrrd sinogram files with convert_apple.py:
- Set "sino_folder" in line 9 to the path containing the challenge .tif sinograms.
- In line 23 adapt range(31101, 32207) according to the challenge filenames.

3. Reconstruct the data with reconstruct_apple.py:
- Set "file_path" in line 79 to the directory containing the .nrrd sinograms (out_dir from convert_apple.py)
- Set "model_path" to in line 82 to the 50-view or 25-view model folder
- Set "sparse" in line 83 to "False" for 50 views and "True" for 25 views

If you have any questions feel free to contact us via Email (Dominik.Bauer@medma.uni-heidelberg.de) or via the CT Codesprint Slack channel (Dominik Bauer).
