# Grabcut and Possion Blending

This repository contains code for a Grabcut and Possion Blending methods.

## Installation
You will need to install the following libraries:
- Argparse: pip install argparse
- Igraph:  pip install igraph
- Numpy: pip install numpy
- Scipy: pip install scipy
- OpenCV: pip install opencv-python

## Usage

### Grabcut
To run the main script with different hyperparameter values, use the `python grabcut.py` command along with the appropriate flags.

#### Hyperparameters:


##### Existing Source image
To choose the existing source image and her mask, use the `--input_name` flag. For example, to take the "banana1" image:

python grabcut.py --input_name banana1


##### Metrics
To calculate or not calculate the metrics, use the `--eval` flag with (0-Not calculating, 1-Calculating). For example:

python grabcut.py --eval 0


##### Custom source image
If you wish to use your own image, use the `-input_img_path` flag. For example:

python grabcut.py <input_img_path.jpg>


##### Custom mask
If you wish to use your own mask, use the `--rect` flag togther with `--use_file_rect`. For example:

python grabcut.py --use_file_rect 0 --rect 1,1,100,100


### Possion Blending
To run the main script with different hyperparameter values, use the `python possion_blending.py --src_path <source_image_path> --mask_path <mask_path> --tgt_path <target_image_path>` command along with the appropriate paths.
