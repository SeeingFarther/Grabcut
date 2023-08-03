import cv2
import numpy as np
import scipy.sparse
from scipy.sparse.linalg import spsolve
import argparse
import time
import numpy as np


# Calculates the boundaries of the rectangle that traps the mask.
def calc_mask_rect(mask):
    h, w = np.shape(mask)[0:2]
    b_right = b_down = 0
    b_left = w
    b_up = h

    # Find rectangle boundaries
    zeros_mask = mask
    zeros_mask = np.where(zeros_mask != 255, 0, zeros_mask)
    where = np.array(np.where(zeros_mask))
    b_up_tmp, b_left_tmp = np.amin(where, axis=1)
    b_down_tmp, b_right_tmp = np.amax(where, axis=1)

    # Change boundaries
    b_up = b_up_tmp if b_up_tmp < b_up else b_up
    b_down = b_down_tmp if b_down_tmp > b_down else b_down
    b_left = b_left_tmp if b_left_tmp < b_left else b_left
    b_right = b_right_tmp if b_right_tmp > b_right else b_right

    return b_left, b_right, b_up, b_down


# Pads the cut mask and source images to the dimensions of the target image.
def pad_mask_src(img, h_tgt, w_tgt, im_src):
    h, w = np.shape(img)
    top = bottom = (h_tgt // 2 - h // 2)
    left = right = (w_tgt // 2 - w // 2)
    p_mask = cv2.copyMakeBorder(img, top, bottom + 1, left, right + 1, cv2.BORDER_CONSTANT, value=0)
    p_src = cv2.copyMakeBorder(im_src, top, bottom + 1, left, right + 1, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return p_mask, p_src


# Numerates white mask pixels and saves at indx_mask.
def numerate_pix(p_mask):
    h, w = np.shape(p_mask)[0:2]
    indx_mask = np.full((h, w), -1)

    # Count pixels
    N = np.count_nonzero(p_mask == 255)

    # Numerates white mask pixels
    inreasing_nums = np.arange(0, N)
    indx_mask[p_mask == 255] = inreasing_nums
    indx_mask = indx_mask.astype('i')

    return N, indx_mask


def poisson_blend(im_src, im_tgt, im_mask, center):
    # Part 1: Boundaries, padding and prepping.
    # Calculating tight rectangle around mask.
    b_left, b_right, b_up, b_down = calc_mask_rect(im_mask)

    # Increasing boundaries for masks with white at the frame.
    sub_width = min(np.shape(im_mask)[1] - b_right, b_left, 10)
    sub_height = min(np.shape(im_mask)[0] - b_down, b_up, 10)
    b_left -= sub_width
    b_up -= sub_height
    b_right += sub_width
    b_down += sub_height

    # Checking if mask is smaller than target...exiting if not
    h_tgt, w_tgt = np.shape(im_tgt)[0:2]
    mask_cut = np.zeros((b_down - b_up, b_right - b_left), dtype=np.uint8)
    src_cut = np.zeros((b_down - b_up, b_right - b_left, 3), dtype=np.uint8)
    h_mask, w_mask = np.shape(mask_cut)

    if h_mask > h_tgt or w_mask > w_tgt:
        print("Mask is bigger than target image, unsupported.")
        exit()

    # Creating a copy of the defining rectangle of im_mask, im_src:
    for i in range(h_mask):
        for j in range(w_mask):
            mask_cut[i][j] = im_mask[i + b_up, j + b_left]
            src_cut[i][j] = im_src[i + b_up, j + b_left]

    # Pad mask and source.
    p_mask, p_src = pad_mask_src(mask_cut, h_tgt, w_tgt, src_cut)
    binary_p_mask = np.zeros((p_mask.shape[0], p_mask.shape[1]))
    binary_p_mask[p_mask == 255] = 1

    # Calculating the boundaries of the mask and number of pixels.
    N, indx_mask = numerate_pix(p_mask)

    # Part 2: Filling matrix A and vector b.
    A = scipy.sparse.lil_matrix((N, N))
    b = np.zeros((N, 3))
    n = -1

    for i in range(h_tgt):
        for j in range(w_tgt):
            if p_mask[i][j] == 255:
                bellow = above = right = left = np.zeros(3)
                n += 1

                # Filling matrix A:
                A[n, n] = -4.0

                if j > 0:
                    A[n, indx_mask[i][j - 1]] += binary_p_mask[i][j - 1]
                    left = p_src[i][j - 1] - (1 - binary_p_mask[i][j - 1]) * im_tgt[i][j - 1]

                if j < w_tgt - 1:
                    A[n, indx_mask[i][j + 1]] += binary_p_mask[i][j + 1]
                    right = p_src[i][j + 1] - (1 - binary_p_mask[i][j + 1]) * im_tgt[i][j + 1]

                if i < h_tgt - 1:
                    A[n, indx_mask[i + 1][j]] += binary_p_mask[i + 1][j]
                    bellow = p_src[i + 1][j] - (1 - binary_p_mask[i + 1][j]) * im_tgt[i + 1][j]

                if i > 0:
                    A[n, indx_mask[i - 1][j]] += binary_p_mask[i - 1][j]
                    above = p_src[i - 1][j] - (1 - binary_p_mask[i - 1][j]) * im_tgt[i - 1][j]

                # Filling vector b:
                b[n] = -4 * p_src[i][j] + above + bellow + right + left

    # Part 3: Solving Poisson equation.
    A = scipy.sparse.csr_matrix(A)
    res = spsolve(A, b)

    # Part 4: Creating the final, blended image.
    indx_mask = indx_mask[:im_tgt.shape[0], :im_tgt.shape[1]]
    white_pix = indx_mask != -1

    # Handling overflow issues.
    res[res > 255] = 255
    res[res < 0] = 0

    # Blending the object into the target image.
    im_tgt[white_pix] = res[indx_mask[white_pix]]
    im_blend = im_tgt
    return im_blend


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_path', type=str, default='./data/imgs/banana1.jpg', help='image file path')
    parser.add_argument('--mask_path', type=str, default='./data/seg_GT/banana1.bmp', help='mask file path')
    parser.add_argument('--tgt_path', type=str, default='./data/bg/table.jpg', help='mask file path')
    return parser.parse_args()


if __name__ == "__main__":
    # Load the source and target images
    args = parse()

    im_tgt = cv2.imread(args.tgt_path, cv2.IMREAD_COLOR)
    im_src = cv2.imread(args.src_path, cv2.IMREAD_COLOR)
    if args.mask_path == '':
        im_mask = np.full(im_src.shape, 255, dtype=np.uint8)
    else:
        im_mask = cv2.imread(args.mask_path, cv2.IMREAD_GRAYSCALE)
        im_mask = cv2.threshold(im_mask, 0, 255, cv2.THRESH_BINARY)[1]

    center = (int(im_tgt.shape[1] / 2), int(im_tgt.shape[0] / 2))
    start_time = time.time()
    im_clone = poisson_blend(im_src, im_tgt, im_mask, center)
    print("Total runtime: ", time.time() - start_time)

    # Save image
    cv2.imwrite('./Blended_image.jpg', im_clone)
    cv2.imshow('Blended image', im_clone)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
