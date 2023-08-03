import numpy as np
import cv2
import argparse
from graph import Graph
from gmm import GMM
from utillies import *
import time

GC_BGD = 0  # Hard bg pixel
GC_FGD = 1  # Hard fg pixel, will not be used
GC_PR_BGD = 2  # Soft bg pixel
GC_PR_FGD = 3  # Soft fg pixel
global graph
global previous_energy


# Define the GrabCut algorithm function
def grabcut(img, rect, n_iter=5):
    global graph
    global previous_energy

    # Assign initial labels to the pixels based on the bounding box
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    mask.fill(GC_BGD)
    x, y, w, h = rect
    w -= x
    h -= y

    # Initialize the inner square to Foreground
    mask[y:y + h, x:x + w] = GC_PR_FGD
    mask[rect[1] + rect[3] // 2, rect[0] + rect[2] // 2] = GC_FGD

    # Build graph
    img = img.astype(np.float64)
    img_h = img.shape[0]
    img_w = img.shape[1]
    graph = Graph(img_h * img_w, 0, img_h * img_w + 2, img_h, img_w)

    # Standard neighbors expectation of image sample
    expectation = compute_sample_expectation(img)

    # Build N-edges, both exception and N-edges function take 0.0037 seconds benchmark together
    calculate_n_edges(graph, img, expectation)

    # build hard T-edges
    vertices_map = graph.get_vertices_map()
    fg = vertices_map[mask == GC_FGD].flatten()
    bg = vertices_map[mask == GC_BGD].flatten()
    build_hard_t_edges(graph, bg, fg)

    # Init GMMs
    bgGMM, fgGMM = initalize_GMMs(img, mask)

    num_iters = 1000
    previous_energy = -1
    for i in range(num_iters):

        # Update GMM
        bgGMM, fgGMM = update_GMMs(img, mask, bgGMM, fgGMM)

        mincut_sets, energy = calculate_mincut(img, mask, bgGMM, fgGMM)

        mask = update_mask(mincut_sets, mask)

        if check_convergence(energy):
            break
        previous_energy = energy

    # Return the final mask and the GMMs
    mask[np.logical_and(mask != GC_BGD, mask != GC_PR_BGD)] = GC_FGD
    mask[np.logical_or(mask == GC_BGD, mask == GC_PR_BGD)] = GC_BGD
    return mask, bgGMM, fgGMM


def initalize_GMMs(img, mask, n_components=5):
    # Create bg GMM and initialize it with k-means
    bgGMM = GMM(img[np.logical_or(mask == GC_BGD, mask == GC_PR_BGD)], n_components)

    # Create fg GMM and initialize it with k-means
    fgGMM = GMM(img[np.logical_and(mask != GC_BGD, mask != GC_PR_BGD)], n_components)

    return bgGMM, fgGMM


# Define helper functions for the GrabCut algorithm
def update_GMMs(img, mask, bgGMM, fgGMM):
    # Get indexes relevant to each GMM
    bg = img[np.logical_or(mask == GC_BGD, mask == GC_PR_BGD)]
    fg = img[np.logical_and(mask != GC_BGD, mask != GC_PR_BGD)]

    # Classify each fg pixel to a component in the fg GMM and update the GMM
    fgGMM.update(fg)

    # Classify each bg pixel to a component in the bg GMM and update the GMM
    bgGMM.update(bg)

    return bgGMM, fgGMM


def calculate_mincut(img, mask, bgGMM, fgGMM):
    vertices_map = graph.get_vertices_map()
    t_edges = []
    capacities = []
    s = graph.get_source()
    t = graph.get_target()

    # Build T-edges
    # Build soft foreground T-edges
    soft_fg = vertices_map[mask == GC_PR_FGD].flatten()
    d_bg = bgGMM.compute_d_function(img[mask == GC_PR_FGD])  # Compute d background function
    d_fg = fgGMM.compute_d_function(img[mask == GC_PR_FGD])  # Compute d foreground function
    t_edges, capacities = build_soft_t_edges(s, t, soft_fg, d_bg, d_fg, t_edges, capacities)

    # Build soft background T-edges
    soft_bg = vertices_map[mask == GC_PR_BGD].flatten()
    d_bg = bgGMM.compute_d_function(img[mask == GC_PR_BGD])  # Compute d background function
    d_fg = fgGMM.compute_d_function(img[mask == GC_PR_BGD])  # Compute d foreground function
    t_edges, capacities = build_soft_t_edges(s, t, soft_bg, d_bg, d_fg, t_edges, capacities)

    # Add T-edges to the graph
    graph.set_soft_t_edges_cap(capacities)
    graph.set_soft_t_edges(t_edges)

    # Calculate min cut
    min_cut = graph.mincut()

    # Get min cut value(energy)
    energy = min_cut.value
    return min_cut, energy


def update_mask(mincut_sets, mask):
    S, T = mincut_sets.partition
    vertices_map = graph.get_vertices_map()
    new_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)

    # Map all soft foreground pixels in source to stay the same while all soft foreground pixels
    # in target become soft background
    soft_fg = np.where(mask == GC_PR_FGD)
    s_indicator = np.isin(vertices_map[soft_fg], S)
    new_mask[soft_fg] = np.where(s_indicator, GC_PR_FGD, GC_PR_BGD)

    # Map all soft background pixels in target to stay the same while all soft background pixels
    # in target become hard background
    soft_bg = np.where(mask == GC_PR_BGD)
    t_indicator = np.isin(vertices_map[soft_bg], T)
    new_mask[soft_bg] = np.where(t_indicator, GC_PR_BGD, GC_PR_FGD)

    # Map all foreground pixels
    new_mask[mask == GC_FGD] = GC_FGD

    # Map all background pixels
    new_mask[mask == GC_BGD] = GC_BGD

    mask = new_mask
    return mask


def check_convergence(energy):
    convergence = False
    if previous_energy >= 0 and (previous_energy - energy < 2500 or previous_energy - energy < previous_energy * 0.005):
        print(f"Convergence time={time.time() - start_time} seconds")
        convergence = True
    return convergence


def cal_metric(predicted_mask, gt_mask):
    # Calculate accuracy
    h, w = predicted_mask.shape
    acc = np.sum((predicted_mask == gt_mask)) / (w * h)
    acc *= 100

    # Calculate Jaccard similarity
    bool_pred = np.asarray(predicted_mask).astype(bool)
    bool_gt = np.asarray(gt_mask).astype(bool)
    union = np.logical_or(bool_gt, bool_pred)
    intersection = np.logical_and(bool_gt, bool_pred)
    jaccard = np.sum(intersection) / float(np.sum(union))

    return acc, jaccard


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_name', type=str, default='banana1', help='name of image from the course files')
    parser.add_argument('--eval', type=int, default=1, help='calculate the metrics')
    parser.add_argument('--input_img_path', type=str, default='', help='if you wish to use your own img_path')
    parser.add_argument('--use_file_rect', type=int, default=1, help='Read rect from course files')
    parser.add_argument('--rect', type=str, default='1,1,100,100', help='if you wish change the rect (x,y,w,h')
    return parser.parse_args()


if __name__ == '__main__':
    start_time = time.time()
    # Load an example image and define a bounding box around the object of interest
    args = parse()

    if args.input_img_path == '':
        input_path = f'data/imgs/{args.input_name}.jpg'
    else:
        input_path = args.input_img_path

    if args.use_file_rect:
        rect = tuple(map(int, open(f"data/bboxes/{args.input_name}.txt", "r").read().split(' ')))
    else:
        rect = tuple(map(int, args.rect.split(',')))

    img = cv2.imread(input_path)
    # Run the GrabCut algorithm on the image and bounding box
    mask, bgGMM, fgGMM = grabcut(img, rect)
    mask = cv2.threshold(mask, 0, 1, cv2.THRESH_BINARY)[1]

    # Print metrics only if requested (valid only for course files)
    if args.eval:
        gt_mask = cv2.imread(f'data/seg_GT/{args.input_name}.bmp', cv2.IMREAD_GRAYSCALE)
        gt_mask = cv2.threshold(gt_mask, 0, 1, cv2.THRESH_BINARY)[1]
        acc, jac = cal_metric(mask, gt_mask)
        print(f'Accuracy={acc}, Jaccard={jac}')

    # Apply the final mask to the input image and display the results
    img_cut = img * (mask[:, :, np.newaxis])
    cv2.imshow('Original Image', img)
    cv2.imshow('GrabCut Mask', 255 * mask)
    cv2.imshow('GrabCut Result', img_cut)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
