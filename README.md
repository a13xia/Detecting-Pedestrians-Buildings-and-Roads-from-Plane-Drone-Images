Script Architecture

main.py: It has as input the train folder from the UAVID dataset (both the images and their masks), and it outputs a .txt file for each image, containing the labelling of the boundary boxes. In addition, it creates a folder debug_view, where each image processed also has boundary boxes around targeted objects.


Why multiple train/predict scripts? I tried two different approaches:

1. Tiling Workflow (train-tiling.py, predict-tiling.py): Implements image slicing to preserve spatial detail for small objects, but fails to detect buildings even with 25 epochs.

2. Standard Workflow (train.py, predict.py): Processes full-resolution 4K images, having way better results overall.



Project Structure

/dataset: Raw and labeled 4K images. (600 images)

/dataset_split_tiling: Pre-processed tiles for optimized training. (around 15.000 small images)

/runs: Training logs and model weights (best.pt).

/predictions: Visual inference results for comparison.

