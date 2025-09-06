# DivineTree: All-in-One 3D Tree Modeling with Diverse and Fused Visual Guidance

DivineTree is a comprehensive framework for 3D tree modeling conditioned on **di**verse **vi**sual guida**n**c**e**, including point clouds, different image styles (e.g., photographs, comics, sketches), and more. This repository provides training data, pre-trained model weights, and Blender examples to facilitate research in 3D tree generation and rendering.

## Features

1. **Training Data**  
   We provide a dataset of 100,000 3D trees generated via procedural modeling.  
   - [tree10w.rar](https://pan.baidu.com/s/19PUGISqa-1lS5aQD7AzMwQ) (2.26GB, code: 1331)  
   Each tree is represented in the branch graph format as an Nx7 matrix. Each row consists of:
     - `(x, y, z, parent_node_ID, width, is_leaf, leaf_size)`  
   You can use the `datasets.py` script to read and visualize the data.

2. **Pretrained Model Weights**  
   We offer pretrained weights for a diffusion model trained on the 100,000 3D tree dataset.  
   - [llama-tree.pth](https://pan.baidu.com/s/1tEga1mMId7wYyIdFbuujng) (code: 6khz)  
   After downloading, place the file in the `params` folder.

3. **Blender Files**  
   We provide 3D tree samples generated using DivineTree and imported into Blender for lighting, rendering, and scene setup.  
   - The `blender_files` folder contains 3 `.blend` files that can be directly opened in Blender.  
   ![Blender.jpg](https://github.com/xujiabo/DivineTree/blob/main/assets/blender.jpg)

## Visualization

1. **Demo**  
   We offer a demo showcasing a 3D tree generated from a point cloud. You can run the `vis_pcd_to_tree_wo_net.py` script for visualization.  
   ![Demo.jpg](https://github.com/xujiabo/DivineTree/blob/main/assets/demo.jpg)

2. **Future Updates**  
   Our paper is currently under review. If fortunate enough to be accepted, we will provide more examples for researchers to generate 3D trees using various visual guidance (e.g., point clouds, images in different styles, etc.). Although the code for these processes is already prepared in `xxx_to_tree.py`, we will provide runnable examples in the future.
