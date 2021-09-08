# Detecting League of Legends champions on the minimap using OpenCV

[Source code to my blog post](https://maknee.github.io/blog/2021/League-ML-Minimap-Detection1/)

![](repo_files/opencv_demo.gif)

![](repo_files/opencv_detection.gif)

# Setup

`pip install -r requirements.txt`

# League version

Assumes patch 10.7

# Running

`python test_ingame.py`

Assumptions: The game is running in 1920x1080 resolution. Modify `minimap_ratio` and `icon_ratio` in `test_ingame.py` to adapt to different resolution. 

This will output something like this in an external window:

![](repo_files/opencv_result.gif)

