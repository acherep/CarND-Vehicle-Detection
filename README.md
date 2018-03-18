
**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/example_car_with_hog.jpg
[image2]: ./output_images/example_notcar_with_hog.jpg
[image4]: ./output_images/1_image.png
[image5]: ./output_images/2_heat.png
[image6]: ./output_images/3_heat_threshold.png
[image7]: ./output_images/4_labels.png
[image8]: ./output_images/5_image_boxes.png

[video1]: ./project_video.mp4

---

# HOG features

## 1. HOG features extraction from the training images.

I started by reading in all the `vehicle` and `non-vehicle` images in the first code cell.

I extracted the HOG features from the training images in the second of the IPython notebook `vehicle_detection`.

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`). I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like. In the third and fourth code cells, I created examples with the HOG features for the `vehicle` image:

![alt text][image1]

and the `non-vehicle` image:

![alt text][image2]

## 2. Final choice of HOG parameters.

I tried various combinations of parameters and found out that the HOG features provide good car/non-car detection and it is quite fast. Playing with parameters I found the following best combination:

| Paramenter        | Value    | 
|:-----------------:|:--------:| 
| color_space       | 'YUV'    | 
| orientations      | 11       |
| pix_per_cell      | (16, 16) |
| cell_per_block    | (2, 2)   |

## 3. Training a classifier using the selected HOG features.

I trained a linear SVM on the prepared dataset (see lines from 147 to 172 of the second cell). The preparation included scaling all features such that they have the same weight before fitting data into the linear SVM. Then the dataset was splited on the training and test sets with 80% and 20% of the overall dataset respectively.

# Sliding Window Search

## 1. Decision on what scales to search and how much to overlap windows.

I decided to apply the sliding window search with small windows to detect vehicles which are far away from the camera and with large windows to detect vehicles which are near by.

## 2. Demonstrating working pipeline with test images and  optimizing the performance of your classifier.

I defined a single function that extracts features using hog sub-sampling and makes predictions. The HOG features are extracted only once for the whole image which makes the vehicle detection faster.

Ultimately I searched on three scales 1.5, 1.75, 2 (corresponds to 96x96, 112x112, 128x128 window sizes respectively) using `YUV` 3-channel HOG features in the feature vector. I used the same HOG parameters as for the training the linear SVM discussed above. As an example I took the following image:

![alt text][image4]

The corresponding heatmap looks like the following:

![alt text][image5]

I defined a threshold such that the heat with low values is excluded. This approach allowed me to reject false positives. The output is shown below:

![alt text][image6]

Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:

![alt text][image7]

The resulting bounding boxes are drawn onto the last frame in the series:

![alt text][image8]


# Video Implementation. Filter false positives and combining overlapping bounding boxes.

Here's a [link to my video result](./project_video_output.mp4)

I recorded the positions of positive detections in each frame of the video. From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions. I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap. I then assumed each blob corresponded to a vehicle. I constructed bounding boxes to cover the area of each blob detected. The corresponding implementation is in cell 5, in particular, function `process_image()`.

To avoid false positives and to make the boxes in the video stream more stable, I averaged the heatmaps over last 15 frames and applied a threshold `40/15`. If the averaged heatmap value for a pixel is below this threshold then this pixel is detected to be a part of a vehicle less than 40 times in the last 15 frames. This pixel has to be rejected.

# Discussion

My pipeline may fail when there are many vehichles on the road. They will be detected within the single object. This problem can be solved by finetuning the scaling of the sliding windows and applying the thresholds.

The pipeline might also fail for different levels of brightness on the road, and for vehichles which come from the opposite dirrection.