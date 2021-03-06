import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
#import math

#from skimage.feature import hog
#from skimage import color, exposure
# images are divided up into vehicles and non-vehicles
#%matplotlib inline

cars = glob.glob('./data/vehicles/*/*.png', recursive=True)
notcars = glob.glob('./data/non-vehicles/*/*.png', recursive=True)

print(len(cars))
print(len(notcars))
        
# Define a function to return some characteristics of the dataset 
def data_look(car_list, notcar_list):
    data_dict = {}
    # Define a key in data_dict "n_cars" and store the number of car images
    data_dict["n_cars"] = len(car_list)
    # Define a key "n_notcars" and store the number of notcar images
    data_dict["n_notcars"] = len(notcar_list)
    # Read in a test image, either car or notcar
    example_img = mpimg.imread(car_list[0])
    # Define a key "image_shape" and store the test image shape 3-tuple
    data_dict["image_shape"] = example_img.shape
    # Define a key "data_type" and store the data type of the test image.
    data_dict["data_type"] = example_img.dtype
    # Return data_dict
    return data_dict
    
data_info = data_look(cars, notcars)

print('Your function returned a count of', 
      data_info["n_cars"], 'cars and', 
      data_info["n_notcars"], 'non-cars')
print('of size: ',data_info["image_shape"], ' and data type:', 
      data_info["data_type"])
# Just for fun choose random car / not-car indices and plot example images   
car_ind = np.random.randint(0, len(cars))
notcar_ind = np.random.randint(0, len(notcars))
    
# Read in car / not-car images
car_image = mpimg.imread(cars[car_ind])
notcar_image = mpimg.imread(notcars[notcar_ind])

# Plot the examples
fig = plt.figure()
plt.subplot(121)
plt.imshow(car_image)
plt.title('Example Car Image')
plt.subplot(122)
plt.imshow(notcar_image)
plt.title('Example Not-car Image')

import time
from sklearn.svm import LinearSVC
# NOTE: the next import is only valid for scikit-learn version <= 0.17
# for scikit-learn >= 0.18 use:
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# from sklearn.cross_validation import train_test_split
from skimage.feature import hog
# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block, 
                        vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, 
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), 
                                  transform_sqrt=True, 
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:      
        features = hog(img, orientations=orient, 
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), 
                       transform_sqrt=True, 
                       visualise=vis, feature_vector=feature_vec)
        return features

# Define a function to compute binned color features  
def bin_spatial(img, size=(32, 32)):
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel() 
    # Return the feature vector
    return features

# Define a function to compute color histogram features 
# NEED TO CHANGE bins_range if reading .png files with mpimg!
def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

# Define a function to extract features from a single image window
# This function is very similar to extract_features()
# just for a single image rather than list of images
def single_img_features(img, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):    
    #1) Define an empty list to receive features
    img_features = []
    #2) Apply color conversion if other than 'RGB'
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else: feature_image = np.copy(img)
    #3) Compute spatial features if flag is set
    if spatial_feat == True:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        #4) Append features to list
        img_features.append(spatial_features)
    #5) Compute histogram features if flag is set
    if hist_feat == True:
        hist_features = color_hist(feature_image, nbins=hist_bins)
        #6) Append features to list
        img_features.append(hist_features)
    #7) Compute HOG features if flag is set
    if hog_feat == True:
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.extend(get_hog_features(feature_image[:,:,channel], 
                                    orient, pix_per_cell, cell_per_block, 
                                    vis=False, feature_vec=True))
            hog_features = np.ravel(hog_features)     
        else:
            hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                        pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        #8) Append features to list
        img_features.append(hog_features)

    #9) Return concatenated array of features
    return np.concatenate(img_features)

# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(imgs, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):
        # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        # Read in each one by one
        image = mpimg.imread(file)
        file_features = single_img_features(image, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)
        # remove NaNs
        if not(np.isnan(file_features).any()):
            features.append(file_features)
    # Return list of feature vectors
    return features
    
# Read in cars and notcars
#images = glob.glob('*.jpeg')
#cars = []
#notcars = []
#for image in images:
#    if 'image' in image or 'extra' in image:
#        notcars.append(image)
#    else:
#        cars.append(image)

# Reduce the sample size because
# The quiz evaluator times out after 13s of CPU time
sample_size = 6000
cars = cars[:sample_size]
notcars = notcars[:sample_size]

### TODO: Tweak these parameters and see how the results change.
color_space = 'YUV' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 11  # HOG orientations
pix_per_cell = 16 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
spatial_size = (16, 16) # Spatial binning dimensions
hist_bins = 16    # Number of histogram bins
spatial_feat = False # Spatial features on or off
hist_feat = False # Histogram features on or off
hog_feat = True # HOG features on or off
y_start_stop = [350, 650] # Min and max in y to search in slide_window()

car_features = extract_features(cars, color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat)
notcar_features = extract_features(notcars, color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat)
feature_index = 0
## remove NaNs
#while feature_index < len(car_features):
#    if np.isnan(car_features[feature_index]).any():
#        del car_features[feature_index]
#    else:
#        feature_index += 1
#
#feature_index = 0
#while feature_index < len(notcar_features):
#    if np.isnan(notcar_features[feature_index]).any():
#        del notcar_features[feature_index]
#    else:
#        feature_index += 1

print(len(car_features))
print(len(notcar_features))
        

X = np.vstack((car_features, notcar_features)).astype(np.float64)                        
# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
scaled_X = X_scaler.transform(X)

# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))


# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
    scaled_X, y, test_size=0.3, random_state=rand_state)

print('Using:',orient,'orientations',pix_per_cell,
    'pixels per cell and', cell_per_block,'cells per block')
print('Feature vector length:', len(X_train[0]))
# Use a linear SVC 
svc = LinearSVC()
# Check the training time for the SVC
t=time.time()
svc.fit(X_train, y_train)
t2 = time.time()
print(round(t2-t, 2), 'Seconds to train SVC...')
# Check the score of the SVC
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
# Check the prediction time for a single sample
t=time.time()

#%%
#
## Define a function that takes an image,
## start and stop positions in both x and y, 
## window size (x and y dimensions),  
## and overlap fraction (for both x and y)
#def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None], 
#                    xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
#    # If x and/or y start/stop positions not defined, set to image size
#    if x_start_stop[0] == None:
#        x_start_stop[0] = 0
#    if x_start_stop[1] == None:
#        x_start_stop[1] = img.shape[1]
#    if y_start_stop[0] == None:
#        y_start_stop[0] = 0
#    if y_start_stop[1] == None:
#        y_start_stop[1] = img.shape[0]
#    # Compute the span of the region to be searched    
#    xspan = x_start_stop[1] - x_start_stop[0]
#    yspan = y_start_stop[1] - y_start_stop[0]
#    # Compute the number of pixels per step in x/y
#    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
#    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
#    # Compute the number of windows in x/y
#    nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
#    ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
#    nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step) 
#    ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step) 
#    # Initialize a list to append window positions to
#    window_list = []
#    # Loop through finding x and y window positions
#    # Note: you could vectorize this step, but in practice
#    # you'll be considering windows one by one with your
#    # classifier, so looping makes sense
#    for ys in range(ny_windows):
#        for xs in range(nx_windows):
#            # Calculate window position
#            startx = xs*nx_pix_per_step + x_start_stop[0]
#            endx = startx + xy_window[0]
#            starty = ys*ny_pix_per_step + y_start_stop[0]
#            endy = starty + xy_window[1]
#            
#            # Append window position to list
#            window_list.append(((startx, starty), (endx, endy)))
#    # Return the list of windows
#    return window_list

## Define a function you will pass an image 
## and the list of windows to be searched (output of slide_windows())
#def search_windows(img, windows, clf, scaler, color_space='RGB', 
#                    spatial_size=(32, 32), hist_bins=32, 
#                    hist_range=(0, 256), orient=9, 
#                    pix_per_cell=8, cell_per_block=2, 
#                    hog_channel=0, spatial_feat=True, 
#                    hist_feat=True, hog_feat=True):
#
#    #1) Create an empty list to receive positive detection windows
#    on_windows = []
#    #2) Iterate over all windows in the list
#    for window in windows:
#        #3) Extract the test window from original image
#        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))      
#        #4) Extract features for that window using single_img_features()
#        features = single_img_features(test_img, color_space=color_space, 
#                            spatial_size=spatial_size, hist_bins=hist_bins, 
#                            orient=orient, pix_per_cell=pix_per_cell, 
#                            cell_per_block=cell_per_block, 
#                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
#                            hist_feat=hist_feat, hog_feat=hog_feat)
#        #5) Scale extracted features to be fed to classifier
#        test_features = scaler.transform(np.array(features).reshape(1, -1))
#        #6) Predict using your classifier
#        prediction = clf.predict(test_features)
#        #7) If positive (prediction == 1) then save the window
#        if prediction == 1:
#            on_windows.append(window)
#    #8) Return windows for positive detections
#    return on_windows

## Define a function to draw bounding boxes
#def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
#    # Make a copy of the image
#    imcopy = np.copy(img)
#    # Iterate through the bounding boxes
#    for bbox in bboxes:
#        # Draw a rectangle given bbox coordinates
#        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
#    # Return the image copy with boxes drawn
#    return imcopy

def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap# Iterate through list of bboxes
    
def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img


#image = mpimg.imread('test_images/test0.jpg')
#
#def process_image(image):
#    global imageCounter
#    plt.imsave("test_images_output/image.jpg", image, format="jpg")
#    image = mpimg.imread("test_images_output/image.jpg")
##    plt.imsave("test_images_output/" +
##               str(imageCounter) + "_0_original.jpg", image, format="jpg")
##    image = mpimg.imread("test_images_output/" +
##               str(imageCounter) + "_0_original.jpg")
#
#    draw_image = np.copy(image)
#    
#    # Uncomment the following line if you extracted training
#    # data from .png images (scaled 0 to 1 by mpimg) and the
#    # image you are searching is a .jpg (scaled 0 to 255)
#    image = image.astype(np.float32) / 255
#
#    heat = np.zeros_like(image[:,:,0]).astype(np.float)
#
#
#    
#    #windows = slide_window(image, x_start_stop=[0, 1280], y_start_stop=y_start_stop, 
#    #                    xy_window=(96, 96), xy_overlap=(0.3, 0.3))
#    hot_windows_all_sizes = []
#    for window_size in np.array([96, 128]):
#        windows = slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop, 
#                            xy_window=(window_size, window_size), xy_overlap=(0.7, 0.7))
#        
#        hot_windows = search_windows(image, windows, svc, X_scaler, color_space=color_space, 
#                                spatial_size=spatial_size, hist_bins=hist_bins, 
#                                orient=orient, pix_per_cell=pix_per_cell, 
#                                cell_per_block=cell_per_block, 
#                                hog_channel=hog_channel, spatial_feat=spatial_feat, 
#                                hist_feat=hist_feat, hog_feat=hog_feat)
#        hot_windows_all_sizes.extend(hot_windows)
#        
#    
#    window_img = draw_boxes(draw_image, hot_windows_all_sizes, color=(0, 0, 255), thick=6)
##    plt.imsave("test_images_output/" +
##           str(imageCounter) + "_1_boxes.jpg", window_img, format="jpg")
#    
#    # Add heat to each box in box list
#    heat = add_heat(heat, hot_windows_all_sizes)
#        
#    # Apply threshold to help remove false positives
#    heat = apply_threshold(heat, 3)
#    
#    # Visualize the heatmap when displaying    
#    heatmap = np.clip(heat, 0, 255)
##    plt.imsave("test_images_output/" +
##       str(imageCounter) + "_2_heatmap.jpg", heatmap, format="jpg")
#    
#    
#    from scipy.ndimage.measurements import label
#    # Find final boxes from heatmap using label function
#    labels = label(heatmap)
#    draw_img = draw_labeled_bboxes(draw_image, labels)
##    plt.imsave("test_images_output/" +
##       str(imageCounter) + "_3_labeled_bboxes.jpg", draw_img, format="jpg")
#    imageCounter = imageCounter + 1
#    return draw_img
#
#draw_img = process_image(image)
#
##fig = plt.figure()
##plt.subplot(131)
##plt.imshow(draw_img)
##plt.title('Car Positions')
##plt.subplot(132)
##plt.imshow(window_img)
##plt.title('Heat Map')
##plt.subplot(133)
##plt.imshow(heatmap, cmap='hot')
##plt.title('Heat Map')
##fig.tight_layout()                 
#
#plt.imshow(draw_img)

from scipy.ndimage.measurements import label

image = mpimg.imread('test_images/test6.jpg')

draw_image = np.copy(image)

# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block):
    
    img = img.astype(np.float32)/255
    
    img_tosearch = img[ystart:ystop,:,:]
    ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2YUV)
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
        
    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1 
#    nfeat_per_block = orient*cell_per_block**2
    
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step
    
    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
    
    on_windows = []
    
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            # Extract the image patch
#            subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
          
            # Get color features
#            spatial_features = bin_spatial(subimg, size=spatial_size)
#            hist_features = color_hist(subimg, nbins=hist_bins)

            # Scale features and make a prediction
            test_features = X_scaler.transform(hog_features.reshape(1, -1))    
            #test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))    
            test_prediction = svc.predict(test_features)
            
            if test_prediction == 1:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                on_windows.append(((xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart)))
                
    return on_windows
heatmap_history_size = 15
heatmap_array = np.zeros((image.shape[0], image.shape[1], heatmap_history_size)).astype(np.float)

def process_image(image):
    global heatmap_array
    plt.imsave("output_images/image.jpg", image, format="jpg")
    image = mpimg.imread("output_images/image.jpg")
    draw_image = np.copy(image)
    
    ystart = 400
    ystop = 670
    orient = 11  # HOG orientations
    pix_per_cell = 16 # HOG pixels per cell
    cell_per_block = 2 # HOG cells per block
    
    hot_windows_all_sizes = []
    for scale in np.array([1.5, 1.75, 2]):
        hot_windows = find_cars(image, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block)
        hot_windows_all_sizes.extend(hot_windows)
        
    heat = np.zeros_like(image[:,:,0]).astype(np.float)
    
    # Add heat to each box in box list
    heat = add_heat(heat, hot_windows_all_sizes)
        
    # Apply threshold to help remove false positives
    heat = apply_threshold(heat, 3)
    
    # Visualize the heatmap when displaying    
    heatmap = np.clip(heat, 0, 255)
    # rolling the heatmap array to remove the last heatmap
    # and replace it with the new one
    heatmap_array = np.roll(heatmap_array, 1, axis=2)
    heatmap_array[:,:,0] = heatmap
    heatmap = np.mean(heatmap_array, axis=2)
    #    plt.imsave("test_images_output/" +
    #       str(imageCounter) + "_2_heatmap.jpg", heatmap, format="jpg")
    
    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(draw_image, labels)
    return draw_img

draw_img = process_image(image)
plt.imshow(draw_img)

#%%
from moviepy.editor import VideoFileClip
#from IPython.display import HTML

imageCounter = 1
heatmap_history_size = 15
heatmap_array = np.zeros((image.shape[0], image.shape[1], heatmap_history_size)).astype(np.float)

file_name = "project_video"
#file_name = "test_video"

## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
#clip = VideoFileClip(file_name + ".mp4").subclip(25,42)
#clip = VideoFileClip(file_name + ".mp4").subclip(0,5)
clip = VideoFileClip(file_name + ".mp4")
white_clip = clip.fl_image(process_image)
white_clip.write_videofile(file_name + "_output.mp4", audio=False)

#imageCounter = 1
#
#def pro_image(image):
#    global imageCounter
#    plt.imsave("project_video_images/image_"+str(imageCounter)+".jpg", image, format="jpg")
#    imageCounter = imageCounter + 1
#    return image
#    
#clip = VideoFileClip("project_video.mp4")
#white_clip = clip.fl_image(pro_image)
#white_clip.write_videofile(file_name + "_output2.mp4", audio=False)
