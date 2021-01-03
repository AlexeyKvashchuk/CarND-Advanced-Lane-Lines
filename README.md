# **Advanced Lane Finding Project**


[image1]: ./output_images/calibration5_Undistorted.png
[image2]: ./output_images/calibration12_corners_output.jpg
[image3]: ./output_images/test6_combined_binary.jpg
[image4]: ./output_images/straight_lines1_src.jpg
[image5]: ./output_images/straight_lines1_warped.jpg
[image6]: ./output_images/Screen_Shot_output.png


## Project Steps/Pipeline

- Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
- Apply a distortion correction to raw images.
- Use color transforms, gradients, etc., to create a thresholded binary image.
- Apply a perspective transform to rectify binary image ("birds-eye view").
- Detect lane pixels and fit to find the lane boundary.
- Determine the curvature of the lane and vehicle position with respect to center.
- Warp the detected lane boundaries back onto the original image.
- Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.


### 1. Description of the pipeline. 

#### Camera Calibration 

The camera calibration process follows a standard procedure of processing various images of a chessboard by 

1. Creating a list of real-world object points that is common to all chessboard images (as stored in *objpoints* variable/list). 
2. Identifying chessboard corners (as stored in *imgpoints*  variable/list) by utilizing the OpenCV function *cv2.findChessboardCorners()*.
3. Calibrating the camera by *cv2.calibrateCamera(objpoints, imgpoints,...)* to obtain the camera matrix and distortion coefficients for further use. 

The above process is done once, so that the key outputs (camera matrix variable *mtx* and distortion coefficients *dist*) are saved by using numpy's *np.save('camera_matrix.npy', mtx)*.

Here is an example of an input/original chessboard image with identified corners, as produced by the OpenCV function *cv2.drawChessboardCorners()*:

![alt text][image2]

And here is an example of an undistored image as output by the OPenCV function *cv2.calibrateCamera(objpoints, imgpoints,...)*:

![alt text][image1]

#### Apply a distortion correction to raw images.

Having calculated camera matrix and distortion coefficients, the same *cv2.undistort(image, mtx, dist, None, mtx)* is then used to process each frame/image of the project video (*project_video.mp4*).



#### Use color transforms, gradients, etc., to create a thresholded binary image.

Unlike in the first project, instead of using Canny algorithm, a simpler thresholding method is used for edge detection. I used a combination of two thresholds: saturation (*s\_thresh=(170, 255)*) and Sobel conv transform in x-direction (*sx\_thresh=(20, 100)*), since the goal is to detect 'vertical' edges.

This is implemented in function *Color_Gradient_Combined_Edges(img, s\_thresh, sx\_thresh)*. 

Here is an example of a binary image, as produced by processing an undistored image from the previous step by using the function *Color_Gradient_Combined_Edges(undistorted_img, s\_thresh=(170, 255), sx\_thresh=(20, 100))*:

![alt text][image3]

#### Apply a perspective transform to rectify binary image ("birds-eye view").

To calculate a perspective transform I first pick a set of four source points (*src*) that form a trapezoidal shape 

<p style="text-align: center;">src = [[190, img.shape[0]], [520, 500], [770, 500], [1135, img.shape[0]]]</p>


followed by a four destination points forming a rectangle 

<p style="text-align: center;">dst = [[190, img.shape[0]], [190, 0], [1135, 0], [1135, img.shape[0]]].</p>

The warped/transformed image is then produced by the function *warper(img, src, dst)* which first calculates a transform matrix as 

<p style="text-align: center;">M = cv2.getPerspectiveTransform(src, dst)</p>

and then applies the tranform as 

<p style="text-align: center;">warped = cv2.warpPerspective(img, M, ...).</p>

Here is an example of the source/trapezoidal shape 

![alt text][image4]

And here is a warped ('bird's eye view') output

![alt text][image5]



#### Detect lane pixels and fit to find the lane boundary.

There are two functions/methods to detect a lane boundary. The first one is based on a 'windows' method, as described in the corresponding Udacity course lecture. This is implemented in *find\_lane\_pixels\_from\_windows(binary\_warped,...)*. This method is applied to the first *n\_from\_windows* frames/images to initialize the second method. 

Once the first *n\_from\_windows* quadratics are calculated, the second method starts searching for a boundary within a margin/neighborhood of an average of the previous *n\_from\_windows* lines. This is implemented in *find\_lane\_pixels\_from\_prior(binary\_warped, left\_fit, right\_fit)*, where *left\_fit, right\_fit* are averages of previous *n\_from\_windows* boundaries.

Finally, the frame's lines are calculated as smoothed/averaged over the recent *n\_prior\_poly\_fit* lines (including the current one). 

#### Determine the curvature of the lane and vehicle position with respect to center.

The real-world (meters) curvature is calculated by the function *measure\_curvature\_real(left\_fit, right\_fit, ploty)*. Conversion to a real-world/meters scale is done by re-scaling quadratic coefficients that are already calculated by *fit\_polynomial(...)* function from the previous step. 

The vehicle position/offset is calculated by function *offset\_real(image\_width, image\_height, left\_fit, right\_fit)*.

#### Warp the detected lane boundaries back onto the original image and output visual display of the lane boundaries.

This is implemented in function *project\_back\_to\_original(...)* by applying an inverse perspective transform *Minv*. This function also draws the lanes onto the unwarped/original image by using OpenCV's function *cv2.fillPoly(...)*. 

*Note*: This function/code is provided as a part of Udacity's project instructions. 

#### Output visual display of numerical estimation of lane curvature and vehicle position.

Finally, curvature and vehicle position/offset are displayed by using OpenCV's function cv2.putText(...). This is implemented in *add\_info\_offset\_curvature(output\_unlabled, curvature, offset)* function. 

An example output is provided below. 

![alt text][image6]

*Note*: The displayed curvature is an average curvature of left/right lane lines. 

### Pipeline (video)

The overall pipeline is implemented in function *process_image(image)* that process a video frame-by-frame. The output (i.e. the annotated project video *project_video.mp4*) is located in the */CarND-Advanced-Lane-Lines/output\_images/*  folder. 

### 2. Discussion

A list of potential shortcomings of the approach: 

- instability with respect to hyper-parameters
- high number of hyper-parameters
- manual search of optimal hyper-parameters values
- I have not implemented sanity checks suggested in project instructions. The method instead relies on smoothing of priors to search around, as well as smoothing of the final output. 
- Edge detection method is a simple combined thresholing approach that might not work well for a more general/challenging setting. A combination of Canny method and color-based thresholding approach might be beneficial. 


As for the methodology, I have also implemented and tested and approach that uses a weighted/exponential moving average as a way of smoothing, with the idea of assigning higher weight to more recent data. 
This resulted in a marginal improvement, at the cost of yet another hyper-parameter alpha (weight). The submitted code doesn't use this method. 
The code for exponential averaging/smoothing is implemented in *exp\_smoothed\_poly\_fit(history\_length, alpha)*. 
