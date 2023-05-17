# Shadow_detection

Introduction:

Shadow detection is one of the important problems in computer vision. Shadows are caused by the occlusion of light sources by objects in the scene and can significantly affect the appearance and quality of images. Accurate detection and removal of shadows can improve the performance of several computer vision applications. In this project, I have tried to address the problem of shadow detection in images using convolutional neural networks and apart from the CNN-based model, I have used another simple and effective approach, which is based on thresholding the luminance channel of the LAB color space and is described in the paper "Shadow Detection and Removal from a Single Image  Using LAB Color Space" by Saritha Murali and V. K. Govindan.

Data:

I have used the SBU Shadow dataset, which consists of 4085 images with varying lighting conditions and shadow types. The dataset includes images of indoor and outdoor scenes, with and without objects, and with various degrees of shadowing. The dataset consists of two main components: shadow images and shadow masks. The shadow images are RGB images that contain objects affected by shadows, while the shadow masks are binary masks or ground truth masks indicating the presence or absence of shadows in the corresponding images.

Dataset was downloaded from- https://www3.cs.stonybrook.edu/~cvl/projects/shadow_noisy_label/index.html

Approach:

For Machine learning  based model

Data Preprocessing:
The first step is to preprocess the input images and masks. I have resized the images and masks to a fixed size of 100x100 pixels to ensure consistency. 

Model Architecture:
The first layer in CNN is a 2D convolutional layer that performs convolution operations on the input images. This layer is followed by batch normalization and activation functions for non-linearity. Max pooling and average pooling layers are used to down-sample the feature maps and capture the most salient features. Dropout is applied to regularize the network and prevent overfitting. Finally, a sigmoid activation function is used to produce the predicted shadow masks.

Model Training:
The model is trained using the training set of images and their corresponding shadow masks. The loss function used is binary cross-entropy, and the Adam optimizer is used for gradient-based optimization. The training process involves iterating over the training images in mini-batches, computing the loss, and updating the network weights using backpropagation

For Non-machine learning based approach

The input images are converted to the LAB color space to separate the luminance (L) channel, which provides information about the lighting conditions. The L channel is then threshold to create a binary mask that indicates regions potentially affected by shadows. The threshold L channel is then used to create a binary mask that represents the shadows in the image. I have used morphological operations to fill in any gaps or holes. Finally, applied the binary mask to the input image to create a shadow-free version using the cv2.bitwise_and function.

Results:

The trained model is evaluated on the test set to assess its performance in shadow detection. The evaluation metrics used include loss and accuracy. 
Loss on test set- 0.35780036449432373
Accuracy on test set- 0.8132389187812805

Additionally, visual analysis is performed by comparing the predicted shadow masks of CNN based model, predicted shadow masks without using the model and the ground truth masks.

CNN based model is visualized using heat maps on the original image
Non machine learning based approach is visualized by converting the predicted shadow regions to white pixels and non-shadow regions to black pixels to be on par with the ground truths.

Both my model and the one without using the model have provided good results, but surprisingly the one without using the model has better results compared to that of the model.

Below are the screenshots of my results
 
 
 
 
 
 

 
 
 
 
 
 
 
![image](https://github.com/Hruday851998/Shadow_detection/assets/98664425/cc1ca124-5b0c-4355-be45-8be98adf87ba)



 
 
 

 
 
 
 
 
 
 
References:

Murali, Saritha and Govindan, V. K.. "Shadow Detection and Removal from a Single Image Using LAB Color Space" Cybernetics and Information Technologies, vol.13, no.1, 2013, pp.95-103. https://doi.org/10.2478/cait-2013-0009

@inproceedings{Vicente-etal-ECCV16, 
Author = {Tomas F. Yago Vicente and Le Hou and Chen-Ping Yu and Minh Hoai and Dimitris Samaras}, 
Booktitle = {Proceedings of European Conference on Computer Vision}, 
Title = {Large-scale Training of Shadow Detectors with Noisily-Annotated Shadow Examples}, 
Year = {2016}} 

@inproceedings{Vicente-etal-CVPR16,
Author = {Tomas F. Yago Vicente and Minh Hoai and Dimitris Samaras},
Booktitle = {Proceedings of IEEE Conference on Computer Vision and Pattern Recognition},
Title = {Noisy Label Recovery for Shadow Detection in Unfamiliar Domains},Year = {2016}}
