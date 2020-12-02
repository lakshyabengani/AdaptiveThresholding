'''
Step by step execution : 

1. Noise reduction : 
    Done Using Gaussian Blur ( The Smaller the Kernel size , lesser is the blurring effect )

2. Gradient calculation : 
    Done by applying Sobels Filters

3. Non-maximum suppression to thin out edges :
    Done by performing non-max-supression using the gradient matrix and Egde Direction got from Gradient Calculation

4. Double threshold : 
    Here we identify 3 types of pixels 
    -> Strong Pixel : Surely Contributes to the edges , Identified by High Threshlold
    -> Weak Pixels : May Contribute to edges , Identified by the region between High and Low Threshold
    -> Non Relevant : Donot contribute to edges , Identified using Low Threshold Value

5. Edge Tracking by Hysteresis :
    Identify if the weak pixels can be edges or not by seeing if there is any strong edge in its neighbourhood

'''

import numpy as np
from PIL import Image

# generate the gaussian kernel 
def generate_gaussian_kernel(sigma = 1 ,dimension = 5):
    x,y = np.meshgrid(np.linspace(-2,2,dimension),np.linspace(-2,2,dimension))
    kernel = np.zeros((dimension,dimension))
    kernel = (np.exp(-1*(x**2 + y**2)/(2 * (sigma**2))))/(2*np.pi*sigma*sigma)
    kernel = kernel/np.sum(kernel)
    return kernel

# generate sobels kernels
def generate_sobels_kernels():
    kx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]],np.float32)
    ky = np.transpose(kx)*-1
    return (kx,ky)

# convolve the image with the kernel
def convolve(img_arr,kernel):
    kernel_row,kernel_col = kernel.shape
    img_row,img_col = img_arr.shape
    pad_h  = (kernel_row-1)//2
    pad_w = (kernel_col-1)//2 
    pad_img = np.zeros((img_row + 2*pad_h , img_col + 2*pad_w),dtype=np.uint8)
    pad_img[pad_h:(pad_img.shape[0]-pad_h),pad_w:(pad_img.shape[1]-pad_w)] = img_arr
    output_arr = np.zeros(img_arr.shape)
    for i in range(img_row):
        for j in range(img_col):
            output_arr[i , j] = int(np.sum(pad_img[ i:(i+kernel_row) , j:(j+kernel_col)]*kernel))
    return output_arr

# Applies Gaussian Blur to the input image and outputs the Blurred Image in array format
def apply_gaussian_blur(img_arr):
    kernel = generate_gaussian_kernel()
    n_arr = convolve(img_arr,kernel)
    return n_arr


# After applying Gaussian Blur it applies sobel's filters by convolving the image with the sobels Kernerls
# and returns the gradient Matrix and direction Matrix 
def sobel_filtering(img_arr):
    kx,ky = generate_sobels_kernels()
    Ix = convolve(img_arr,kx)
    Iy = convolve(img_arr,ky)
    # hypot(x,y) claculates sqrt(x*x + y*y)
    gradient = np.hypot(Ix,Iy)
    gradient = (gradient * 255.0 )/(gradient.max())
    # arctan2(y,x) returns the angle in radians of tan-1(y/x)
    edge_direction = np.arctan2(Iy,Ix)
    return (gradient,edge_direction)

# Considering the Gradient Matrix and Edge Direction Matrix received after applying the Sobels Filters
# edges are thinned out and the resultant image array is returned
def non_max_suppression(gradient_matrix,edge_direction):
    row,col = gradient_matrix.shape
    output_arr = np.zeros(gradient_matrix.shape,dtype=np.uint8)
    # converting the edge_direction from radians into degrees
    direction = (edge_direction * 180.0)/np.pi
    # if angle is -ve then adding 180 degree to it
    direction[ direction < 0] += 180
    for i in range(1,row-1):
        for j in range(1,col-1):
            q = 255
            r = 255
            
            #angle 0
            if (0 <= direction[i,j] < 22.5) or (157.5 <= direction[i,j] <= 180):
                q = gradient_matrix[i, j+1]
                r = gradient_matrix[i, j-1]
            #angle 45
            elif (22.5 <= direction[i,j] < 67.5):
                q = gradient_matrix[i+1, j-1]
                r = gradient_matrix[i-1, j+1]
            #angle 90
            elif (67.5 <= direction[i,j] < 112.5):
                q = gradient_matrix[i+1, j]
                r = gradient_matrix[i-1, j]
            #angle 135
            elif (112.5 <= direction[i,j] < 157.5):
                q = gradient_matrix[i-1, j-1]
                r = gradient_matrix[i+1, j+1]

            if (gradient_matrix[i,j] >= q) and (gradient_matrix[i,j] >= r):
                output_arr[i,j] = int(gradient_matrix[i,j])
            else:
                output_arr[i,j] = 0
    return output_arr

# after thinning out the edges we clasify the edges as weak , strong and non-relevant and 
# output a 3 pixel image consisting of only weak,strong and non-relevant pixels , value of strong and weak pixel
def double_threshold(img_arr,lowThresholdRatio = 0.05 , highThresholdRatio = 0.09):
    
    highThreshold = img_arr.max()*highThresholdRatio
    lowThreshold = highThreshold*lowThresholdRatio
    
    output_arr = np.zeros(img_arr.shape,dtype=np.uint8)
    
    weak = np.uint8(25)
    strong = np.uint8(255)

    strong_i, strong_j = np.where(img_arr >= highThreshold)
    weak_i, weak_j = np.where((img_arr <= highThreshold) & (img_arr >= lowThreshold))
    
    output_arr[strong_i, strong_j] = strong
    output_arr[weak_i, weak_j] = weak
    
    return (output_arr,strong,weak)

# After performing double Threshold and we decide which weak pixels contribute to the edges and return the final image
def hysteresis(img_arr,weak,strong=255):
    for i in range(1,img_arr.shape[0]-1):
        for j in range(1,img_arr.shape[1]-1):
            if(img_arr[i][j] == weak):
                if ((img_arr[i+1, j-1] == strong) or (img_arr[i+1, j] == strong) or (img_arr[i+1, j+1] == strong)
                        or (img_arr[i, j-1] == strong) or (img_arr[i, j+1] == strong)
                        or (img_arr[i-1, j-1] == strong) or (img_arr[i-1, j] == strong) or (img_arr[i-1, j+1] == strong)):
                        img_arr[i, j] = strong
                else:
                    img_arr[i, j] = 0
    return img_arr

# Combine all the required steps for canny edge detection and return the final image for the provided image array
def canny_edge_detection(img_arr):
    
    blur_arr = apply_gaussian_blur(img_arr)
    
    gradient_matrix,direction = sobel_filtering(blur_arr)
    
    non_max_suppress_arr = non_max_suppression(gradient_matrix,direction)

    threshold_arr,strong,weak = double_threshold(non_max_suppress_arr)

    final_arr = hysteresis(threshold_arr,weak,strong)
    
    return final_arr

if __name__ == "__main__":
    img = Image.open('assets/grayscale.jpg')
    img_arr = np.array(img)
    canny_edge_detection(img_arr)
    canny_arr = canny_edge_detection(img_arr)
    canny_img = Image.fromarray(canny_arr)
    canny_img.show(title="Canny Edge Detection")
    canny_img.save("results/Canny Edge Detection.jpg")