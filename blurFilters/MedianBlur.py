'''
MEDIAN BLUR/FILTER : 

This algorithm filter considers each pixel in the image in turn and looks at its 
nearby neighbors to decide whether or not it is representative of its surroundings.

'''
import numpy as np
from PIL import Image

# Create a window of specific size x
def create_window(x):
    window = np.zeros((1,x**2))
    return window

# Caluclates the output pixel value for the provided pixel in the input image
def slide_window(arr,i,j,k,window):
    if i<k or j<k or i+k >= arr.shape[0] or j+k >= arr.shape[1]:
        return arr[i][j]
    else:
    	x = 0
    	for a in range(0,2*k+1):
    	    for b in range(0,2*k+1):
    		    window[0][x] = arr[i-k+a][j-k+b]
    		    x += 1
    	window = np.sort(window)
    	x = window.shape[0]*window.shape[1]
    	return window[0][x//2]

# This function performs the Median Blur operation on a single clour channel and returns the output array for that channel
def MedianBlur(arr,window):
    k = np.sqrt(window.shape[1])
    k = int(k//2)
    new_arr = np.zeros(arr.shape)

    # for each pixel calculate the pixel for the blurred image 
    for i in range(arr.shape[0]):
	    for j in range(arr.shape[1]):
		    new_arr[i][j] = slide_window(arr,i,j,k,window)
    new_arr.astype('uint8')

    return new_arr

if __name__ == "__main__":
    img = Image.open('assets/hacking.jpg')
    arr = np.array(img)

    # define the window size and create a window
    window = create_window(5)

    # create a new numpy array with the same shape as that of the pic
    # this is serve as the array fo the new pic 
    new_arr = np.zeros(arr.shape,dtype='uint8')

    # for each colour channel perform Median Blur
    for k in range(arr.shape[2]):
        n_arr = MedianBlur(arr[:,:,k],window)
        for i in range(n_arr.shape[0]):
            for j in range(n_arr.shape[1]):
                new_arr[i][j][k] = int(n_arr[i][j])

    # Combine the results from all the channels  and save the pic 
    new_img = Image.fromarray(new_arr)
    new_img.show(title="Median Blur")
    new_img.save('results/MedianBlur.jpg')    
