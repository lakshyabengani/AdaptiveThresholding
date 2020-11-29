from PIL import Image
import numpy as np

# Taking a window starting from (i,j) as its top left corner and h,w as its maximum height and width respectively
# The threshold is calucated as the mean of all the pixels in that particular window
# r,c is the no:of rows and columns in the considered window
# background is assumed to be white and borders/edges detected as black

def window_thresholding(arr,i,j,h,w,new_arr):
    if i+h >= arr.shape[0] or j+w >= arr.shape[1]:
        r = arr.shape[0]
        c = arr.shape[1]
    else:
        r = i + h
        c = j + w 
    window = arr[i:r,j:c]
    threshold = (np.sum(window))/((r-i)*(c-j))   
    for a in range(i,r):
        for b in range(j,c):
            if arr[a][b] >= threshold :
                new_arr[a][b] = 255
            else:
                new_arr[a][b] = 0          

# An array of the grayscale image is taken as the input
# h and w are the desired height and width of the sub_images(or windows)
# Here we want 16 such windows
# Global Thresholding is done for each window and the results combined into a single array to be output
                    
def AdaptiveThreshold(arr):
    new_arr = np.zeros(arr.shape)
    h = arr.shape[0]//4
    w = arr.shape[1]//4
    for i in range(0,arr.shape[0],h):
        for j in range(0,arr.shape[1],w):
            window_thresholding(arr,i,j,h,w,new_arr)
    return new_arr


if __name__ == '__main__':
    
    img = Image.open('assets\actressBlur.jpg').convert('L')
    arr = np.array(img)

    new_arr = np.zeros(arr.shape,dtype='uint8')

    n_arr = AdaptiveThreshold(arr)

    for i in range(n_arr.shape[0]):
        for j in range(n_arr.shape[1]):
            new_arr[i][j] = int(n_arr[i][j])

    new_img = Image.fromarray(new_arr)
    new_img.save('assets\actressAdaptiveThreshold.jpg')
    
