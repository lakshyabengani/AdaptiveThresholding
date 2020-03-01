import numpy as np
from PIL import Image

def create_window(x):
    window = np.zeros((1,x**2))
    return window

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

def MedianBlur(arr,window):
    k = np.sqrt(window.shape[1])
    k = int(k//2)
    new_arr = np.zeros(arr.shape)
    for i in range(arr.shape[0]):
	for j in range(arr.shape[1]):
		new_arr[i][j] = slide_window(arr,i,j,k,window)
    new_arr.astype('uint8')
    return new_arr

if __name__ == "__main__":
    img = Image.open('original.jpg')
    arr = np.array(img)
    window = create_window(5)
    new_arr = np.zeros(arr.shape,dtype='uint8')
    n_arr = MedianBlur(arr,window)
    for i in range(n_arr.shape[0]):
        for j in range(n_arr.shape[1]):
            new_arr[i][j] = int(n_arr[i][j])
    new_img = Image.fromarray(new_arr)
    new_img.save('MeadianBlur.jpg')
    
