import numpy as np
from PIL import Image

#edge detection using sobels method

# generate sobels kernels
def sobels_kernels():
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
            output_arr[i , j] = np.sum(pad_img[ i:(i+kernel_row) , j:(j+kernel_col)]*kernel)
    return output_arr

def edge_detection(arr):
    kx,ky = sobels_kernels()
    gx = convolve(arr,kx)
    gy = convolve(arr,ky)
    new_arr = np.hypot(gx,gy)
    new_arr = (new_arr * 255.0)/(new_arr.max())
    output_arr = np.zeros(arr.shape,dtype=np.uint8)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            output_arr[i][j] = int(new_arr[i][j])
    return output_arr

if __name__ == "__main__":
    img = Image.open('assets/grayscale.jpg')
    arr = np.array(img)
    ans = edge_detection(arr)
    new_img = Image.fromarray(ans)
    new_img.show(title="Sobels Edge Detection")
    new_img.save("results/Sobels Edge Detection.jpg")
