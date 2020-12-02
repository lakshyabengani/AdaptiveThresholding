from PIL import Image
import numpy as np

def generate_kernel(sigma,dim):
    x,y = np.meshgrid(np.linspace(-2,2,dim),np.linspace(-2,2,dim))
    kernel = np.zeros((dim,dim))
    kernel = (np.exp(-1*(x**2 + y**2)/(2 * (sigma**2))))/(2*np.pi*sigma*sigma)
    kernel = kernel/np.sum(kernel)
    return kernel

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

def GausianBlur(arr,kernel):
    new_arr = np.zeros(arr.shape)
    new_arr = convolve(arr,kernel)
    new_arr.astype('uint8')
    return new_arr

if __name__ == "__main__":
    img = Image.open('assets/hacking.jpg')
    sigma = 1
    kernel_dimension = 5
    arr = np.array(img)
    kernel = generate_kernel(sigma,kernel_dimension)
    new_arr = np.zeros(arr.shape,dtype='uint8')
    for k in range(arr.shape[2]):
        n_arr = GausianBlur(arr[:,:,k],kernel)
        for i in range(n_arr.shape[0]):
              for j in range(n_arr.shape[1]):
                  new_arr[i][j][k] = int(n_arr[i][j])
    new_img = Image.fromarray(new_arr)
    new_img.show(title="Gaussian Blur")
    new_img.save("results/Gaussian Blur.jpg")
    
