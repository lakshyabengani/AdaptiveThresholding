from PIL import Image
import numpy as np

def generate_kernel(sigma,dim):
    x,y = np.meshgrid(np.linspace(-2,2,dim),np.linspace(-2,2,dim))
    kernel = np.zeros((dim,dim))
    kernel = np.exp(-1*(x**2 + y**2)/(2 * (sigma**2)))/(2*np.pi*sigma*sigma)
    kernel = kernel/np.sum(kernel)
    return kernel

def apply_filter(img_arr,domain_kernel,sigma_r):
    kernel_row,kernel_col = domain_kernel.shape
    img_row,img_col = img_arr.shape
    pad_h  = (kernel_row-1)//2
    pad_w = (kernel_col-1)//2 
    pad_img = np.zeros((img_row + 2*pad_h , img_col + 2*pad_w),dtype=np.uint8)
    pad_img[pad_h:(pad_img.shape[0]-pad_h),pad_w:(pad_img.shape[1]-pad_w)] = img_arr
    output_arr = np.zeros(img_arr.shape)
    for i in range(img_row):
        for j in range(img_col):
            I = pad_img[ i:(i+kernel_row) , j:(j+kernel_col)]
            range_filter = (np.exp(-1*(I-img_arr[i][j]))/(2*(sigma_r**2)))/(2*np.pi*sigma_r*sigma_r)
            range_filter = range_filter/np.sum(range_filter)
            bilateral_filter = range_filter*domain_kernel
            norm = np.sum(bilateral_filter) 
            output_arr[i , j] = int(np.sum(bilateral_filter*I)/norm)
    return output_arr

def BilateralfilterSingleChannel(arr,sigma_r,sigma_d,window):
    new_arr = np.zeros(arr.shape)
    domain_kernel = generate_kernel(sigma_d,window)
    new_arr = apply_filter(arr,domain_kernel,sigma_r)
    new_arr.astype('uint8')
    return new_arr

def Bilateralfilter(arr,sigma_r = 1,sigma_d = 1,window = 5):
    new_arr = np.zeros(arr.shape,dtype='uint8')
    for k in range(arr.shape[2]):
        n_arr = BilateralfilterSingleChannel(arr[:,:,k],sigma_r,sigma_d,window)
        for i in range(n_arr.shape[0]):
            for j in range(n_arr.shape[1]):
                new_arr[i][j][k] = n_arr[i][j]
    return new_arr

if __name__ == "__main__":

    img = Image.open('assets/hacking.jpg')
    old_arr = np.array(img)
    new_arr = np.zeros(old_arr.shape,dtype='uint8')

    sigma_d = 1
    sigma_r = 30
    window = 9
    cycle = 14

    for i in range(cycle):
        print("executing round ",i)
        new_arr = Bilateralfilter(old_arr,sigma_r,sigma_d,window)
        # print(np.sum(new_arr - old_arr))
        old_arr = new_arr

    new_img = Image.fromarray(new_arr)
    new_img.show(title="Bilateral Filter")
    new_img.save("results/Bilateral filter.jpg")
