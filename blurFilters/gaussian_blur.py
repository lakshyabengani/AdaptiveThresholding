from PIL import Image
import numpy as np

def generate_kernel(sigma,dim):
    x,y = np.meshgrid(np.linspace(-2,2,dim),np.linspace(-2,2,dim))
    kernel = np.zeros((dim,dim))
    kernel = (np.exp(-1*(x**2 + y**2)/(2 * (sigma**2))))/(2*np.pi*sigma*sigma)
    kernel = kernel/np.sum(kernel)
    return kernel

def convolve(i,j,k,arr):
    if i<k or j<k or i+k >= arr.shape[0] or j+k >= arr.shape[1]:
        return arr[i][j]
    else:
        total = 0
        for a in range(0,2*k+1):
            for b in range(0,2*k+1):
                total += arr[i-k+a][j-k+b]*kernel[a][b]
        return total

def GausianBlur(arr,kernel):
    new_arr = np.zeros(arr.shape)
    k = kernel.shape[0]//2
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            new_arr[i][j] = convolve(i,j,k,arr)
    new_arr.astype('uint8')
    return new_arr

if __name__ == "__main__":
    img = Image.open('../assets/hacking.jpg')
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
    new_img.save('GaussianBlur.jpg')
            
    
