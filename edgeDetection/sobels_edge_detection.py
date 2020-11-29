import numpy as np
from PIL import Image

#edge detection using sobels method


def sobels_kernels():
    kx = np.zeros((3,3))
    ky = np.zeros((3,3))
    kx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]],np.float32)
    ky = np.transpose(kx)*-1
    return (kx,ky)

def edge_detection(img):
    arr = np.array(img)
    r,c = arr.shape
    r -= 3
    c -= 3
    new_arr = np.zeros((r,c))
    gx = np.zeros((r,c))
    gy = np.zeros((r,c))
    kx,ky = sobels_kernels()
    for i in range(r):
        for j in range(c):
            gx[i][j] = np.sum(arr[i:i+3,j:j+3]*kx)
            gy[i][j] = np.sum(arr[i:i+3,j:j+3]*ky)
    new_arr = abs(gx) + abs(gy)
    new_arr.astype('uint8')
    new_img = Image.fromarray(new_arr)
    return new_img

if __name__ == "__main__":
    img = Image.open('original.jpg')
    ans = edge_detection(img)
    ans.show("Sobels Edge Detection")
