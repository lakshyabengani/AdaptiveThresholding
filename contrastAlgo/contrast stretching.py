import numpy as np
from PIL import Image

# only for grayscale images
def contrast_stretch(img):
    arr = np.array(img)
    hist = img.histogram()
    c=0
    d=0
    t = arr.shape[1]*arr.shape[0]
    l = 0
    for i in range(256):
        if (l + hist[i]) <= 0.05*t:
            c = i
            l += hist[i]
        else:
            break
    l = 0 
    for i in range(255,-1,-1):
        if (l + hist[i]) <= 0.05*t:
            l += hist[i]
            d = i
        else:
            break

    r = 255/(d-c)
    
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            l = (arr[i][j] - c)*r
            arr[i][j] = max(0,min(255,l))
    new_img = Image.fromarray(arr)
    return new_img

if __name__ == "__main__":
    img = Image.open('../assets/original.jpg')
    n_img = contrast_stretch(img)
    n_img.show(title="Contrast Stretching")
