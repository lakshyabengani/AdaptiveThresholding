from PIL import Image
import numpy as np

# for grayscale images
def histogram_equilization(img):
    arr = np.array(img)
    hist = img.histogram()
    for i in range(1,256):
        hist[i] += hist[i-1]
    nj = hist - np.amin(hist)
    hist = (nj/(np.amax(hist) + np.amin(hist)))*255
    hist.astype('uint8')
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            arr[i][j] = hist[arr[i][j]]

    new_img = Image.fromarray(arr)
    return new_img

if __name__ == "__main__":
    img = Image.open('../assets/original.jpg')
    n_img = histogram_equilization(img)
    n_img.show(title="Histogram Equilization")    
