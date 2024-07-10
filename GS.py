from PIL import Image
import numpy as np
from scipy.fftpack import fft2, ifft2, fftshift
import matplotlib.pyplot as plt
from tqdm import tqdm

class GS:
    def __init__(self, image):
        self.raw_image = np.array(image)
        self.width, self.height = self.raw_image.shape[0], self.raw_image.shape[1]
        self.amplitude = self.norm_amplitude()
        self.phase = 2 * np.pi * np.random.rand(self.width, self.height)
        self.complex_amplitude = self.amplitude * np.exp(1j * self.phase)
        self.RMSE = None
        self.phase_result = None
        self.result = None

    def norm_amplitude(self):
        return self.raw_image / np.max(self.raw_image)

    def train(self, epoch=500):
        self.RMSE = np.zeros(epoch)
        for i in tqdm(range(epoch)):
            freq_img = ifft2(fftshift(self.complex_amplitude))
            f_img_phase = np.angle(freq_img)
            f_img_norm = self.amplitude * np.exp(1j * f_img_phase)
            space_img = fft2(fftshift(f_img_norm))
            error = np.abs(self.amplitude) - fftshift(np.abs(space_img) / np.max(space_img))
            self.RMSE[i] = np.sqrt(np.mean(np.power(error, 2)))
            self.complex_amplitude = np.abs(self.amplitude) * (space_img / np.abs(space_img))
        self.phase_result = np.angle(freq_img)
        self.result = np.abs(fftshift(space_img))

        plt.figure(0)
        plt.imshow(self.raw_image, cmap="gray")
        plt.figure(1)
        plt.imshow(self.phase_result, cmap="gray")
        plt.figure(2)
        plt.imshow(self.format_image(self.result), cmap="gray")
        plt.figure(3)
        plt.plot(list(range(epoch)), self.RMSE)
        plt.show()

        # Save the phase mask image as a bitmap
        phase_mask_image = Image.fromarray(self.format_image((self.phase_result / np.pi) * 127.5 + 127.5))
        phase_mask_image.save("quadrant_1_image_0_resized_phasemask.bmp")

    def format_image(self, img):
        img = img * 255 / np.max(img)
        img = img.astype(np.uint8)
        return img

if __name__ == "__main__":
    image_path = "images/resized_quadrant_1_image_0 copy.png"
    g = Image.open(image_path, mode="r")
    g = g.convert("L")
    g = g.resize((256, 256), Image.BILINEAR)
    gs = GS(g)
    gs.train()