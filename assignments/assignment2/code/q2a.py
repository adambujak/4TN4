import numpy as np
import cv2
import time

# import messi image
messi = cv2.imread('messi.jpg', 0)
# import ronaldo image
ronaldo = cv2.imread('ronaldo.jpg', 0)

messi_fft = np.fft.fft2(messi)
ronaldo_fft = np.fft.fft2(ronaldo)

magnitude_messi = np.abs(messi_fft)
phase_messi = np.angle(messi_fft)

magnitude_ronaldo = np.abs(ronaldo_fft)
phase_ronaldo = np.angle(ronaldo_fft)

messi_ronaldo = np.multiply(magnitude_messi, np.exp(1j*phase_ronaldo))
ronaldo_messi = np.multiply(magnitude_ronaldo, np.exp(1j*phase_messi))

messi_mag = np.fft.ifft2(messi_ronaldo)
ronaldo_mag = np.fft.ifft2(ronaldo_messi)

messi_mag = np.abs(messi_mag)
ronaldo_mag = np.abs(ronaldo_mag)
#save img
cv2.imwrite('messi_mag_ronaldo_phase.png', messi_mag)
cv2.imwrite('ronaldo_mag_messi_phase.png', ronaldo_mag)
