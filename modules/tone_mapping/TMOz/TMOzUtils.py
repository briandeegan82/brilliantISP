# -*- coding: utf-8 -*-
"""
Utility Functions for TMOz Pipeline
Created on Mon Sep 22 16:58:05 2025
@author: Imra
"""

import numpy as np
import imageio
from pathlib import Path
from scipy.ndimage import gaussian_filter, zoom
import matplotlib.pyplot as plt


class utils:
    # -------------------------------
    # File I/O
    # -------------------------------
    @staticmethod
    def saveFile(sdr, img_name, output_dir="output"):
        """Save SDR result as PNG in output directory."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        file_path = output_path / f"{img_name}.png"
        imageio.imwrite(file_path, np.clip(sdr.astype(np.float32)/65535*255, 0,65535).astype(np.uint8))
        print(f"Saved: {file_path}")
        
    def imshow (rgbimg, titleStr):
        plt.imshow(rgbimg)
        plt.axis("off")
        plt.title(titleStr)
        plt.show()

    # -------------------------------
    # Color Space Conversions
    # -------------------------------
    @staticmethod
    def xyz2xyY(xyz):
        """Convert from XYZ to xyY."""
        sum1 = np.sum(xyz, axis=1, keepdims=True)
        x = xyz[:, 0:1] / sum1
        y = xyz[:, 1:2] / sum1
        return np.hstack([x, y, xyz[:, 1:2]])
    
    def getcond(cond):
        """
        Extract surround conditions from the input structure
        """
        XYZw1 = cond['XYZw1']
        Yb = cond['Yb']
        La1 = cond['Lw1'] * Yb / 100
        sr = cond['surround']
        XYZw2 = cond['XYZw2']
        La2 = cond['Lw2'] * Yb / 100

        return XYZw1, La1, Yb, sr, XYZw2, La2

    @staticmethod
    def xyY2xyz(xyY):
        """Convert from xyY to XYZ."""
        x = xyY[:, 0]
        y = xyY[:, 1]
        if xyY.shape[1] == 3:
            Y = xyY[:, 2]
        else:
            Y = np.ones_like(x) * 100
        X = (x / y) * Y
        Z = ((1 - x - y) / y) * Y
        return np.column_stack([X, Y, Z])
    
    @staticmethod
    def srgb2xyzLinear(img):
        """
        Convert an sRGB image (luminance in cd/m^2) to XYZ using the sRGB matrix.
        
        Parameters:
            img (numpy.ndarray): Input RGB image of shape (H, W, 3)
        
        Returns:
            XYZpred (numpy.ndarray): Output XYZ values as a 2D array (H*W, 3)
        """
        # sRGB to XYZ conversion matrix
        M = np.array([
            [0.412424, 0.212656, 0.0193324],
            [0.357579, 0.715158, 0.119193],
            [0.180464, 0.0721856, 0.950444]
        ])
        
        # Flatten image to (num_pixels, 3)
        H, W, C = img.shape
        scalars = img.reshape(H * W, C)
        
        # Matrix multiplication
        XYZpred = scalars @ M  # Equivalent to scalars * M in MATLAB
        
        # Avoid very small values
        XYZpred[XYZpred < 1e-8] = 1e-8
        
        return XYZpred


    @staticmethod
    def xyz2srgb(XYZ):
        """Convert XYZ to 8-bit sRGB."""
        if XYZ.shape[1] != 3:
            raise ValueError("XYZ must be n by 3")

        if np.max(XYZ[:, 1]) > 1:
            XYZ = utils.rescale21(XYZ)

        M = np.array([
            [3.2406, -1.5372, -0.4986],
            [-0.9689,  1.8758,  0.0415],
            [0.0557, -0.2040,  1.0570]
        ])

        RGB = (M @ XYZ.T).T
        return np.clip(RGB*65535, 0, 65535).astype(np.uint16)
        # RGB = np.clip(RGB, 0, 1)

        # DACS = np.where(RGB <= 0.0031308,
        #                 12.92 * RGB,
        #                 1.055 * (RGB ** (1 / 2.4)) - 0.055)

        # RGB = np.ceil(DACS * 255).astype(np.uint8)
        # return np.clip(RGB, 0, 255)

    @staticmethod
    def rescale21(XYZ):
        """Rescale XYZ luminance to [0, 1]."""
        xyy = utils.xyz2xyY(XYZ)
        maxy = np.max(xyy[:, 2])
        y = (xyy[:, 2] / maxy)
        y[y < 0] = 0
        xyy[:, 2] = y
        return utils.xyY2xyz(xyy)

    # -------------------------------
    # Tone Mapping Helpers
    # -------------------------------
    @staticmethod
    def TMOzclip(xyzin):
        """Apply luminance clipping for TMOz."""
        # Convert XYZ → xyY
        xyy = utils.xyz2xyY(xyzin)
        outY = xyy[:, 2]

        # Normalize luminance
        outY = outY / np.max(outY)

        # Clip 1% dark/light
        min_y = np.percentile(outY, 1)
        max_y = np.percentile(outY, 99)

        outY1 = (outY - min_y) / (max_y - min_y)
        outY1 = np.clip(outY1, 0, 1)
        outY1[outY1 <= 0] = 1e-9

        # Update luminance
        xyy[:, 2] = outY1

        # Convert back xyY → XYZ
        return utils.xyY2xyz(xyy)
    @staticmethod
    def imgKey(L):
        """Calculate key value from luminance."""
        L = L / np.max(L)
        LMin = utils.MaxQuart(L, 0.01)
        LMax = utils.MaxQuart(L, 0.99)

        log2Min = np.log2(LMin + 1e-9)
        log2Max = np.log2(LMax + 1e-9)
        logAverage = utils.logMean(L)
        log2Average = np.log2(logAverage + 1e-9)

        return 0.18 * 4 ** ((2.0 * log2Average - log2Min - log2Max) /
                             (log2Max - log2Min))
    @staticmethod
    def MaxQuart(matrix, percentile):
        """Percentile helper."""
        flat = np.sort(matrix.ravel())
        idx = max(int(round(len(flat) * percentile)), 1) - 1
        return flat[idx]
    
    @staticmethod
    def logMean(img):
        """Logarithmic mean luminance."""
        delta = 1e-6
        return np.exp(np.mean(np.log(img + delta)))
    
    @staticmethod
    def Qimg_LocalContrast_Enhancement(detail):
        """Enhance local contrast."""
        maxd = np.max(detail)
        return maxd * (detail / maxd) ** 1
    
    @staticmethod
    def tonecurveM(base_Q, key):
        """Tone curve adjustment."""
        a, b = 1.6781, 0.3128
        gamma = a * key + b
        return np.power(base_Q, gamma)
    def fast_bilateral_filter(self, image):
        """
        Approximate bilateral filtering using a downsampled approach.
        """
        small_img = zoom(image, 1 / self.downsample_factor, order=1)
        small_filtered = self.bilateral_filter(small_img, self.sigma_color, self.sigma_space)
        return zoom(small_filtered, self.downsample_factor, order=1)
    
    def bilateral_filter(self, Qimg, sigma_color=0.4, sigma_space=2.0):
        """
        Custom bilateral filter using Gaussian filtering approximation.
        """
        spatial_filtered = gaussian_filter(Qimg, sigma=sigma_space)
        intensity_diff = Qimg - spatial_filtered
        range_kernel = np.exp(-0.5 * (intensity_diff / sigma_color) ** 2)
        log_base= spatial_filtered + range_kernel * intensity_diff
        log_detail = Qimg - log_base
        return 10**log_base, 10**log_detail
    
    # --------------------------
# CAM16 Functions
# --------------------------
    def XYZ2CAM16Q_RGBa(XYZ, XYZw, La=None, Yb=20, surround='avg'):
        """
        Convert XYZ values to CAM16Q color space (brightness Q and adapted RGB).
        
        Parameters:
            XYZ      : (n,3) array of XYZ values
            XYZw     : 3-element array, reference white point
            La       : adapted luminance (scalar)
            Yb       : background luminance (default 20)
            surround : 'avg', 'dim', 'dark', or 'T1'
        
        Returns:
            Q    : CAM16Q brightness (n-element array)
            RGBa : adapted RGB values (n,3 array)
        """
        # ----------------------
        # Defaults
        if La is None:
            La = 2000 / (5 * np.pi)
    
        # Surround parameters
        surround_map = {
            'avg':  (0.69, 1, 1),
            'dim':  (0.59, 0.9, 0.9),
            'dark': (0.525, 0.8, 0.8),
            'T1':   (0.46, 0.9, 0.9)
        }
        c, Nc, F = surround_map.get(surround, (0.69, 1, 1))
    
        # ----------------------
        # Step 0: CAT16 matrix
        M_CAT16 = np.array([
            [0.401288, 0.650173, -0.051461],
            [-0.250268, 1.204414, 0.045854],
            [-0.002079, 0.048952, 0.953127]
        ])
    
        # Convert reference white
        RGBw = M_CAT16 @ XYZw
    
        # Degree of adaptation
        D_pre = F * (1 - (1/3.6) * np.exp((-La - 42) / 92))
        D = np.clip(D_pre, 0, 1)
    
        # Chromatic adaptation ratios
        Dr = D * (XYZw[1] / RGBw[0]) + 1 - D
        Dg = D * (XYZw[1] / RGBw[1]) + 1 - D
        Db = D * (XYZw[1] / RGBw[2]) + 1 - D
    
        # Luminance factors
        k = 1 / (5 * La + 1)
        Fl = 0.2 * (k**4) * (5 * La) + 0.1 * ((1 - k**4)**2) * ((5 * La)**(1/3))
        n = Yb / XYZw[1]
        z = 1.48 + np.sqrt(n)
        Nbb = 0.725 * (1 / n)**0.2
    
        # Apply chromatic adaptation to reference white
        RGBwc = np.array([Dr*RGBw[0], Dg*RGBw[1], Db*RGBw[2]])
        RGBaw = (400 * (Fl * RGBwc / 100)**0.42) / (27.13 + (Fl * RGBwc / 100)**0.42) + 0.1
        Aw = (2*RGBaw[0] + RGBaw[1] + RGBaw[2]/20 - 0.305) * Nbb
    
        # Convert test colors
        RGB = (M_CAT16 @ XYZ.T).T  # shape (n,3)
        RGBc = np.zeros_like(RGB)
        RGBc[:,0] = Dr * RGB[:,0]
        RGBc[:,1] = Dg * RGB[:,1]
        RGBc[:,2] = Db * RGB[:,2]
    
        # Nonlinear response
        RGBa = np.zeros_like(RGBc)
        div100 = 0.01
        indxg = RGBc >= 0
        indxl = RGBc < 0
    
        RGBa[indxg] = (400 * (Fl * RGBc[indxg] * div100)**0.42) / (27.13 + (Fl * RGBc[indxg] * div100)**0.42) + 0.1
        RGBa[indxl] = (-400 * (-Fl * RGBc[indxl] * div100)**0.42) / (27.13 + (-Fl * RGBc[indxl] * div100)**0.42) + 0.1
    
        # Brightness correlate
        A = (2*RGBa[:,0] + RGBa[:,1] + RGBa[:,2]/20 - 0.305) * Nbb
        J = 100 * (A / Aw)**(c * z)
        Q = (4 / c) * np.sqrt(J / 100) * (Aw + 4) * (Fl**0.25)
    
        return np.real(Q), np.real(RGBa)
    
    @staticmethod

    def newM(Q, XYZw, La=None, Yb=20, Surround='avg', RGBa=None):
        """
        Calculate the new colorfulness (M) and hue (h) from CAM16Q brightness values.
        
        Parameters:
            Q       : (N,) array of CAM16Q brightness values
            XYZw    : 3-element array, reference white
            La      : adaptive luminance (default 2000/(5*pi))
            Yb      : background luminance factor (default 20)
            Surround: 'avg', 'dim', 'dark', 'T1'
            RGBa    : (N,3) array of adapted RGB values
        
        Returns:
            M : (N,) array of colorfulness values
            h : (N,) array of hue angles in degrees
        """
        if La is None:
            La = 2000 / (5 * np.pi)
        if RGBa is None:
            raise ValueError("RGBa must be provided")

        # Surround parameters
        surrounds = {
            'avg':  (0.69, 1.0, 1.0),
            'dim':  (0.59, 0.9, 0.9),
            'dark': (0.525, 0.8, 0.8),
            'T1':   (0.46, 0.9, 0.9)
        }
        c, Nc, F = surrounds.get(Surround, (0.69, 1.0, 1.0))

        # CAT16 matrix
        M_CAT16 = np.array([[0.401288, 0.650173, -0.051461],
                            [-0.250268, 1.204414, 0.045854],
                            [-0.002079, 0.048952, 0.953127]])
        
        RGBw = M_CAT16 @ XYZw

        # Degree of adaptation
        D_pre = F * (1 - (1 / 3.6) * np.exp((-La - 42) / 92))
        D = np.clip(D_pre, 0, 1)
        Dr, Dg, Db = D * XYZw[1] / RGBw + (1 - D)

        # Lightness factor
        k = 1 / (5 * La + 1)
        Fl = 0.2 * (k**4) * (5*La) + 0.1 * ((1 - k**4)**2) * ((5*La)**(1/3))
        n = Yb / XYZw[1]
        Nbb = 0.725 * n**-0.2

        # Adapted white RGB
        RGBwc = np.array([Dr*RGBw[0], Dg*RGBw[1], Db*RGBw[2]])
        RGBaw = (400 * (Fl*RGBwc/100)**0.42) / (27.13 + (Fl*RGBwc/100)**0.42) + 0.1

        # Compute Aw and J
        Aw = (2*RGBaw[0] + RGBaw[1] + RGBaw[2]/20 - 0.305) * Nbb
        J = 6.25 * ((c * Q) / ((Aw + 4) * Fl**0.25))**2

        # Chromaticity a and b
        a = RGBa[:,0] - 12*RGBa[:,1]/11 + RGBa[:,2]/11
        b = (RGBa[:,0] + RGBa[:,1] - 2*RGBa[:,2])/9

        # Hue angle in degrees
        h = np.degrees(np.arctan2(b, a))
        h[h < 0] += 360

        # Chromatic induction factor
        et = (np.cos(2 + np.radians(h)) + 3.8)/4
        t = ((Nc * Nbb * 50000 / 13) * (et * np.sqrt(a**2 + b**2))) / \
            (RGBa[:,0] + RGBa[:,1] + 21*RGBa[:,2]/20)

        # Colorfulness
        C = (t**0.9) * np.sqrt(J/100) * (1.64 - 0.29**n)**0.73
        M = C * Fl**0.25

        return M, h



    @staticmethod
    def CAM16UCS2XYZ_QMhs(QMh, XYZw, La=2000 / (np.pi * 5), Yb=20, surround="avg"):
        """
        Convert CAM16-UCS (Q, M, h) values back to XYZ color space.
    
        Parameters:
            QMh      : (N,3) array: [Q, M, h] in CAM16-UCS
            XYZw     : 3-element array, reference white
            La       : adapted luminance (default 2000/(5*pi))
            Yb       : background luminance factor (default 20)
            Surround : 'avg', 'dim', 'dark', 'T1'
    
        Returns:
            XYZ : (N,3) array in CIE XYZ color space
        """
        XYZw = np.asarray(XYZw, dtype=float)
        QMh = np.atleast_2d(np.asarray(QMh, dtype=float))
    
        # Defaults
        if La is None:
            La = 2000 / (np.pi * 5)
        if Yb is None:
            Yb = 20
    
        # Surround parameters
        surrounds = {
            'avg':  (0.69, 1.0, 1.0),
            'dim':  (0.59, 0.9, 0.9),
            'dark': (0.525, 0.8, 0.8),
            'T1':   (0.46, 0.9, 0.9)
        }
        c, Nc, F = surrounds.get(surround, (0.69, 1.0, 1.0))

    
        # Step 0: constants
        k = 1 / (5 * La + 1)
        FL = 0.2 * k**4 * 5 * La + 0.1 * (1 - k**4)**2 * (5 * La)**(1/3)
    
        # Step 1: CAT16 matrices
        M_CAT16 = np.array([[0.401288, 0.650173, -0.051461],
                            [-0.250268, 1.204414, 0.045854],
                            [-0.002079, 0.048952, 0.953127]])
        M_CAT16inv = np.array([[1.86206786, -1.01125463, 0.14918677],
                               [0.38752654,  0.62144744, -0.00897398],
                               [-0.01584150, -0.03412294, 1.04996444]])
        
        RGBw = M_CAT16 @ XYZw
    
        D_pre = F * (1 - (1/3.6) * np.exp((-La - 42) / 92))
        D = np.clip(D_pre, 0, 1)
    
        Dr = D * (XYZw[1] / RGBw[0]) + 1 - D
        Dg = D * (XYZw[1] / RGBw[1]) + 1 - D
        Db = D * (XYZw[1] / RGBw[2]) + 1 - D
    
        k = 1 / (5 * La + 1)
        Fl = 0.2 * (k**4) * (5 * La) + 0.1 * ((1 - k**4)**2) * (5 * La)**(1/3)
        n = Yb / XYZw[1]
        z = 1.48 + np.sqrt(n)
        Nbb = 0.725 * (1 / n)**0.2
        Ncb = Nbb
    
        RGBwc = np.array([Dr * RGBw[0], Dg * RGBw[1], Db * RGBw[2]])
    
        RGBaw = (400 * (Fl * RGBwc / 100)**0.42) / (27.13 + (Fl * RGBwc / 100)**0.42) + 0.1
        Aw = (2 * RGBaw[0] + RGBaw[1] + RGBaw[2] / 20 - 0.305) * Nbb
    
        J = 6.25 * ((c * QMh[:, 0] / ((Aw + 4) * Fl**0.25))**2)
        C = QMh[:, 1] / (Fl**0.25)
        h = QMh[:, 2]
    
        # Step 3
        t = (C / ((J / 100)**0.5 * (1.64 - 0.29**n)**0.73))**(1/0.9)
        et = (np.cos(np.deg2rad(h) + 2) + 3.8) / 4
        A = Aw * (J / 100)**(1 / (c * z))
    
        p1 = (50000 / 13) * Nc * Ncb * et / t
        p2 = A / Nbb + 0.305
        p3 = 21 / 20
    
        a = np.zeros_like(J)
        b = np.zeros_like(J)
    
        nonzero_t = t != 0
        if np.any(nonzero_t):
            at = np.cos(np.deg2rad(h))
            bt = np.sin(np.deg2rad(h))
            p4 = p1 / bt
            p5 = p1 / at
    
            mask_p = np.abs(bt) >= np.abs(at)
            mask_q = ~mask_p
    
            b[mask_p] = p2[mask_p] * (2 + p3) * (460/1403) / (
                p4[mask_p] + (2 + p3) * (220/1403) * (at[mask_p] / bt[mask_p]) - (27/1403) + p3 * (6300/1403)
            )
            a[mask_p] = b[mask_p] * (at[mask_p] / bt[mask_p])
    
            a[mask_q] = p2[mask_q] * (2 + p3) * (460/1403) / (
                p5[mask_q] + (2 + p3) * (220/1403) - ((27/1403) - p3 * (6300/1403)) * (bt[mask_q] / at[mask_q])
            )
            b[mask_q] = a[mask_q] * (bt[mask_q] / at[mask_q])
    
        # Step 5
        Rpa = (460 * p2 + 451 * a + 288 * b) / 1403
        Gpa = (460 * p2 - 891 * a - 261 * b) / 1403
        Bpa = (460 * p2 - 220 * a - 6300 * b) / 1403
    
        def inv_nonlinear(Ppa, FL):
            val = 100 * ((27.13 * np.abs(Ppa - 0.1)) / (400 - np.abs(Ppa - 0.1)))**(1/0.42) / FL
            val[Ppa < 0.1] *= -1
            val[Ppa == 0.1] = 0
            return val
    
        Rc = inv_nonlinear(Rpa, FL)
        Gc = inv_nonlinear(Gpa, FL)
        Bc = inv_nonlinear(Bpa, FL)
    
        R = np.real(Rc / Dr)
        G = np.real(Gc / Dg)
        B = np.real(Bc / Db)
    
        XYZ = (M_CAT16inv @ np.vstack([R, G, B])).T
        XYZ[XYZ < 0] = 0
    
        return XYZ
    

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
