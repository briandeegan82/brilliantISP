# HDR-ISP
This project is based on infiniteISP by 10X Engineering, which in turn is based on FastOpenISP and OpenISP. Really standing on the shoulders of giants here.

## Modifications
The main modifications compared to infiniteISP are:
- Added a decompanding function to linearize companded data
- Implemented Durand's HDR tone mapping algorithm
- Modified the bit depth of the ISP pipeline
- updated debug logging
- optimized execution time (ongoing)
- added extra processing options

As of now, this is very much a work in progress. Features to be added include, but are not limited to:
- HDR multicapture merge
- Lens shading correction
- Code optimizations

## Instructions
To run, execute the file "isp_pipeline.py"

The ISP parameters are defined in the config .yml files 

### more to follow




## Acknowledgments
- This project is a continuation of the work by cruxopen, fast-openISP, and infiniteISP, to name but a few.

## List of Open Source ISPs
- [infiniteISP] (https://github.com/10x-Engineers/Infinite-ISP/tree/main)
- [openISP](https://github.com/cruxopen/openISP.git)
- [Fast Open Image Signal Processor](https://github.com/QiuJueqin/fast-openISP.git)
- [AbdoKamel - simple-camera-pipeline](https://github.com/AbdoKamel/simple-camera-pipeline.git)
- [Mushfiqulalam - isp](https://github.com/mushfiqulalam/isp)
- [Karaimer - A Software Platform for Manipulating the Camera Imaging Pipeline](https://karaimer.github.io/camera-pipeline)
- [rawpy](https://github.com/letmaik/rawpy.git)
- [cruxopen/openISP](https://github.com/cruxopen/openISP.git)
