# BrilliantISP / HDR-ISP

This project is a derivative work based on [Infinite-ISP](https://github.com/10x-Engineers/Infinite-ISP) by 10xEngineers, which in turn is based on FastOpenISP and OpenISP. Standing on the shoulders of giants.

**License**: Apache License 2.0 (see LICENSE and NOTICE files)

## Features

- **Decompanding (PWC)**: Linearizes companded sensor data for HDR pipelines
- **Tone mapping** (before or after demosaic):
  - `durand` – Durand bilateral-filter local TMO
  - `aces` – ACES filmic (float)
  - `aces_integer` – ACES via LUT (production-style)
  - `hable` / `hable_integer` – Hable/Uncharted 2 filmic
  - `integer` – Rational curve (Reinhard-style)
- **Lens shading correction**: Radial polynomial per-channel vignetting correction
- **Gamma correction**: Power curve or sRGB OETF (IEC 61966-2-1)
- Config-driven pipeline; HDR-aware bit depths
- Optional GPU acceleration (BNR, Durand TMO, scale, sharpen)

## Instructions

Run the pipeline:

```bash
python isp_pipeline.py
```

Parameters are defined in config YAML files (e.g. `config/svs_cam.yml`). Set `CONFIG_PATH` and `RAW_DATA` in `isp_pipeline.py` for your setup.

## Configuration

See `config/svs_cam.yml` for a full example. Key sections:

- `tone_mapping` – `tone_mapper`: durand, aces, aces_integer, hable, hable_integer
- `gamma_correction` – `curve`: gamma or srgb
- `lens_shading_correction` – radial k1/k2 per channel




## Acknowledgments

This project is a derivative work licensed under Apache License 2.0.

**Original Work**: [Infinite-ISP](https://github.com/10x-Engineers/Infinite-ISP) by 10xEngineers  
**Copyright**: 2024, 10xEngineers  
**License**: Apache License 2.0

This derivative work includes significant modifications and enhancements (see NOTICE file for details).

Additional acknowledgments:
- cruxopen for the original openISP
- fast-openISP contributors
- Various academic researchers (cited in code and documentation)

## License

Copyright 2026, Brian Deegan

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

See the NOTICE file for additional attribution requirements.

## List of Open Source ISPs
- [infiniteISP](https://github.com/10x-Engineers/Infinite-ISP/tree/main)
- [openISP](https://github.com/cruxopen/openISP.git)
- [Fast Open Image Signal Processor](https://github.com/QiuJueqin/fast-openISP.git)
- [AbdoKamel - simple-camera-pipeline](https://github.com/AbdoKamel/simple-camera-pipeline.git)
- [Mushfiqulalam - isp](https://github.com/mushfiqulalam/isp)
- [Karaimer - A Software Platform for Manipulating the Camera Imaging Pipeline](https://karaimer.github.io/camera-pipeline)
- [rawpy](https://github.com/letmaik/rawpy.git)
- [cruxopen/openISP](https://github.com/cruxopen/openISP.git)
