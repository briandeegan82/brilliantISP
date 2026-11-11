# r04q vehicle calibration notes

Source: the vehicle workbook `Extrinsics and Intrinsics.xlsx` (not in the repo). These notes explain how the
`config/r04q/SVS_MVL.yml` and `SVS_MVR.yml` metering ROIs were derived, and what in the workbook is usable.

## SVS intrinsics (sheet "Intrinsics", EOL station GEN6_5EOL01, all PASS)

Fisheye model, verified against the sheet's own FOV columns and against raw SVS imagery:

    r_px(theta) = c1*theta + c2*theta^2 + c3*theta^3 + c4*theta^4 + c5*theta^5   (theta in rad, full-res px)
    centre      = (960 + codXoffset_px, 768 + codYoffset_px)   on the 1920x1536 raster (subtract 4 rows for 4/4 embedded-row removal)
    linearTransform_c/d/e: small affine terms (c ~ 1.000, d/e ~ 1e-2) - not applied here

| cam | serial | c1 | c2 | c3 | c4 | c5 | codX | codY | r(90 deg) | r(99 deg) | HFOV | VFOV |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| FV  | C0D2602222900126 | 487.776 | 8.49089 | 27.2007 | -5.68792 | 0.28527  | 7.865 | 1.030  | 860.7 | ~959 | 197.6 | 163.0 |
| MVL | C0D2602225701192 | 486.817 | 8.78551 | 26.6242 | -5.81276 | 0.30173  | 8.749 | -0.861 | 857.1 | ~954 | 198.5 | 163.6 |
| MVR | C0D2602225501417 | 487.311 | 8.37916 | 27.4747 | -6.06716 | 0.319124 | 5.011 | 5.577  | 858.7 | ~956 | 198.0 | 163.3 |
| RV  | C0D2602223000095 | 484.642 | 9.15166 | 27.4774 | -6.27007 | 0.338602 | 5.470 | 2.593  | 855.4 | ~952 | 198.4 | 163.8 |

Checks: r(81.6 deg) = 764-769 px (half-height 768, VFOV/2 = 81.5-81.9 deg); r(99.4 deg) = 960-967 px (half-width 960, HFOV/2 = 98.8-99.3 deg).
On r04q frames the theta = 99 deg circle lies on the visible image-circle edge on all four cameras (`images/r04q_svs_fov_overlay.jpg`), and the r04q raw-measured FV/RV AE ellipses sit between the 90 and 99 deg circles.
Timestamps are Excel serials 44849-44850 (October 2022), i.e. EOL calibration from about 2022.

## Extrinsics (sheet "Extrinsics"): incomplete or inconsistent, do not use as-is

- The sheet says everything is relative to the vehicle body. The LiDAR rows use mm, x forward and y left (Left +606, Right -606, Rear x = -1110).
- SVS rows look like metres despite the "mm" label: FV x = -0.809, RV x = 4.003, MVR y = 1.015. MVL y = -900 looks like mm with the wrong sign for the left side.
- With x forward, FV (-0.809) behind RV (+4.003) is reversed. The SVS rows may use an x-rearward or front-bumper origin.
- All SVS yaw/pitch/roll are 0, which can't be right for side mirrors (they should be about +/-90 deg yaw and pitched down).
- AD block: columns are labelled FV/MVL/MVR/RV (the AD cameras are FL/FR/RL/RR), all values are 0 except FV z = 0.692 (copied from SVS FV). There are no AD intrinsics.
- Radar, OxTS and Ouster: radar FV has a pose; OxTS and Ouster are all zero.

## SVS_MVL / SVS_MVR metering ROI (`config/r04q/SVS_MVL.yml`, `SVS_MVR.yml`)

- Configs = `config/r04q/SVS.yml` + an identical `roi` block in `auto_white_balance` and `auto_exposure` (C3: no other keys differ; checked by dict diff).
- Ellipse = intrinsics lens circle at theta = 92 deg (same angular radius as the raw-measured r04q FV/RV ellipses), centre from codX/codY, normalised to the 1920x1528 statistics image.
- exclude_polygons = car body (hand-placed on the per-pixel temporal mean of ~560 frames from three recordings, where the static body edge is sharp) + the dark housing sliver at the outer edge.
- Active area: about 40% of the frame (MVL 0.403, MVR 0.412). No zone weights.
- 40-frame test on 6ACE6258_..._175609_000: ROI luma for MVL went from 123 to 101 and for MVR from 110-123 to 94-105 with the ROI, against FV 94-102 and RV 96 on the same recording.
- Overlay: `images/r04q_mv_roi_overlay.jpg`. Not part of the frozen KPI evaluation; r04q FV/RV/AD are unchanged.
