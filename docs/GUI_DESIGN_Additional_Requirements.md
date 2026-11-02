# Additional Requirements for GUI_DESIGN.md

## 1. Add a Session Management Section

Add a new section describing how GUI sessions are created, saved, restored, and recovered.

### Requirements

- New Session
- Save Session
- Open Session
- Auto-Recover Previous Session

### Session Contents

Each session directory should contain:

```text
tmp/gui_session/<session_id>/
    input.raw
    config.yml
    output.png
    session.json
```

Where:

- `input.raw` = working RAW copy
- `config.yml` = current ISP configuration
- `output.png` = latest preview/output image
- `session.json` = GUI-specific state not represented in YAML

### Rationale

Allows long tuning sessions to be resumed and ensures complete reproducibility of results.

---

## 2. Add an Internal Data Model Section

Add a section describing the GUI architecture and state ownership.

### Requirements

The GUI shall maintain a single shared configuration object in memory whose structure mirrors the camera YAML schema.

```text
Image Format Window
          \
           \
            --> Shared Configuration State
           /
ISP Config Window
```

On Process:

```text
Shared State
      ↓
temp config.yml
      ↓
BrilliantISP
```

### Explicitly State

The YAML schema remains the source of truth.

The GUI is an editor for that schema and should not introduce an independent configuration representation.

### Rationale

Avoids synchronization issues between windows and ensures one-to-one mapping between GUI state and pipeline configuration.

---

## 3. Add Parameter Validation Requirements

Add a new section describing validation before processing.

### Minimum Validation

```text
width > 0
height > 0

crop region within image bounds

valid Bayer pattern

valid bit depth

valid endian selection

WB gains > 0

digital gain > 0
```

### Behavior

Invalid settings should:

- Highlight offending controls
- Display clear error message
- Prevent Process execution

### Rationale

Prevents generation of invalid YAML files and pipeline failures.

---

## 4. Expand Preview Pane Requirements

The preview pane should be described as more than an output image viewer.

### Required Preview Features

```text
Fit to screen
1:1 zoom
Zoom in/out
Pan
Refresh preview
```

### Future Enhancements

```text
Before/After comparison
Split-slider comparison
Blink comparison

RGB histogram
Luma histogram

Pixel inspector
ROI statistics
```

### Rationale

These are standard ISP tuning workflows and should influence the layout design now even if implementation comes later.

---

## 5. Add Change Tracking Requirements

Describe how modified parameters are visually identified.

### Requirements

Parameters modified from the loaded preset should be highlighted.

Example:

```text
R Gain = 1.95 *
G Gain = 1.00
B Gain = 2.12 *
```

Where:

- `*` indicates modified value
- Modified controls may also use highlighting

### Additional Actions

```text
Show Modified Only
Reset Parameter
Reset Module
Reset All
```

### Rationale

Engineers need to know what tuning changes produced the current result.

---

## 6. Add Computed Read-Only Fields

Expand "Effective Width/Height" into a dedicated section.

### Computed Fields

```text
Effective Width
Effective Height

Active Pixel Count

Crop Percentage

Estimated RAW Size

Output Resolution
```

### Behavior

Values update automatically whenever related parameters change.

### Rationale

Provides immediate feedback when changing format parameters.

---

## 7. Add YAML Compatibility Notes

Add a dedicated subsection documenting schema inconsistencies.

### Examples

```text
filter_window vs filt_window

gamma LUT implementations
vs
curve/gamma implementations

algorithm-document names
vs
actual YAML names
```

### Requirements

The GUI must serialize using the real YAML schema supported by BrilliantISP.

### Rationale

Avoids ambiguity between documentation terminology and implementation terminology.

---

## 8. Add Future Processing Architecture Notes

Document potential future processing modes.

### Possible Future Modes

```text
Run Full Pipeline

Run From Demosaic

Run From CCM

Run From Sharpen

Run Selected Module
```

### Note

These are future enhancements and are not part of the initial implementation.

### Rationale

Prevents future GUI redesign when incremental processing is added.

---

# Highest-Priority Additions

If only a subset of enhancements is adopted, prioritize the following:

1. Session Management
2. Shared Configuration State / Internal Data Model
3. Parameter Validation
4. Enhanced Preview Pane Design

These additions will have the greatest impact on long-term usability, maintainability, and extensibility of the GUI design.
