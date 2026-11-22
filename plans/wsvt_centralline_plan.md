# Plan: WSVT absolute-phase centralLine mode

## Goal

Add a `centralLine` reconstruction mode to the WSVT absolute-phase path, mirroring
the existing WXST behavior, so users can trade 2D area reconstruction for a much
faster 1D (vertical + horizontal central-strip) reconstruction when only line
profiles matter.

- `mode == 'area'`    : current behavior (full 2D WSVT solve). Unchanged default.
- `mode == 'centralLine'`: run the WSVT solver on two thin central strips only
  (a vertical strip for displace_y, a horizontal strip for displace_x), then build
  1D line profiles. Much faster because the solver cost scales with the number of
  pixels, and each strip is `~block_width` wide instead of the full crop.

## Background: how WXST does it (reference implementation)

`process_images_executor.py`, `execute_process_image()`:

- `mode` + `lineWidth` args (lines 1453-1454). `block_width` in pixels =
  `int(lineWidth * pattern_size / p_x) + 2 * (window_searching + template_size/down_sampling)`.
- `area` branch (line 1805): speckle-tracks the whole crop, produces 2D
  `displace_x/y`, then extracts line profiles.
- `centralLine` branch (lines 1948-2038):
  - Crop a **vertical** block `I_*_v = I[:, cx-bw/2 : cx+bw/2]` and run
    `speckle_tracking` to get `displace_y` only.
  - Crop a **horizontal** block `I_*_h = I[cy-bw/2 : cy+bw/2, :]` and run
    `speckle_tracking` to get `displace_x` only.
  - `line_displace = [mean(displace_y, axis=1), mean(displace_x, axis=0)]` (demeaned).
  - `line_curve`, `x/y_scaling`, `avg_source_d_x/y`, `line_dpc`, `line_phase`
    (cumsum), `line_curve_filter` — all 1D.
  - `int_x/int_y` from summing flat over the strip.
  - Saves `line_*` via `save_figure_1D`, `result.json`, and `save_data` with
    `int_x/int_y` + line arrays. Returns
    `ProcessImageResult('centralLine', [int_x, int_y], None, line_phase, line_displace, line_curve_filter)`
    (note: `phase` (2D) is `None`, `intensity` holds `[int_x, int_y]`).

Key point: the centralLine speed-up comes from tracking only two thin strips
instead of the full area, and the x/y displacements are computed independently
from the horizontal/vertical strips.

### Verified: how WXST pads / trims / averages (matches the user's spec)

Confirmed by reading `speckle_tracking()` (lines 742-818) and the centralLine
branch (lines 1948-1998):

- `speckle_tracking` returns displacement at the SAME size as its input strip
  (the WXST/simple solvers do not trim the search footprint themselves). The only
  trim is `para_XST['crop_boundary']` applied afterwards (lines 812-818).
- `para_XST['crop_boundary']` (line 1524) defaults to
  `window_searching + template_size*int(1/down_sampling)` on each side when the
  GUI `Crop-Boundary == -1` (the default "use searching window").
- centralLine strip width (line 1951):
  `block_width = int(lineWidth*pattern_size/p_x) + 2*(window_searching + template_size*int(1/down_sampling))`.
- So after tracking + crop_boundary trim, the strip output width is exactly
  `linewidth_px = int(lineWidth*pattern_size/p_x)`, and `line_displace` averages
  over that width (`mean(displace_y, axis=1)`, `mean(displace_x, axis=0)`).

=> WXST does exactly: pad by the search footprint on each side, trim it off after
tracking, then average the remaining `linewidth_px`. This is the behavior the user
asked WSVT to replicate.

Difference for WSVT: the search footprint is `cal_half_window`, and the WSVT solver
ALWAYS trims exactly `cal_half_window` from each output edge
(`WSVT.py` lines 415-418) -- it is not user-overridable like WXST's crop_boundary.
Therefore WSVT's trimmed strip width is ALWAYS exactly `linewidth_px`, so the
average is always over exactly the requested line width (cleaner than WXST, where a
non-default Crop-Boundary could change the averaged width).

## How WSVT currently works (area only)

`process_images_WSVT_executor.py`, `execute_process_images_WSVT()`:

- Generates (or loads) `ref_stack` and `img_stack_cropped`, each `[n_scan, H, W]`
  cropped to `crop_edge` (= full crop).
- Runs one WSVT solver over the full stacks (lines 828-858). Solver trims
  `cal_half_window` from every output edge.
- Adds back the geometric `displace_*_offset`, applies calibration, then builds
  2D `displace_x/y`, `DPC`, `phase`, `curve`, and 1D line profiles (area-mode
  post-processing, lines 860-1037). Returns `ProcessImageResult('area', ...)`.

There is no `mode` argument in the executor and the analyzer does not pass one.

## Design for WSVT centralLine

Reuse everything up to the point where the stacks are ready; branch only at the
solver + post-processing step.

### Strip geometry (per user spec)

- `lineWidth` input: any number >= 1 (in units of pattern_size). The averaging
  width in pixels is `linewidth_px = max(1, int(lineWidth * pattern_size / p_x))`.
- The strip fed to the solver is padded by `cal_half_window` (`hw`) on EACH side so
  every pixel in the averaging region has a full search window:
  `block_width = linewidth_px + 2 * hw`.
- The WSVT solver trims exactly `hw` from each edge of its output
  (`WSVT.py` lines 415-418: `displace[..][hw:-hw, hw:-hw]`). So after the solve the
  strip output covers exactly `linewidth_px` across the width direction.
- The final 1D profile is the AVERAGE over those `linewidth_px` columns/rows.

Define, after `ref_stack` / `img_stack_cropped` are ready:

```
linewidth_px = max(1, int(lineWidth * pattern_size / p_x))
block_width  = linewidth_px + 2 * hw          # hw = cal_half_window
H, W = img_stack_cropped.shape[1], img_stack_cropped.shape[2]
block_width = min(block_width, H, W)          # clamp to crop; warn if clamped
cx = W // 2; cy = H // 2
vx0 = cx - block_width // 2; vx1 = vx0 + block_width   # vertical strip columns
hy0 = cy - block_width // 2; hy1 = hy0 + block_width   # horizontal strip rows
```

- **Vertical strip** (full height, `block_width` columns) -> displace_y:
  - `img_v = img_stack_cropped[:, :, vx0:vx1]`  (shape `[n_scan, H, block_width]`)
  - `ref_v = ref_stack[:, :, vx0:vx1]`
  - Run WSVT solver on `(img_v, ref_v)`; `displace_y = solver.displace[0]`
    (shape `[H-2hw, linewidth_px]` after the solver's own hw trim).
- **Horizontal strip** (`block_width` rows, full width) -> displace_x:
  - `img_h = img_stack_cropped[:, hy0:hy1, :]`  (shape `[n_scan, block_width, W]`)
  - `ref_h = ref_stack[:, hy0:hy1, :]`
  - Run WSVT solver on `(img_h, ref_h)`; `displace_x = solver.displace[1]`
    (shape `[linewidth_px, W-2hw]`).

Cost: two solves over `H*block_width` and `block_width*W` pixels instead of one
over `H*W`. For block_width << H,W this is a large speed-up.

### Offsets and cal_half_window cropping

The solver trims `hw` from each edge of its output. For the geometric offset
add-back, crop `displace_*_offset` to match each strip's solver output shape
(slice the same strip columns/rows, then trim `hw` on each edge):

- Vertical offset:  `displace_y_offset[:, vx0:vx1][hw:-hw, hw:-hw]`  -> `[H-2hw, linewidth_px]`.
- Horizontal offset:`displace_x_offset[hy0:hy1, :][hw:-hw, hw:-hw]` -> `[linewidth_px, W-2hw]`.
- Demean each, then `displace_y += dy_off`, `displace_x += dx_off` (as in area).

Calibration (`cali_path`) must be sliced to the same strip + `hw` trim if used.

### Averaging to 1D over the line width

```
line_displace = [np.mean(displace_y, axis=1),   # average over linewidth_px columns
                 np.mean(displace_x, axis=0)]    # average over linewidth_px rows
line_displace = [l - np.mean(l) for l in line_displace]  # demean
```

This is the explicit "average over the defined line width" step. With
`linewidth_px == 1` it degenerates to a single central line (fastest).

### 1D profiles / phase / curvature

Follow the WXST centralLine math exactly (lines 1969-1998):

```
line_displace = [mean(displace_y, axis=1), mean(displace_x, axis=0)]  # demeaned
line_curve    = grad(line_displace)/d_prop
x_scaling = 1/(1 + d_prop*mean(line_curve[1])); y_scaling similarly
line_curve *= scaling
avg_source_d_x = 1/mean(line_curve[1]); avg_source_d_y = 1/mean(line_curve[0])
line_dpc   = line_displace * p_x / d_prop * scaling
line_phase = cumsum(line_dpc) * p_x * 2*pi / c_w
line_curve_filter = gaussian_filter(line_curve, 21)
int_x/int_y from flat strip sums (reuse flat_cropped)
```

Return `ProcessImageResult('centralLine', [int_x, int_y], None, line_phase,
line_displace, line_curve_filter)` and save `line_*` + `int_x/int_y` via
`save_figure_1D` / `save_data` and `result.json` — matching WXST centralLine
output exactly so back-propagation (1D) and plotting keep working.

### Area branch unchanged

Keep the current full-stack solve + 2D post-processing as the `area` branch.

## Files to modify

1. `aps/wf_suite/absolute_phase/legacy/process_images_WSVT_executor.py`
   - Add `arguments["mode"]` (default `'area'`) and ensure `lineWidth` exists.
   - After stacks are ready, branch on `args.mode`:
     - `area`: existing code path.
     - `centralLine`: strip extraction + two solver runs + 1D post-processing +
       1D saving + `ProcessImageResult('centralLine', ...)`.
   - Factor the shared solver-construction into a small helper
     `_run_wsvt(img, ref)` to avoid duplicating the ~30-line WSVT(...) call.

2. `aps/wf_suite/absolute_phase/absolute_phase_analyzer.py`
   - In `_process_images_WSVT()`, forward `mode=kwargs.get("mode", MODE)` to the
     executor (currently not passed; defaults to area).

3. `aps/wf_suite/absolute_phase/gui/absolute_phase_manager.py`
   - `_process_images_WSVT()` already funnels through the analyzer; `MODE` is set
     from the GUI via `set_ini_from_initialization_parameters`, so no change may
     be needed. Verify `mode` reaches the analyzer (it reads the `MODE` global).

4. (Optional GUI) `aps/wf_suite/absolute_phase/gui/absolute_phase_widget.py`
   - The `mode` field already exists (used by WXST). No new field required; it now
     also applies to WSVT. Optionally update any help text that says WSVT is area
     only.

No changes to the WSVT solver (`relative_metrology/legacy/WSVT.py`).

## Caching interaction

Strip extraction happens AFTER `ref_stack`/`img_stack_cropped` are built or loaded,
so the existing 3-file cache (`propagated_pattern`, `propagated_patternDet`,
`simulated_ref_stack`) is fully reusable for both modes. Switching mode does not
require regenerating the simulated mask. (The cached stacks are full-crop; strips
are sliced from them at solve time.)

## Edge cases / risks

- `block_width` must be >= `2*cal_half_window + a few` so the solver output is
  non-empty after trimming. Clamp `block_width` to the crop size and warn if too
  large (falls back to effectively area for that axis).
- Non-square strips through the WSVT solver: relative_metrology already uses
  arbitrary rectangular crops, so the solver handles `H != W`. Verify `M_image`
  usage (pass the strip's larger/min dim consistently; follow area-mode which uses
  `img_stack_cropped.shape[1]`). Need to confirm `M_image` semantics for strips.
- Two solver runs in centralLine vs one in area: still faster because each strip
  is far smaller, but for very large block_width the benefit shrinks.
- GPU path: solver sets `pyramid_level=0` on GPU; unaffected by strip shape.

## Validation

1. Run the same WSVT dataset in `area` and `centralLine`; confirm centralLine is
   significantly faster and the central `line_phase`/`line_displace` match the
   area-mode central profiles within tolerance.
2. Confirm `result.json`, `single_shot_1.hdf5` (line arrays + int_x/int_y), and
   1D figures are written in centralLine, matching WXST centralLine output keys.
3. Confirm 2D `phase` is `None` in centralLine result and the GUI plotting path
   handles it (WXST centralLine already returns `phase=None`, so the plotting
   code already tolerates this).
4. Confirm back-propagation still works from centralLine output where applicable
   (WXST centralLine + back-prop is the existing reference).

## Open questions for the user

- Should centralLine also support `d_source_recal`? (WXST recal path is tied to
  mask generation, which still happens full-area before strip slicing, so recal
  can run unchanged.)
- Any desired default for `lineWidth` specific to WSVT, or reuse the WXST ini
  value (`Reconstruction/Line-Width`)?
