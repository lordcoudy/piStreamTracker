# IMX219 160-degree colour shading

`imx219-160-colour.json` preserves all algorithms from the original
`imx219-160-no-alsc.json` and adds static colour lens shading. Both the Pi 3B+
and Pi 4 configurations select it for the matching IMX219/160-degree modules.
Pi 4 was measured directly; the two-Pi hardware was not retested here.

The starting ALSC tables are Arducam's IMX219 160-degree tables for Pi 4 and
older, from its [lens-shading guide](https://docs.arducam.com/Raspberry-Pi-Camera/Native-camera/Lens-Shading-json-file/)
and [published JSON](https://www.arducam.com/wp-content/uploads/2023/08/imx219_160d.json).
The rest of that older tuning file is deliberately not imported.

The manufacturer's red correction overcompensated on the tested module. Its
red tables are raised to power 0.3169050100131657 and blue tables to
1.6648238341843424, rounded to six decimal places. These two strengths were
fitted from independent before/after measurements in six white-paper regions,
then checked in another capture with independently balanced centre WB.
This is empirical tuning under the tested warm room lighting, not a full-sensor
RAW calibration or a guarantee for every IMX219 module and lighting condition.

`n_iter: 0` disables scene-dependent adaptive colour changes.
`luminance_strength: 0` leaves brightness shading unchanged and avoids lifting
noise in dim corners. Central WB was retuned to red 1.0606 / blue 1.87502.
In the 720p/25 test, central R/G and B/G were 1.001/1.002, and measured red
ratios relative to centre stayed within 0.948–1.019 across the sampled regions.
Left-edge blue excess fell from about 84% to 39%; some cold shading remains.
Frame duration stayed 39,999 µs and capture measured 25.000 FPS.

For complete per-module correction, acquire uniformly lit, unclipped RAW/DNG
flat fields over the whole 4:3 sensor at representative colour temperatures,
then use Raspberry Pi's [camera tuning tools](https://github.com/raspberrypi/ctt)
to derive proper ALSC tables. A curved sheet with visible background and mixed
room illumination cannot establish that calibration. Run the tuning tools in
a separate environment because their dependencies differ from the tracker.

To revert, select `tuning/imx219-160-no-alsc.json` in `camera.tuning_file` and
restart the camera service. Restore the previous room-light WB gains
`[1.15644, 2.24688]`, or choose automatic WB. The original file remains unchanged.
Compare profiles in separate camera processes: changing a tuning dictionary
while reusing libcamera's camera manager can retain the previous IPA instance.
