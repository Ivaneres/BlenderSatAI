# BlenderSatAI
The goal of this project is to use computer vision classification and object detection to construct 3D building models from satellite imagery, extending the [BlenderGIS](https://github.com/domlysz/BlenderGIS) plugin with more accurate buildings out of the box.

## Roadmap
### Computer Vision
#### Object detection:
 * [ ] Research satellite imagery building detection algorithms, build an implementation that works for one country/area.
 * [ ] Attempt to infer shape of roof from satellite image.
 * [ ] Extend to be robust with any building types/styles.

#### Classification:
 * [ ] Algorithm to sort detected buildings into categories.
 * [ ] Feature-based classification - roof type, materials etc.

#### Blender:
 * [ ] Extend BlenderGIS to feed satellite data into CV algorithms.
 * [ ] Construct a few rudimentary Blender models based on building categories.
 * [ ] Place basic building models on map with correct orientation and location.
 * [ ] Improve building model construction - automated generation based on data from classification.

