# :crystal_ball: Ray Tracer

![Rainbow Scene](https://github.com/lewcab/raytracer/blob/main/scenes/rainbow.png?raw=true)

This project is a  Python Ray Tracer, which can render simple scenes containing multiple light sources and spheres. These lights and spheres take on certain attributes which determine its size, position, color. 

Since the nature of each pixel is independent of others in the scene, Pythons `multiprocessing` library is used to speed up the work.

# :gear: Requirements 
- `python` version `3.9` was used to develop this project.
- `numpy` for vector and matrix math.

# :book: How to Run
1. Decide on a input file to read from. Some examples are included in the `scenes/` directory.
2. Execute the file with the following arguments:
```
python RayTracer.py <scene> <output>
```
For example:
```
python RayTracer.py scenes/sample.txt scenes/sample.png
```
3. The details of the scene will be printed out (image dimensions, spheres, lights, etc.)
4. Finally, the image will be saved to the specified directory.
