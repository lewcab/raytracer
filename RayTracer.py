import sys
from typing import TextIO
import numpy as np
from numpy import dot
from PIL import Image as PILImage
from classes import NearPlane, Image, Sphere, Light, Scene, Ray, Color
import multiprocessing as mp

# Constants
from constants import MAX_DEPTH, MAX_PIXEL


def main():
    if len(sys.argv) != 3:
        sys.exit("Usage: python ray_tracer.py <input_file> <output_file>")

    # Parse input file
    with open(sys.argv[1], 'r') as f:
        near_plane, image, scene, spheres, lights = parse_input(f, sys.argv[2], True)

    # Initialize pixels array
    n_cols = image.width
    n_rows = image.height
    pixels = np.zeros((n_rows, n_cols, 3), dtype=np.float64)

    # Prepare arguments for multiprocessing
    args = [(r, n_cols, n_rows, near_plane, spheres, lights, scene) for r in range(n_rows)]

    # Use multiprocessing to process rows in parallel
    with mp.Pool(mp.cpu_count()) as pool:
        results = pool.map(process_row, args)

    # Collect results
    for r, row_pixels in results:
        pixels[r] = row_pixels

    # Save image to file
    save_image(pixels, image)
    print(f"Image saved to {sys.argv[2]}")


def process_row(args):
    r, n_cols, n_rows, near_plane, spheres, lights, scene = args
    row_pixels = np.zeros((n_cols, 3), dtype=np.float64)
    for c in range(n_cols):
        ray = Ray()
        ray.thru_pixel(c, r, n_cols, n_rows, near_plane)
        row_pixels[c] = trace_ray(ray, 0, spheres, lights, scene)
    return r, row_pixels


def trace_ray(ray: Ray, depth: int, spheres: list, lights: list, scene: Scene) -> np.array:
    """
    Trace ray and return color in range [0, 1]
    :param ray: ray to trace
    :param depth: recursion depth
    :param spheres: list of Sphere objects
    :param lights: list of Light objects
    :param scene: Scene object
    """
    if depth > MAX_DEPTH:
        return np.array([0, 0, 0], dtype=np.float64)

    # Calculate the closest intersection with any sphere
    try:
        t_h, s = ray.find_intersection_spheres(spheres)
    except ValueError as e:
        return no_intersection(scene, depth)
    else:
        # Start calculating color for pixel
        p = ray.get_point_on_ray(t_h)
        ray_norm = s.get_normal_ray(ray, t_h, p)
        ray_refl = ray.get_reflected_ray(ray_norm, p)

        color_local = calculate_local_color(ray, ray_norm, p, s, lights, scene, spheres)
        color_reflected = trace_ray(ray_refl, depth + 1, spheres, lights, scene)

        color = color_local + color_reflected * s.k_refl

        return color


def calculate_local_color(ray: Ray, ray_norm: Ray, p: np.ndarray, s: Sphere, lights: list[Light], scene: Scene, spheres: list[Sphere]) -> np.ndarray:
    """
    Calculate local color at intersection point
    :param ray: view or reflected ray
    :param ray_norm: normal vector of ray
    :param p: intersection point
    :param s: sphere object at intersection
    :param lights: list of Light objects
    :param scene: Scene object
    :param spheres: list of Sphere objects
    :return: local color at intersection point
    """
    # Material properties
    o_c = s.color.get_color_float()     # Object color
    k_a = s.k_amb   # Ambient coefficient
    k_d = s.k_dif   # Diffuse coefficient
    k_s = s.k_spec  # Specular coefficient
    n = s.n         # Specular exponent

    # Ambient properties
    l_a = scene.amb_intensity.get_color_float()  # Ambient intensity

    # Other vectors
    view = -1 * ray.direction / np.linalg.norm(ray.direction)  # View vector
    norm = ray_norm.direction  # Normal vector

    ambient = np.array([0, 0, 0], dtype=np.float64)
    diffuse = np.array([0, 0, 0], dtype=np.float64)
    specular = np.array([0, 0, 0], dtype=np.float64)

    # Ambient component
    ambient += k_a * l_a * o_c

    for l_obj in lights:
        l_i = l_obj.color.get_color_float()     # Light intensity

        shadow_ray = Ray()
        shadow_ray.shadow_ray(p, l_obj.position)

        light = shadow_ray.direction
        refl_shadow_ray = Ray(p, -1 * light).get_reflected_ray(ray_norm, p)
        refl = refl_shadow_ray.direction

        try:
            t_h, s = shadow_ray.find_intersection_spheres(spheres)
        except ValueError as e:
            # No intersection between shadow and light source
            # Add diffuse and specular components
            n_dot_l = max(dot(norm, light), 0)
            diffuse += k_d * l_i * n_dot_l * o_c

            r_dot_v = max(dot(refl, view), 0)
            specular += k_s * l_i * (r_dot_v ** n)
        else:
            # Check if intersection is closer than light source
            if t_h > np.linalg.norm(l_obj.position - p):
                n_dot_l = max(dot(norm, light), 0)
                diffuse += k_d * l_i * n_dot_l * o_c

                r_dot_v = max(dot(refl, view), 0)
                specular += k_s * l_i * (r_dot_v ** n)

    return ambient + diffuse + specular


def no_intersection(scene: Scene, depth: int):
    """
    Handle color when no intersection is found
    :param scene: Scene object
    :param depth: recursion depth
    """
    if depth == 0:
        # Ray thru pixel
        return scene.bg_color.get_color_scaled()
    else:
        # Reflected ray
        return np.array([0, 0, 0], dtype=np.float64)


def save_image(pixels: np.ndarray, image: Image):
    """
    Save pixels to file
    :param pixels: array of pixels
    :param image: Image object
    """
    # flip pixels
    pixels = np.flip(pixels, 0)
    img = PILImage.new('RGB', (image.width, image.height))
    for r in range(image.height):
        for c in range(image.width):
            img.putpixel((c, r), (int(pixels[r][c][0] * MAX_PIXEL), int(pixels[r][c][1] * MAX_PIXEL), int(pixels[r][c][2] * MAX_PIXEL)))
    img.save(image.output_file)


def parse_input(f: TextIO, output_file: str, verbose: bool = False):
    """
    Parse input file and return NearPlane, Image, Scene, Sphere, and Light objects
    :param f: input file
    :param output_file: output file name
    :param verbose: print parsed objects
    :return:
    """
    near_plane = NearPlane(0, 0, 0)
    image = Image(0, 0, output_file)
    spheres = []
    lights = []
    scene = Scene(0, 0, 0, 0, 0, 0)

    lines = f.readlines()

    for line in lines:
        line = line.strip()
        if line == "" or line[0] == "#":
            continue

        tokens = line.split()
        if tokens[0] == "NEAR":
            near_plane = NearPlane(float(tokens[1]), float(tokens[2]), float(tokens[3]))
        elif tokens[0] == "RES":
            image = Image(int(tokens[1]), int(tokens[2]), output_file)
        elif tokens[0] == "SPHERE":
            spheres.append(
                Sphere(
                    str(tokens[1]), float(tokens[2]), float(tokens[3]),
                    float(tokens[4]), float(tokens[5]), float(tokens[6]),
                    float(tokens[7]), float(tokens[8]), float(tokens[9]),
                    float(tokens[10]), float(tokens[11]), float(tokens[12]),
                    float(tokens[13]), float(tokens[14]), int(tokens[15])
                )
            )
        elif tokens[0] == "LIGHT":
            lights.append(
                Light(
                    str(tokens[1]),
                    float(tokens[2]), float(tokens[3]), float(tokens[4]),
                    float(tokens[5]), float(tokens[6]), float(tokens[7])
                )
            )
        elif tokens[0] == "BACK":
            scene.bg_color = Color(float(tokens[1]), float(tokens[2]), float(tokens[3]))
        elif tokens[0] == "AMBIENT":
            scene.amb_intensity = Color(float(tokens[1]), float(tokens[2]), float(tokens[3]))
        else:
            sys.exit(f'Invalid input file: {line}')

    if verbose:
        print(near_plane)
        print(image)
        print(scene)
        for sphere in spheres:
            print(sphere)
        for light in lights:
            print(light)

    return near_plane, image, scene, spheres, lights


if __name__ == '__main__':
    main()
