from __future__ import annotations

import numpy as np
from numpy import array as vec
from numpy import clip

from constants import MAX_PIXEL, SMALL_NUM


class NearPlane:
    def __init__(self, near: float, width: float, height: float):
        self.near = near
        self.width = width
        self.height = height

    def __str__(self):
        near_plane_str = ""
        for key, value in self.__dict__.items():
            near_plane_str += f'{key} = {value}, '
        return f'NearPlane: {near_plane_str[:-2]}'


class Image:
    def __init__(self, width: int, height: int, output_file: str):
        self.width = width
        self.height = height
        self.output_file = output_file

    def __str__(self):
        image_str = ""
        for key, value in self.__dict__.items():
            image_str += f'{key} = {value}, '
        return f'Image: {image_str[:-2]}'


class Scene:
    def __init__(
            self,
            bg_r: float, bg_g: float, bg_b: float,
            amb_ir: float, amb_ig: float, amb_ib: float
    ):
        self.bg_color = Color(bg_r, bg_g, bg_b)
        self.amb_intensity = Color(amb_ir, amb_ig, amb_ib)

    def __str__(self):
        scene_str = ""
        for key, value in self.__dict__.items():
            scene_str += f'{key} = {value}, '
        return f'Scene: {scene_str[:-2]}'


class Sphere:
    def __init__(
            self, name: str,
            x_pos: float, y_pos: float, z_pos: float,
            x_scl: float, y_scl: float, z_scl: float,
            r: float, g: float, b: float,
            k_amb: float, k_dif: float, k_spec: float, k_refl: float, n: int
    ):
        self.name = name
        self.position = vec([x_pos, y_pos, z_pos], dtype=float)
        self.scale = vec([x_scl, y_scl, z_scl], dtype=float)
        self.color = Color(r, g, b)
        self.k_amb = k_amb
        self.k_dif = k_dif
        self.k_spec = k_spec
        self.k_refl = k_refl
        self.n = n

    def __str__(self):
        sphere_str = ""
        for key, value in self.__dict__.items():
            sphere_str += f'{key} = {value}, '
        return f'Sphere: {sphere_str[:-2]}'

    def get_normal_ray(self, ray: Ray, t: float, point: vec) -> Ray:
        """
        Get normal vector of sphere at point of intersection.
        :param ray: ray object
        :param t: distance to intersection point
        :param point: point of intersection with sphere
        :return: Ray object normal to sphere at intersection point
        """
        can_pos = ray.get_canonical_position(self)
        can_dir = ray.get_canonical_direction(self)
        can_ray = Ray(can_pos, can_dir)
        n_dir = can_ray.get_point_on_ray(t)

        # Treat n_dir as a vector from the origin to the intersection point
        # Get normal vector of sphere at intersection point by inverse transforming n_dir
        m_inv_scale = self.get_inverse_scale()

        n = np.matmul(m_inv_scale, np.array([n_dir[0], n_dir[1], n_dir[2], 0]))
        n = n[:3]
        n_unit = n / np.linalg.norm(n)
        return Ray(point, n_unit)

    def get_inverse_matrix(self) -> vec:
        """
        Get inverse transformation matrix of sphere
        :return: inverse transformation matrix of sphere
        """
        m_inv_scale = self.get_inverse_scale()
        m_inv_translate = self.get_inverse_translate()
        m_inv = np.matmul(m_inv_scale, m_inv_translate)
        return m_inv

    def get_inverse_scale(self) -> vec:
        m_inv_scale = np.array([
            [1/self.scale[0], 0, 0, 0],
            [0, 1/self.scale[1], 0, 0],
            [0, 0, 1/self.scale[2], 0],
            [0, 0, 0, 1]
        ])
        return m_inv_scale

    def get_inverse_translate(self) -> vec:
        m_inv_translate = np.array([
            [1, 0, 0, -self.position[0]],
            [0, 1, 0, -self.position[1]],
            [0, 0, 1, -self.position[2]],
            [0, 0, 0, 1]
        ])
        return m_inv_translate


class Light:
    def __init__(
            self, name: str,
            x_pos: float, y_pos: float, z_pos: float,
            ir: float, ig: float, ib: float
    ):
        self.name = name
        self.position = vec([x_pos, y_pos, z_pos], dtype=float)
        self.color = Color(ir, ig, ib)

    def __str__(self):
        light_str = ""
        for key, value in self.__dict__.items():
            light_str += f'{key} = {value}, '
        return f'Light: {light_str[:-2]}'


class Ray:
    def __init__(
            self,
            position: vec = vec([0, 0, 0], dtype=float),
            direction: vec = vec([0, 0, 0], dtype=float)
    ):
        self.position = position
        self.direction = direction

    def thru_pixel(self, c: int, r: int, n_cols: int, n_rows: int, near_plane: NearPlane) -> None:
        """
        Get ray through pixel.
        :param c: pixels column
        :param r: pixels row
        :param n_cols: number of columns
        :param n_rows: number of rows
        :param near_plane: near plane object
        :return: ray through pixel
        """
        u = near_plane.width * (((2 * c) / n_cols) - 1)
        v = near_plane.height * (((2 * r) / n_rows) - 1)
        n = near_plane.near * -1

        self.position = vec([0, 0, 0], dtype=float)
        self.direction = vec([u, v, n], dtype=float)

    def get_point_on_ray(self, t: float) -> vec:
        """
        Get point on ray at distance t.
        :param t: distance from ray origin
        :return: point on ray at distance t
        """
        return self.position + (t * self.direction)

    def get_reflected_ray(self, normal: Ray, point: vec) -> Ray:
        """
        Get ray reflected about normal vector at point.
        :param normal: normal Ray
        :param point: point of reflection
        :return: reflected Ray
        """
        dir_refl = -2 * np.dot(self.direction, normal.direction) * normal.direction + self.direction
        dir_refl_unit = dir_refl / np.linalg.norm(dir_refl)
        start = point + (SMALL_NUM * dir_refl_unit)
        return Ray(start, dir_refl_unit)

    def shadow_ray(self, position: vec, light_position: vec) -> None:
        """
        Get shadow ray.
        :param position: starting position of shadow ray
        :param light_position: light source position
        """
        c = light_position - position
        c_unit = c / np.linalg.norm(c)
        position = position + SMALL_NUM * c_unit
        self.position = position
        self.direction = c_unit

    def find_intersection_spheres(self, spheres: list[Sphere]) -> tuple[float, Sphere]:
        """
        Find intersection of ray with spheres.
        :param spheres: list of Sphere objects
        :return: distance to hit and sphere hit
        """
        int_t = []
        int_s = []

        for s in spheres:
            t = self.intersects_sphere(s)
            if t > 0:
                int_t.append(t)
                int_s.append(s)

        if len(int_t) == 0:
            raise ValueError("No intersection found")
        else:
            min_t = min(int_t)
            min_s = int_s[int_t.index(min_t)]
            return min_t, min_s

    def intersects_sphere(self, sphere: Sphere) -> int:
        """
        Get intersection of ray with sphere.
        :param sphere: sphere to intersect with
        :return: t value of intersection point
        """
        # Get canonical position and direction of ray with respect to sphere
        can_pos = self.get_canonical_position(sphere)
        can_dir = self.get_canonical_direction(sphere)

        # Calculate coefficients of quadratic equation
        a = np.dot(can_dir, can_dir)
        b = np.dot(can_pos, can_dir)
        c = np.dot(can_pos, can_pos) - 1

        # Calculate two possible t values
        discriminant = (b**2) - (a * c)
        if discriminant < 0:
            return -1

        sqrt_discriminant = np.sqrt(discriminant)
        t1 = (-(b/a)) + sqrt_discriminant / a
        t2 = (-(b/a)) - sqrt_discriminant / a

        # Return smallest positive t value
        if t1 > 0 and t2 > 0:
            return min(t1, t2)
        elif t1 > 0:
            return t1
        elif t2 > 0:
            return t2
        else:
            return -1

    def get_canonical_position(self, sphere: Sphere) -> vec:
        """
        Get canonical position of ray with respect to sphere.
        :param sphere: canonical sphere
        :return: canonical position
        """
        m_inv = sphere.get_inverse_matrix()
        canonical_position = np.matmul(m_inv, np.append(self.position, 1))
        return canonical_position[:3]

    def get_canonical_direction(self, sphere: Sphere) -> vec:
        """
        Get canonical direction of ray with respect to sphere.
        :param sphere: canonical sphere
        :return: canonical direction
        """
        m_inv = sphere.get_inverse_matrix()
        canonical_direction = np.matmul(m_inv, np.append(self.direction, 0))
        return canonical_direction[:3]

    def __str__(self):
        ray_str = ""
        for key, value in self.__dict__.items():
            ray_str += f'{key} = {value}, '
        return f'Ray: {ray_str[:-2]}'


class Color:
    def __init__(self, r: float, g: float, b: float):
        """
        clamp color values to 0 and 1 inclusive
        :param r:
        :param g:
        :param b:
        """
        self.r = clip(r, 0, 1)
        self.g = clip(g, 0, 1)
        self.b = clip(b, 0, 1)

    def get_color_float(self) -> vec:
        return vec([self.r, self.g, self.b], dtype=float)

    def get_color_scaled(self) -> tuple[int, int, int]:
        return int(self.r * MAX_PIXEL), int(self.g * MAX_PIXEL), int(self.b * MAX_PIXEL)

    def __str__(self):
        color_str = ""
        for key, value in self.__dict__.items():
            color_str += f'{value:.2f}, '
        return f'({color_str[:-2]})'
