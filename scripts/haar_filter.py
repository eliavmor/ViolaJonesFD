import numpy as np
import pickle
import matplotlib.pyplot as plt
import PIL.Image as Image
import os
from copy import deepcopy


class Rectangle:
    def __init__(self, top_left, H, W, color):
        assert color == 0 or color == 1
        self.H = H
        self.W = W
        self.x0 = top_left[0]
        self.y0 = top_left[1]
        self.color = color

    def __eq__(self, other):
        status = self.H == other.H
        status &= self.W == other.W
        status &= self.x0 == other.x0
        status &= self.y0 == other.y0
        status &= self.color == other.color
        return status

    def __repr__(self):
        return f"rectangle height: {self.H} rectangle width: {self.W}\n x0={self.x0}, y0={self.y0} color={'white' if self.color else 'black'}\n"

    def __dict__(self):
        return {"H": self.H, "W": self.W, "x0": self.x0, "y0": self.y0, "color": self.color}

    def __getstate__(self):
        return self.__dict__()

    def __setstate__(self, state):
        self.H = state["H"]
        self.W = state["W"]
        self.x0 = state["x0"]
        self.y0 = state["y0"]
        self.color = state["color"]

    def compute(self, integral_image):
        return integral_image[self.y0, self.x0] + integral_image[self.y0 + self.H - 1, self.x0 + self.W - 1] - \
               integral_image[self.y0 + self.H - 1, self.x0] - integral_image[self.y0, self.x0 + self.W - 1]


class HaarFilter:
    def __init__(self, H, W, rectangles):
        assert H > 0
        assert W > 0
        self.H = H
        self.W = W
        self.data = np.ones((H, W))
        self.white_map = np.zeros((1, H, W))
        self.black_map = np.zeros((1, H, W))
        self.rectangles = rectangles
        for rect in rectangles:
            if rect.color == 1:
                self.white_map[0, rect.y0, rect.x0] += 1
                self.white_map[0, rect.y0 + rect.H - 1, rect.x0] -= 1
                self.white_map[0, rect.y0, rect.x0 + rect.W -1] -= 1
                self.white_map[0, rect.y0 + rect.H - 1, rect.x0 + rect.W - 1] += 1
            else:
                self.black_map[0, rect.y0, rect.x0] += 1
                self.black_map[0, rect.y0 + rect.H - 1, rect.x0] -= 1
                self.black_map[0, rect.y0, rect.x0 + rect.W -1] -= 1
                self.black_map[0, rect.y0 + rect.H - 1, rect.x0 + rect.W - 1] += 1

    def __dict__(self):
        return {"H": self.H, "W": self.W, "data": self.data, "rectangles": self.rectangles, "white_map": self.white_map,
                "black_map": self.black_map}

    def __getstate__(self):
        return self.__dict__()

    def __setstate__(self, state):
        self.H = state["H"]
        self.W = state["W"]
        self.data = state["data"]
        self.rectangles = state["rectangles"]
        self.white_map = state["white_map"]
        self.black_map = state["black_map"]

    def show(self):
        data = deepcopy(self.data) * 125
        for rect in self.rectangles:
            data[rect.y0: rect.y0 + rect.H, rect.x0: rect.x0 + rect.W] = rect.color * 255
        plt.imshow(data, cmap="gray")
        plt.show()
        plt.close('all')

    def __eq__(self, other):
        if other.H != self.H or other.W != self.W:
            return False
        if len(self.rectangles) != len(other.rectangles):
            return False
        for rect in self.rectangles:
            if rect not in other.rectangles:
                return False
        return True

    def save(self, output_path):
        data = deepcopy(self.data) * 125
        for rect in self.rectangles:
            data[rect.y0: rect.y0 + rect.H, rect.x0: rect.x0 + rect.W] = rect.color * 255
        data = Image.fromarray(data)
        data = data.convert("L")
        data.save(output_path)

    def compute_on_image(self, batch):
        result = np.sum(batch * self.white_map, axis=(1, 2)) - np.sum(batch * self.black_map, axis=(1, 2))
        return result
