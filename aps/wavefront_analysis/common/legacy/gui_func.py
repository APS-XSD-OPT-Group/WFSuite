import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
import numpy as np

class crop_img:
    def __init__(self, data) -> None:
        self.data = data
        self.fig = plt.figure()
        self.ax1 = self.fig.add_subplot(121)
        self.ax1.imshow(data)
        self.ax2 = self.fig.add_subplot(122)
        self.ax2.imshow(np.zeros_like(data))
        plt.subplots_adjust()

    def line_select_callback(self, eclick, erelease):
        """
        Callback for line selection.

        *eclick* and *erelease* are the press and release events.
        """
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        print(f"({x1:3.2f}, {y1:3.2f}) --> ({x2:3.2f}, {y2:3.2f})")
        print(f" The buttons you used were: {eclick.button} {erelease.button}")

        self.croped = self.data[int(y1):int(y2), int(x1):int(x2)]
        self.ax2.imshow(self.croped)
        self.fig.canvas.draw_idle()

        self.corner1 = [y1, x1]
        self.corner2 = [y2, x2]
    
    def crop(self):
        _ = RectangleSelector(self.ax1, self.line_select_callback,
                              drawtype='box', useblit=True,
                              button=[1, 3],  # disable middle button
                              minspanx=5, minspany=5,
                              spancoords='pixels',
                              interactive=True)

        plt.show()




def crop_gui(data):
    data_crop = crop_img(data)
    
    data_crop.crop()
    cropped_img = data_crop.croped
    corner1 = data_crop.corner1
    corner2 = data_crop.corner2

    return cropped_img, [corner1, corner2]

