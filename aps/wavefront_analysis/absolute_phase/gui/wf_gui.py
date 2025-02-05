import os, sys
import traceback

import numpy as np

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.ticker import FuncFormatter
from matplotlib.figure import Figure
from cmasher import cm as cmm

from aps.wavefront_analysis.absolute_phase.factory import create_wavefront_analyzer
from aps.wavefront_analysis.driver.factory import create_wavefront_sensor


try:
    from epics import ca
    ca.finalize_libca()
except:
    pass

def initialize(working_directory, energy):
    measurement_directory = os.path.abspath(os.path.join(working_directory, "wf_images"))

    wavefront_sensor   = create_wavefront_sensor(measurement_directory=measurement_directory)
    wavefront_analyzer = create_wavefront_analyzer(data_collection_directory=measurement_directory, energy=energy)

    try:    wavefront_sensor.restore_status()
    except: pass

    return wavefront_sensor, wavefront_analyzer

from PyQt5.QtWidgets import (QApplication, QLabel, QWidget, QVBoxLayout, QHBoxLayout,
                             QTextEdit, QPushButton, QMessageBox, QLineEdit, QGridLayout)
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from aps.wavefront_analysis.absolute_phase.wavefront_analyzer import KIND, DISTANCE_H, DISTANCE_V, CROP_H, CROP_V, MAGNIFICATION_H, MAGNIFICATION_V
from aps.wavefront_analysis.absolute_phase.wavefront_analyzer import MODE, LINE_WIDTH, DOWN_SAMPLING, N_CORES, WINDOW_SEARCH, REBINNING

WIDTH = 500

class WavefrontAnalysisForm(QWidget):
    def __init__(self,
                 working_directory=os.path.abspath(os.curdir),
                 energy=12398.0,
                 is_detector_rotated=False):
        super().__init__()

        def get_line(label: str, text, parent_layout) -> QLineEdit:
            box = QWidget()
            lt = QHBoxLayout()
            la = QLabel(label)
            la.setFixedWidth(90)
            le = QLineEdit()
            le.setText(str(text))
            lt.addWidget(la)
            lt.addWidget(le)
            box.setLayout(lt)
            parent_layout.addWidget(box)
            return le

        self.setFixedWidth(2*WIDTH + 10)

        main_layout = QHBoxLayout()

        left_box = QWidget()
        right_box = QWidget()

        layout_left = QVBoxLayout()
        layout_right = QVBoxLayout()

        input_box = QWidget()
        input_box.setMinimumWidth(WIDTH)

        layout_ib = QHBoxLayout()

        # -------------------------------------
        text_field_box_1 = QWidget()
        text_field_box_1.setMinimumWidth(int(WIDTH/2)-5)
        layout_1 = QVBoxLayout()

        self.mode          = get_line("Mode", MODE, layout_1)
        self.line_width    = get_line("Line W", LINE_WIDTH, layout_1)
        self.rebinning     = get_line("Rebin", REBINNING, layout_1)
        self.down_sampling = get_line("Down Samp", DOWN_SAMPLING, layout_1)
        self.window_search = get_line("W Search", WINDOW_SEARCH, layout_1)
        self.n_cores       = get_line("N Cores", N_CORES, layout_1)

        text_field_box_1.setLayout(layout_1)

        # -------------------------------------

        text_field_box_2 = QWidget()
        text_field_box_2.setMinimumWidth(int(WIDTH / 2) - 5)
        text_field_box_2.setFixedHeight(400)
        layout_2 = QVBoxLayout()

        self.kind                   = get_line("Kind", KIND, layout_2)
        self.propagation_distance_h = get_line("Pr Dist H/2D", DISTANCE_H, layout_2)
        self.propagation_distance_v = get_line("Pr Dist V", DISTANCE_V, layout_2)
        self.crop_h                 = get_line("Crop H", CROP_H, layout_2)
        self.crop_v                 = get_line("Crop V", CROP_V, layout_2)
        self.magnification_h        = get_line("Mag H", MAGNIFICATION_H, layout_2)
        self.magnification_v        = get_line("Mag V", MAGNIFICATION_V, layout_2)

        text_field_box_2.setLayout(layout_2)

        # -------------------------------------

        layout_ib.addWidget(text_field_box_1)
        layout_ib.addWidget(text_field_box_2)

        input_box.setLayout(layout_ib)

        layout_left.addWidget(input_box)

        # -------------------------------------

        button_box = QWidget()
        button_box.setMinimumWidth(WIDTH-10)

        button_grid = QGridLayout()

        # Add 8 buttons in a 2x4 grid
        button = QPushButton("Take Shot\n Only", self)
        button.clicked.connect(self.take_shot)
        button_grid.addWidget(button, 0, 0)

        button = QPushButton("Take Shot +\nGenerate Mask", self)
        button.clicked.connect(self.take_shot_and_generate_mask)
        button_grid.addWidget(button, 0, 1)

        button = QPushButton("Take Shot +\nProcess Image", self)
        button.clicked.connect(self.take_shot_and_process_image)
        button_grid.addWidget(button, 0, 2)

        button = QPushButton("Take Shot +\nBack Propagate", self)
        button.clicked.connect(self.take_shot_and_back_propagate)
        button_grid.addWidget(button, 0, 3)

        button_grid.addWidget(QWidget(), 1, 0)

        button = QPushButton("Generate Mask\nFrom File")
        button.clicked.connect(self.generate_mask)
        button_grid.addWidget(button, 1, 1)

        button = QPushButton("Process Image\nFrom File")
        button.clicked.connect(self.process_image)
        button_grid.addWidget(button, 1, 2)

        button = QPushButton("Back Propagate\nFrom File")
        button.clicked.connect(self.back_propagate)
        button_grid.addWidget(button, 1, 3)

        button_box.setLayout(button_grid)

        layout_left.addWidget(button_box)

        # Create text field
        self.output_data = QTextEdit()
        self.output_data.setMinimumHeight(150)
        self.output_data.setMinimumWidth(350)
        self.output_data.setReadOnly(True)

        layout_left.addWidget(self.output_data)

        image_box = QWidget()
        image_box.setFixedHeight(WIDTH)
        image_box.setFixedWidth(WIDTH)
        layout_box = QVBoxLayout()

        self._result_figure        = Figure(figsize=(6.0, 6.0), constrained_layout=True)
        self._result_figure_canvas = FigureCanvas(self._result_figure)

        self._toolbar = NavigationToolbar(self._result_figure_canvas, self)
        layout_box.addWidget(self._toolbar)

        layout_box.addWidget(self._result_figure_canvas)
        image_box.setLayout(layout_box)

        layout_right.addWidget(image_box)

        left_box.setLayout(layout_left)
        right_box.setLayout(layout_right)

        main_layout.addWidget(left_box)
        main_layout.addWidget(right_box)

        # Set the layout for the widget
        self.setLayout(main_layout)

        self.__wavefront_sensor, self.__wavefront_analyzer = initialize(working_directory, energy)
        self.__is_detector_rotated = is_detector_rotated

    def take_shot(self):
        try:
            try:
                self.__wavefront_sensor.collect_single_shot_image(index=1)

                image, h_coord, v_coord = self.__wavefront_sensor.get_image_stream_data(units="um")

                if self.__is_detector_rotated: self.plot_image(image.T, v_coord, h_coord)
                else:                     self.plot_image(image, h_coord, v_coord)

                try:    self.__wavefront_sensor.save_status()
                except: pass
                try:    self.__wavefront_sensor.end_collection()
                except: pass

            except Exception:
                try:    self.__wavefront_sensor.save_status()
                except: pass
                try:    self.__wavefront_sensor.end_collection()
                except: pass

                QMessageBox.information(self, "Error", traceback.format_exc())

        except Exception:
            QMessageBox.information(self, "Error", traceback.format_exc())

    def take_shot_and_generate_mask(self):
        try:
            try:
                self.__wavefront_sensor.collect_single_shot_image(index=1)

                image, h_coord, v_coord = self.__wavefront_sensor.get_image_stream_data(units="um")

                if self.__is_detector_rotated: self.plot_image(image.T, v_coord, h_coord)
                else:                     self.plot_image(image, h_coord, v_coord)

                self.__wavefront_analyzer.generate_simulated_mask(image_data=image,
                                                                  mode = self.mode.text(),
                                                                  line_width = int(self.line_width.text()),
                                                                  rebinning = int(self.rebinning.text()),
                                                                  down_sampling = float(self.down_sampling.text()),
                                                                  window_search = int(self.window_search.text()),
                                                                  n_cores = int(self.n_cores.text()))

                try:    self.__wavefront_sensor.save_status()
                except: pass
                try:    self.__wavefront_sensor.end_collection()
                except: pass

            except Exception as e:

                try:    self.__wavefront_sensor.save_status()
                except: pass
                try:    self.__wavefront_sensor.end_collection()
                except: pass

                QMessageBox.information(self, "Error", traceback.format_exc())
        except Exception as e:
            QMessageBox.information(self, "Error", traceback.format_exc())

    def take_shot_and_process_image(self):
        try:
            self.output_data.clear()

            try:
                self.__wavefront_sensor.collect_single_shot_image(index=1)

                image, h_coord, v_coord = self.__wavefront_sensor.get_image_stream_data(units="um")

                self.__wavefront_analyzer.process_image(image_index=1,
                                                        image_data=image,
                                                        mode=self.mode.text(),
                                                        line_width=int(self.line_width.text()),
                                                        rebinning=int(self.rebinning.text()),
                                                        down_sampling=float(self.down_sampling.text()),
                                                        window_search=int(self.window_search.text()),
                                                        n_cores=int(self.n_cores.text()))

                if self.__is_detector_rotated: self.plot_image(image.T, v_coord, h_coord)
                else:                     self.plot_image(image, h_coord, v_coord)

                try:    self.__wavefront_sensor.save_status()
                except: pass
                try:    self.__wavefront_sensor.end_collection()
                except: pass
            except Exception as e:

                try:    self.__wavefront_sensor.save_status()
                except: pass
                try:    self.__wavefront_sensor.end_collection()
                except: pass

                QMessageBox.information(self, "Error", traceback.format_exc())
        except Exception as e:
            QMessageBox.information(self, "Error", traceback.format_exc())

    def take_shot_and_back_propagate(self):
        try:
            self.output_data.clear()

            try:
                self.__wavefront_sensor.collect_single_shot_image(index=1)

                image, h_coord, v_coord = self.__wavefront_sensor.get_image_stream_data(units="um")

                self.__wavefront_analyzer.process_image(image_index=1,
                                                        image_data=image,
                                                        mode=self.mode.text(),
                                                        line_width=int(self.line_width.text()),
                                                        rebinning=int(self.rebinning.text()),
                                                        down_sampling=float(self.down_sampling.text()),
                                                        window_search=int(self.window_search.text()),
                                                        n_cores=int(self.n_cores.text()))

                wavefront_data = self.__wavefront_analyzer.back_propagate_wavefront(image_index=1,
                                                                                    kind=self.kind.text(),
                                                                                    propagation_distance=float(self.propagation_distance_h.text()),
                                                                                    propagation_distance_h=float(self.propagation_distance_h.text()),
                                                                                    propagation_distance_v=float(self.propagation_distance_v.text()),
                                                                                    crop_h=int(self.crop_h.text()),
                                                                                    crop_v=int(self.crop_v.text()),
                                                                                    magnification_h=float(self.magnification_h.text()),
                                                                                    magnification_v=float(self.magnification_v.text()),
                                                                                    scan_best_focus=False,
                                                                                    show_figure=True,
                                                                                    save_result=True,
                                                                                    verbose=True)
                self.output_data.setText(str(wavefront_data))

                if self.__is_detector_rotated: self.plot_image(image.T, v_coord, h_coord)
                else:                     self.plot_image(image, h_coord, v_coord)

                try:    self.__wavefront_sensor.save_status()
                except: pass
                try:    self.__wavefront_sensor.end_collection()
                except: pass

            except Exception as e:

                try:    self.__wavefront_sensor.save_status()
                except: pass
                try:    self.__wavefront_sensor.end_collection()
                except: pass

                QMessageBox.information(self, "Error", traceback.format_exc())
        except Exception as e:
            QMessageBox.information(self, "Error", traceback.format_exc())

    def generate_mask(self):
        try:
            self.__wavefront_analyzer.generate_simulated_mask(image_index_for_mask=1,
                                                              mode=self.mode.text(),
                                                              line_width=int(self.line_width.text()),
                                                              rebinning=int(self.rebinning.text()),
                                                              down_sampling=float(self.down_sampling.text()),
                                                              window_search=int(self.window_search.text()),
                                                              n_cores=int(self.n_cores.text()))
        except Exception as e:
            QMessageBox.information(self, "Error", traceback.format_exc())

    def process_image(self):
        try:
            self.__wavefront_analyzer.process_image(image_index=1,
                                                    mode=self.mode.text(),
                                                    line_width=int(self.line_width.text()),
                                                    rebinning=int(self.rebinning.text()),
                                                    down_sampling=float(self.down_sampling.text()),
                                                    window_search=int(self.window_search.text()),
                                                    n_cores=int(self.n_cores.text()))
        except Exception as e:
            QMessageBox.information(self, "Error", traceback.format_exc())

    def back_propagate(self):
        try:
            self.output_data.clear()

            wavefront_data = self.__wavefront_analyzer.back_propagate_wavefront(image_index=1,
                                                                                kind=self.kind.text(),
                                                                                propagation_distance=float(self.propagation_distance_h.text()),
                                                                                propagation_distance_h=float(self.propagation_distance_h.text()),
                                                                                propagation_distance_v=float(self.propagation_distance_v.text()),
                                                                                crop_h=int(self.crop_h.text()),
                                                                                crop_v=int(self.crop_v.text()),
                                                                                magnification_h=float(self.magnification_h.text()),
                                                                                magnification_v=float(self.magnification_v.text()),
                                                                                scan_best_focus=False,
                                                                                show_figure=True,
                                                                                save_result=True,
                                                                                verbose=True)

            self.output_data.setText(str(wavefront_data))
        except Exception as e:
            QMessageBox.information(self, "Error", traceback.format_exc())

    def plot_image(self, image, hh, vv):
        data_2D = image.T
        fig     = self._result_figure

        xrange = [np.min(hh), np.max(hh)]
        yrange = [np.min(vv), np.max(vv)]

        fig.clear()
        ax_1 = fig.gca()

        def custom_formatter(x, pos): return f'{x:.2f}'

        image = ax_1.pcolormesh(hh, vv, data_2D, cmap=cmm.sunburst_r, rasterized=True)
        ax_1.set_xlim(xrange[0], xrange[1])
        ax_1.set_ylim(yrange[0], yrange[1])
        ax_1.set_xticks(np.linspace(xrange[0], xrange[1], 11, endpoint=True))
        ax_1.set_yticks(np.linspace(yrange[0], yrange[1], 11, endpoint=True))
        ax_1.xaxis.set_major_formatter(FuncFormatter(custom_formatter))
        ax_1.yaxis.set_major_formatter(FuncFormatter(custom_formatter))
        ax_1.axhline(0, color="gray", ls="--", linewidth=1, alpha=0.7)
        ax_1.axvline(0, color="gray", ls="--", linewidth=1, alpha=0.7)
        ax_1.set_xlabel("Horizontal ($\mu$m)")
        ax_1.set_ylabel("Vertical ($\mu$m)")

        cbar = fig.colorbar(mappable=image, ax=ax_1, pad=0.01, aspect=30, shrink=0.6)
        cbar.ax.text(0.5, 1.05, "pI", transform=cbar.ax.transAxes, ha="center", va="bottom", fontsize=10, color="black")

        fig.savefig("test.png")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    form = WavefrontAnalysisForm()
    form.setWindowTitle('Wavefront Analysis')
    form.show()
    sys.exit(app.exec_())

