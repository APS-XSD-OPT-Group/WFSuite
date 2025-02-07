import os, sys
import time
import traceback

import matplotlib.pyplot as plt
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

    wavefront_sensor = create_wavefront_sensor(measurement_directory=measurement_directory)
    wavefront_analyzer = create_wavefront_analyzer(data_collection_directory=measurement_directory, energy=energy)

    try:    wavefront_sensor.restore_status()
    except: pass

    return wavefront_sensor, wavefront_analyzer


from PyQt5.QtWidgets import (QApplication, QWidget, QTextEdit, QPushButton, QMessageBox, QGridLayout)
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from aps.common.plot import gui
from aps.common.plot.image import apply_transformations

from aps.wavefront_analysis.absolute_phase.wavefront_analyzer import IMAGE_OPS
from aps.wavefront_analysis.absolute_phase.wavefront_analyzer import (KIND, DISTANCE_H, DISTANCE_V, CROP_H, CROP_V, MAGNIFICATION_H, MAGNIFICATION_V,
                                                                      REBINNING_BP, SIGMA_INTENSITY, SIGMA_PHASE, SMOOTH_INTENSITY, SMOOTH_PHASE, SCAN_BEST_FOCUS, BEST_FOCUS_FROM)
from aps.wavefront_analysis.absolute_phase.wavefront_analyzer import MODE, LINE_WIDTH, DOWN_SAMPLING, N_CORES, WINDOW_SEARCH, REBINNING

WIDTH = 500
HEIGHT = 660
class WavefrontAnalysisForm(QWidget):
    mode          = MODE
    image_ops     = str(IMAGE_OPS).replace("[", "").replace("]", "").replace("'", "")
    line_width    = LINE_WIDTH
    rebinning     = REBINNING
    down_sampling = DOWN_SAMPLING
    window_search = WINDOW_SEARCH
    n_cores       = N_CORES

    kind                   = KIND
    propagation_distance_h = DISTANCE_H
    propagation_distance_v = DISTANCE_V
    rebinning_bp           = REBINNING_BP
    sigma_intensity        = 0 if not SMOOTH_INTENSITY else SIGMA_INTENSITY
    sigma_phase            = 0 if not SMOOTH_PHASE else SIGMA_PHASE
    crop_h                 = CROP_H
    crop_v                 = CROP_V
    magnification_h        = MAGNIFICATION_H
    magnification_v        = MAGNIFICATION_V
    scan_best_focus        = 1 if SCAN_BEST_FOCUS else 0
    best_focus_from        = BEST_FOCUS_FROM

    def __init__(self,
                 working_directory=os.path.abspath(os.curdir),
                 energy=12398.0):
        super().__init__()

        self.setFixedWidth(2 * WIDTH + 10)
        self.setFixedHeight(HEIGHT + 10)

        main_box  = gui.widgetBox(self, "", orientation="horizontal", addSpace=False)
        left_box  = gui.widgetBox(main_box, "", orientation="vertical", addSpace=False)
        input_box = gui.widgetBox(left_box, "", orientation="horizontal", width=WIDTH, addSpace=False)
        right_box = gui.widgetBox(main_box, "", orientation="vertical", addSpace=False)
        image_box = gui.widgetBox(right_box, "", orientation="vertical", width=WIDTH, height=WIDTH)

        # -------------------------------------
        text_field_box_1 = gui.widgetBox(input_box, "Process Image", orientation="vertical", width=int(WIDTH / 2) - 10)

        self.le_mode          = gui.lineEdit(text_field_box_1, self, "mode", label="Mode", labelWidth=150, orientation='horizontal', controlWidth=100, valueType=str)
        self.le_image_ops     = gui.lineEdit(text_field_box_1, self, "image_ops", label="Image Ops", labelWidth=150, orientation='horizontal', controlWidth=100, valueType=str)
        self.le_line_width    = gui.lineEdit(text_field_box_1, self, "line_width", label="Line W", labelWidth=150, orientation='horizontal', controlWidth=100, valueType=int)
        self.le_rebinning     = gui.lineEdit(text_field_box_1, self, "rebinning", label="Rebin", labelWidth=150, orientation='horizontal', controlWidth=100, valueType=float)
        self.le_down_sampling = gui.lineEdit(text_field_box_1, self, "down_sampling", label="Down Samp", labelWidth=150, orientation='horizontal', controlWidth=100, valueType=float)
        self.le_window_search = gui.lineEdit(text_field_box_1, self, "window_search", label="W Search", labelWidth=150, orientation='horizontal', controlWidth=100, valueType=int)
        self.le_n_cores       = gui.lineEdit(text_field_box_1, self, "n_cores", label="N Cores", labelWidth=150, orientation='horizontal', controlWidth=100, valueType=int)

        # -------------------------------------

        text_field_box_2 = gui.widgetBox(input_box, "Back Propagation", orientation="vertical", width=int(WIDTH / 2) - 10, height=350)

        self.le_kind                   = gui.lineEdit(text_field_box_2, self, "kind", label="Kind", labelWidth=150, orientation='horizontal', controlWidth=100, valueType=str)
        self.le_propagation_distance_h = gui.lineEdit(text_field_box_2, self, "propagation_distance_h", label="Pr Dist H/2D", labelWidth=150, orientation='horizontal', controlWidth=100, valueType=float)
        self.le_propagation_distance_v = gui.lineEdit(text_field_box_2, self, "propagation_distance_v", label="Pr Dist V", labelWidth=150, orientation='horizontal', controlWidth=100, valueType=float)
        self.le_rebinning_bp           = gui.lineEdit(text_field_box_2, self, "rebinning_bp", label="Rebin", labelWidth=150, orientation='horizontal', controlWidth=100, valueType=float)
        self.le_sigma_intensity        = gui.lineEdit(text_field_box_2, self, "sigma_intensity", label="Sigma Int", labelWidth=150, orientation='horizontal', controlWidth=100, valueType=int)
        self.le_sigma_phase            = gui.lineEdit(text_field_box_2, self, "sigma_phase",    label="Sigma Ph", labelWidth=150, orientation='horizontal', controlWidth=100, valueType=int)

        self.le_crop_h                 = gui.lineEdit(text_field_box_2, self, "crop_h", label="Crop H", labelWidth=150, orientation='horizontal', controlWidth=100, valueType=int)
        self.le_crop_v                 = gui.lineEdit(text_field_box_2, self, "crop_v", label="Crop V", labelWidth=150, orientation='horizontal', controlWidth=100, valueType=int)
        self.le_magnification_h        = gui.lineEdit(text_field_box_2, self, "magnification_h", label="Mag H", labelWidth=150, orientation='horizontal', controlWidth=100, valueType=float)
        self.le_magnification_v        = gui.lineEdit(text_field_box_2, self, "magnification_v", label="Mag V", labelWidth=150, orientation='horizontal', controlWidth=100, valueType=float)
        self.cb_scan_best_focus        = gui.comboBox(text_field_box_2, self, "scan_best_focus", label="Scan Best F", labelWidth=250, orientation='horizontal', items=["no", "yes"])
        self.le_best_focus_from        = gui.lineEdit(text_field_box_2, self, "best_focus_from", label="Best F From", labelWidth=150, orientation='horizontal', controlWidth=100, valueType=str)

        # -------------------------------------

        button_box = QWidget()
        button_box.setMinimumWidth(WIDTH - 10)

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

        left_box.layout().addWidget(button_box)

        # Create text field
        self.output_data = QTextEdit()
        self.output_data.setMinimumHeight(150)
        self.output_data.setMinimumWidth(350)
        self.output_data.setReadOnly(True)

        left_box.layout().addWidget(self.output_data)

        self._result_figure = Figure(figsize=(6.0, 6.0), constrained_layout=True)
        self._result_figure_canvas = FigureCanvas(self._result_figure)
        self._toolbar = NavigationToolbar(self._result_figure_canvas, self)

        image_box.layout().addWidget(self._toolbar)
        image_box.layout().addWidget(self._result_figure_canvas)

        self.__working_directory = working_directory
        self.__energy = energy
        self.__wavefront_sensor, self.__wavefront_analyzer = initialize(working_directory, energy)

    def take_shot(self):
        try:
            try:
                self.__wavefront_sensor.collect_single_shot_image(index=1)

                image, h_coord, v_coord = self.__wavefront_sensor.get_image_stream_data(units="mm")

                h_coord, v_coord, image = apply_transformations(h_coord, v_coord, image, self.image_ops.split(sep=","))

                self.plot_image(image, h_coord, v_coord)

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

                image, h_coord, v_coord = self.__wavefront_sensor.get_image_stream_data(units="mm")

                self.__wavefront_analyzer.generate_simulated_mask(image_data=image,
                                                                  image_ops=self.image_ops.split(sep=","),
                                                                  mode=self.mode,
                                                                  line_width=int(self.line_width),
                                                                  rebinning=int(self.rebinning),
                                                                  down_sampling=float(self.down_sampling),
                                                                  window_search=int(self.window_search),
                                                                  n_cores=int(self.n_cores))

                h_coord, v_coord, image = apply_transformations(h_coord, v_coord, image, self.image_ops.split(sep=","))
                self.plot_image(image, h_coord, v_coord)

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

                image, h_coord, v_coord = self.__wavefront_sensor.get_image_stream_data(units="mm")

                self.__wavefront_analyzer.process_image(image_index=1,
                                                        image_data=image,
                                                        image_ops=self.image_ops.split(sep=","),
                                                        mode=self.mode,
                                                        line_width=int(self.line_width),
                                                        rebinning=int(self.rebinning),
                                                        down_sampling=float(self.down_sampling),
                                                        window_search=int(self.window_search),
                                                        n_cores=int(self.n_cores))

                h_coord, v_coord, image = apply_transformations(h_coord, v_coord, image, self.image_ops.split(sep=","))
                self.plot_image(image, h_coord, v_coord)

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
                                                        image_ops=self.image_ops.split(sep=","),
                                                        mode=self.mode,
                                                        line_width=int(self.line_width),
                                                        rebinning=int(self.rebinning),
                                                        down_sampling=float(self.down_sampling),
                                                        window_search=int(self.window_search),
                                                        n_cores=int(self.n_cores))

                wavefront_data = self.__wavefront_analyzer.back_propagate_wavefront(image_index=1,
                                                                                    kind=self.kind,
                                                                                    propagation_distance=float(self.propagation_distance_h),
                                                                                    propagation_distance_h=float(self.propagation_distance_h),
                                                                                    propagation_distance_v=float(self.propagation_distance_v),
                                                                                    rebinning=int(self.rebinning_bp),
                                                                                    smooth_intensity=True if int(self.sigma_intensity) > 0 else False,
                                                                                    smooth_phase=True if int(self.sigma_phase) > 0 else False,
                                                                                    sigma_intensity = int(self.sigma_intensity),
                                                                                    sigma_phase = int(self.sigma_phase),
                                                                                    crop_h=int(self.crop_h),
                                                                                    crop_v=int(self.crop_v),
                                                                                    magnification_h=float(self.magnification_h),
                                                                                    magnification_v=float(self.magnification_v),
                                                                                    scan_best_focus=self.scan_best_focus==1,
                                                                                    best_focus_from=self.best_focus_from,
                                                                                    show_figure=True,
                                                                                    save_result=True,
                                                                                    verbose=True)
                self.output_data.setText(str(wavefront_data))

                h_coord, v_coord, image = apply_transformations(h_coord, v_coord, image, self.image_ops.split(sep=","))
                self.plot_image(image, h_coord, v_coord)

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
                                                              mode=self.mode,
                                                              line_width=int(self.line_width),
                                                              rebinning=int(self.rebinning),
                                                              down_sampling=float(self.down_sampling),
                                                              window_search=int(self.window_search),
                                                              n_cores=int(self.n_cores))
        except Exception as e:
            QMessageBox.information(self, "Error", traceback.format_exc())

    def process_image(self):
        try:
            self.__wavefront_analyzer.process_image(image_index=1,
                                                    mode=self.mode,
                                                    line_width=int(self.line_width),
                                                    rebinning=int(self.rebinning),
                                                    down_sampling=float(self.down_sampling),
                                                    window_search=int(self.window_search),
                                                    n_cores=int(self.n_cores))
        except Exception as e:
            QMessageBox.information(self, "Error", traceback.format_exc())

    def back_propagate(self):
        try:
            self.output_data.clear()

            wavefront_data = self.__wavefront_analyzer.back_propagate_wavefront(image_index=1,
                                                                                kind=self.kind,
                                                                                propagation_distance=float(self.propagation_distance_h),
                                                                                propagation_distance_h=float(self.propagation_distance_h),
                                                                                propagation_distance_v=float(self.propagation_distance_v),
                                                                                rebinning=int(self.rebinning_bp),
                                                                                smooth_intensity=True if int(self.sigma_intensity) > 0 else False,
                                                                                smooth_phase=True if int(self.sigma_phase) > 0 else False,
                                                                                sigma_intensity=int(self.sigma_intensity),
                                                                                sigma_phase=int(self.sigma_phase),
                                                                                crop_h=int(self.crop_h),
                                                                                crop_v=int(self.crop_v),
                                                                                magnification_h=float(self.magnification_h),
                                                                                magnification_v=float(self.magnification_v),
                                                                                scan_best_focus=self.scan_best_focus == 1,
                                                                                best_focus_from=self.best_focus_from,
                                                                                show_figure=True,
                                                                                save_result=True,
                                                                                verbose=True)

            self.output_data.setText(str(wavefront_data))
        except Exception as e:
            QMessageBox.information(self, "Error", traceback.format_exc())

    def plot_image(self, image, hh, vv):
        data_2D = image
        fig = self._result_figure

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

