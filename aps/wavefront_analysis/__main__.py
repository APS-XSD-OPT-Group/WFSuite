import sys

from aps.wavefront_analysis.launcher.main_launcher import MainLauncher
from aps.wavefront_analysis.absolute_phase.gui.main_absolute_phase import MainAbsolutePhase
from aps.wavefront_analysis.relative_metrology.gui.main_relative_metrology import MainRelativeMetrology
from aps.wavefront_analysis.driver.gui.main_wavefront_sensor import MainWavefrontSensor

if __name__ == "__main__":
    def show_help(error=False):
        print("")
        if error:
            print("*************************************************************")
            print("********              Command not valid!             ********")
            print("*************************************************************\n")
        else:
            print("=============================================================")
            print("           WELCOME TO APS - Wavefront Analysis Suite")
            print("=============================================================\n")
        print("To launch a script:       python -m aps.wavefront_analysis <script id> <options>\n")
        print("To show help of a script: python -m aps.wavefront_analysis <script id> --h\n")
        print("To show this help:        python -m aps.wavefront_analysis --h\n")
        print("* Available scripts:\n" +
              "    1) Main GUI,                   id: Launcher\n"
              "    2) Absolute Phase (batch),     id: Absolute-Phase\n"
              "    3) Relative Metrology (batch), id: Relative-Metrology\n"
              "")

    if len(sys.argv) == 1 or sys.argv[1] == "--h":
        show_help()
    else:
        if sys.argv[1]   == MainLauncher.SCRIPT_ID:          MainLauncher(sys_argv=sys.argv).run_script()
        elif sys.argv[1] == MainAbsolutePhase.SCRIPT_ID:     MainAbsolutePhase(sys_argv=sys.argv).run_script()
        elif sys.argv[1] == MainRelativeMetrology.SCRIPT_ID: MainRelativeMetrology(sys_argv=sys.argv).run_script()
        elif sys.argv[1] == MainWavefrontSensor.SCRIPT_ID:   MainWavefrontSensor(sys_argv=sys.argv).run_script()
