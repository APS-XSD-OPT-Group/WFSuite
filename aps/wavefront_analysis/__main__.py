import sys
from aps.wavefront_analysis.absolute_phase.gui.wf_gui import run_gui


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
              "    1) Absolute Phase,             id: AP\n")

    if len(sys.argv) == 1 or sys.argv[1] == "--h":
        show_help()
    else:
        if   sys.argv[1] =="AP": run_gui(sys.argv)
        else: show_help(error=True)