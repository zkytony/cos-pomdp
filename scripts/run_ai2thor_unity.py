from ai2thor.controller import Controller
import os
ABS_PATH = os.path.dirname(os.path.abspath(__file__))

LOAD_EXECUTABLE_PATH = os.path.join("/home/kaiyuzh/.ai2thor/releases/thor-201903131714-Linux64/thor-201903131714-Linux64")#repo/ai2thor/unity/builds/thor-Linux64-local/thor-Linux64-local")

controller = Controller(
    local_executable_path=LOAD_EXECUTABLE_PATH
)
controller.start()
