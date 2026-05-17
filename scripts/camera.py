import cv2
from lerobot.cameras.configs import ColorMode, Cv2Rotation
from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

config = OpenCVCameraConfig(
    index_or_path=0,
    fps=15,
    width=1920,
    height=1080,
    color_mode=ColorMode.RGB,
    rotation=Cv2Rotation.NO_ROTATION,
)

WINDOW = "lerobot camera (press q to quit)"

with OpenCVCamera(config) as camera:
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, 640, 480)

    while True:
        try:
            frame = camera.async_read(timeout_ms=500)
        except TimeoutError:
            continue

        cv2.imshow(WINDOW, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            break

    cv2.destroyAllWindows()
