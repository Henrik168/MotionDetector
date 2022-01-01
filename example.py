import cv2
from os import path
from MotionDetector import MotionDetector
from MotionDetector.item_image import ImageItem
from CustomLogger import getLogger
from src.lib_path import make_path

dir_log = make_path("./log/")
dir_original = make_path("./images/raw/")
dir_overlay = make_path("./images/overlay/")
dir_video = make_path("./Videos/")

url = "url to stream"

logger = getLogger(name="MotionDetector", level=20, log_path=dir_log)


def log_start_motion(image_item: ImageItem) -> None:
    logger.info(f"Motion detected with "
                f"min Area: '{image_item.min_area_ratio}', "
                f"max Area: '{image_item.max_area_ratio}', "
                f"sum Area: '{image_item.sum_area_ratio}' ")


def log_end_motion(image_item: ImageItem) -> None:
    logger.info("Motion ended.")


def save_image_overlay(image_item: ImageItem) -> None:
    file_path = path.join(dir_overlay, f"{image_item.timestamp}.jpg")
    with open(file_path, "wb") as image:
        image.write(image_item.binary_overlay)
    logger.info(f"Wrote image to path: {file_path}")


def save_image_original(image_item: ImageItem) -> None:
    file_path = path.join(dir_original, f"{image_item.timestamp}.jpg")
    with open(file_path, "wb") as image:
        image.write(image_item.binary_image)
    logger.info(f"Wrote image to path: {file_path}")


def save_video(image_item: ImageItem) -> None:
    file_path = path.join(dir_video, f"{image_item.timestamp}.avi")
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    fps = 0.4
    resolution = image_item.resolution
    writer = cv2.VideoWriter(file_path, fourcc, fps, resolution)
    for frame in image_item.frames:
        _ = writer.write(frame)
    writer.release()
    logger.info(f"Wrote Video to path: {file_path}")


def main(debug: bool) -> None:
    # Create Motion Detector Object
    motion_detector = MotionDetector(url=url,
                                     logger=logger)
    # Register Functions to Eventhandler
    motion_detector.motion_start_handler.subscribe(log_start_motion)
    motion_detector.motion_start_handler.subscribe(save_image_overlay)
    motion_detector.motion_start_handler.subscribe(save_image_original)
    motion_detector.motion_end_handler.subscribe(log_end_motion)
    motion_detector.motion_end_handler.subscribe(save_video)
    # start Motion Detector
    motion_detector.start()

    while True:
        try:
            if motion_detector.get_exception():
                raise Exception(motion_detector.get_exception())

            if debug:
                cv2.imshow("Motion-detector", motion_detector.get_frame())

            key = cv2.waitKey(100)
            if key == 27:
                raise KeyboardInterrupt
            elif key == -1:
                pass
            else:
                logger.info("Press ESC to exit the program.")

        except KeyboardInterrupt:
            logger.error("Keyboard Interrupt")
            break
        except Exception as e:
            logger.exception(e)
            break
        finally:
            pass
    if debug:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main(debug=True)
