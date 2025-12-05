import cv2 as cv
import numpy as np
import os
import matplotlib.pyplot as plt

root = os.getcwd()
videoPath = os.path.join(root, "walk1.mp4")
videoCapObj = cv.VideoCapture(videoPath)
# ---------------------------------------------------------
#   SPARSE OPTICAL FLOW — LUCAS-KANADE (FEATURE TRACKING)
# ---------------------------------------------------------
def lucasKanade(videoCapObj):

    shiTomasiCornerParams = dict(maxCorners=20, qualityLevel=0.3, minDistance=50, blockSize=7)
    lucasKanadeParams = dict(winSize=(15, 15), maxLevel=2,
                             criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

    randomColors = np.random.randint(0, 255, (100, 3))

    # First frame
    ret, frameFirst = videoCapObj.read()
    if not ret:
        print("Video not found")
        return

    frameGrayPrev = cv.cvtColor(frameFirst, cv.COLOR_BGR2GRAY)
    cornersPrev = cv.goodFeaturesToTrack(frameGrayPrev, mask=None, **shiTomasiCornerParams)
    mask = np.zeros_like(frameFirst)

    # -------------------- SAVE VIDEO SETUP --------------------
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    fps = 30
    height, width, _ = frameFirst.shape
    videoWriter = cv.VideoWriter('lucas_kanade_output.mp4', fourcc, fps, (width, height))
    # -----------------------------------------------------------

    while True:
        ret, frame = videoCapObj.read()
        if not ret:
            break

        frameGrayCur = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        cornersCur, foundStatus, _ = cv.calcOpticalFlowPyrLK(
            frameGrayPrev, frameGrayCur, cornersPrev, None, **lucasKanadeParams
        )

        if cornersCur is not None:
            cornersMatchedCur = cornersCur[foundStatus == 1]
            cornersMatchedPrev = cornersPrev[foundStatus == 1]

            for i, (curCorner, prevCorner) in enumerate(zip(cornersMatchedCur, cornersMatchedPrev)):
                xCur, yCur = curCorner.ravel()
                xPrev, yPrev = prevCorner.ravel()

                cv.line(mask, (int(xCur), int(yCur)),
                        (int(xPrev), int(yPrev)),
                        randomColors[i].tolist(), 2)
                cv.circle(frame, (int(xCur), int(yCur)),
                          5, randomColors[i].tolist(), -1)

            img = cv.add(frame, mask)

            cv.imshow("Lucas-Kanade Optical Flow", img)

            # --------------- WRITE FRAME TO VIDEO ----------------
            videoWriter.write(img)
            # -----------------------------------------------------

            frameGrayPrev = frameGrayCur.copy()
            cornersPrev = cornersMatchedCur.reshape(-1, 1, 2)

        if cv.waitKey(15) & 0xFF == ord('q'):
            break

    # ----------------- RELEASE VIDEO WRITER --------------------
    videoWriter.release()
    # ------------------------------------------------------------

    videoCapObj.release()
    cv.destroyAllWindows()



# ---------------------------------------------------------
#   DENSE OPTICAL FLOW — FARNEBACK (HSV Visualization)
# ---------------------------------------------------------
def denseOpticalFlow(videoCapObj):

    # Read first frame
    ret, frameFirst = videoCapObj.read()
    if not ret:
        print("Failed to load video")
        return

    imgPrev = cv.cvtColor(frameFirst, cv.COLOR_BGR2GRAY)

    # Create HSV image for visualization
    imgHSV = np.zeros_like(frameFirst)
    imgHSV[..., 1] = 255

    # ------------------- SAVE VIDEO SETUP -------------------
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    fps = 30  # You can also use videoCapObj.get(cv.CAP_PROP_FPS)
    height, width, _ = frameFirst.shape
    videoWriter = cv.VideoWriter('dense_optical_flow.mp4', fourcc, fps, (width, height))
    # ---------------------------------------------------------

    # Process video frames
    while True:
        ret, frameCur = videoCapObj.read()
        if not ret:
            break

        imgCur = cv.cvtColor(frameCur, cv.COLOR_BGR2GRAY)

        # Compute dense optical flow
        flow = cv.calcOpticalFlowFarneback(
            prev=imgPrev,
            next=imgCur,
            flow=None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=cv.OPTFLOW_FARNEBACK_GAUSSIAN
        )

        mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])

        # Hue = direction
        imgHSV[..., 0] = ang * 180 / np.pi / 2

        # Value = magnitude
        imgHSV[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)

        # Convert HSV to BGR for viewing
        imgBGR = cv.cvtColor(imgHSV, cv.COLOR_HSV2BGR)

        # Show output
        cv.imshow("Dense Optical Flow (Farneback)", imgBGR)

        # ---------------- WRITE FRAME TO VIDEO ----------------
        videoWriter.write(imgBGR)
        # -------------------------------------------------------

        imgPrev = imgCur

        if cv.waitKey(15) & 0xFF == ord('q'):
            break

    # ---------------- RELEASE VIDEO WRITER -------------------
    videoWriter.release()
    # ---------------------------------------------------------

    cv.destroyAllWindows()



# ---------------------------------------------------------
#   MAIN
# ---------------------------------------------------------
if __name__ == "__main__":
    #lucasKanade(videoCapObj)
    denseOpticalFlow(videoCapObj)
