"""Skeleton code for python script to process a video using OpenCV package

:copyright: (c) 2021, Joeri Nicolaes
:license: BSD license
"""
import argparse
import cv2
import sys
import numpy as np
import time

ddepth = cv2.CV_16S
kernel_size_gaussian = 5
kernel_size_laplacian = 3
sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])


# helper function to change what you do based on video seconds
def between(cap, lower: int, upper: int) -> bool:
    return lower <= int(cap.get(cv2.CAP_PROP_POS_MSEC)) < upper


def sobel(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    scale = 1
    delta = 0
    grad_x = cv2.Sobel(gray, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    grad_y = cv2.Sobel(gray, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    return cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)


def canny(frame, threshold1, threshold2, apertureSize):
    frame = frame.astype(np.uint8)
    return cv2.Canny(frame, threshold1, threshold2, apertureSize)


def add_text(frame, text, x=50, y=50, scale=1, thickness=2):
    font = cv2.FONT_HERSHEY_SIMPLEX  # Creates a font
    return cv2.putText(frame, text, (x, y), font, scale, 255, thickness)  # Draw the text


def blur(frame):
    return cv2.blur(frame, (10, 10))


def DFT(frame, rows, cols):
    m = cv2.getOptimalDFTSize(rows)
    n = cv2.getOptimalDFTSize(cols)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    padded = cv2.copyMakeBorder(gray, 0, m - rows, 0, n - cols, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    planes = [np.float32(padded), np.zeros(padded.shape, np.float32)]
    complexI = cv2.merge(planes)  # Add to the expanded another plane with zeros
    cv2.dft(complexI, complexI)  # this way the result may fit in the source matrix
    cv2.split(complexI, planes)  # planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
    cv2.magnitude(planes[0], planes[1], planes[0])  # planes[0] = magnitude
    magI = planes[0]
    matOfOnes = np.ones(magI.shape, dtype=magI.dtype)
    cv2.add(matOfOnes, magI, magI)  # switch to logarithmic scale
    cv2.log(magI, magI)
    magI_rows, magI_cols = magI.shape
    # crop the spectrum, if it has an odd number of rows or columns
    magI = magI[0:(magI_rows & -2), 0:(magI_cols & -2)]
    cx = int(magI_rows / 2)
    cy = int(magI_cols / 2)
    q0 = magI[0:cx, 0:cy]  # Top-Left - Create a ROI per quadrant
    q1 = magI[cx:cx + cx, 0:cy]  # Top-Right
    q2 = magI[0:cx, cy:cy + cy]  # Bottom-Left
    q3 = magI[cx:cx + cx, cy:cy + cy]  # Bottom-Right
    tmp = np.copy(q0)  # swap quadrants (Top-Left with Bottom-Right)
    magI[0:cx, 0:cy] = q3
    magI[cx:cx + cx, cy:cy + cy] = tmp
    tmp = np.copy(q1)  # swap quadrant (Top-Right with Bottom-Left)
    magI[cx:cx + cx, 0:cy] = q2
    magI[0:cx, cy:cy + cy] = tmp
    # frame = cv2.normalize(magI, magI, 0, 1, cv2.NORM_MINMAX)  # Transform the matrix with float values into a
    # return frame
    return cv2.normalize(magI, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)


def iDFT(frame, rows, cols):
    crow, ccol = rows // 2, cols // 2
    # create a mask first, center square is 1, remaining all zeros
    mask = np.zeros((rows, cols, 2), np.uint8)
    mask[crow - 30:crow + 30, ccol - 30:ccol + 30] = 1
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # apply mask and inverse DFT
    dft = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    fshift = dft_shift * mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
    return cv2.normalize(img_back, img_back, 0, 1, cv2.NORM_MINMAX)


def gaussian_low_pass_filter(frame, kernel_size):
    frame = cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)
    return frame


def laplacian_high_pass_filter(frame, kernel_size):
    frame = cv2.Laplacian(frame, ddepth, ksize=kernel_size)
    return cv2.convertScaleAbs(frame)


def band_pass_filter(frame, kernel_size_low, kernel_size_high):
    gaussian_frame = gaussian_low_pass_filter(frame, kernel_size_low)
    final_frame = laplacian_high_pass_filter(gaussian_frame, kernel_size_high)
    return cv2.convertScaleAbs(final_frame)


def optical_flow(old_frame, new_frame):
    # params for ShiTomasi corner detection
    feature_params = dict(maxCorners=100,
                          qualityLevel=0.3,
                          minDistance=7,
                          blockSize=7)

    # Parameters for lucas kanade optical flow
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    color = np.random.randint(0, 255, (100, 3))

    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
    mask = np.zeros_like(old_frame)

    frame_gray = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)
    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    # select good points
    if p1 is not None:
        good_new = p1[st == 1]
        good_old = p0[st == 1]
    # draw the tracks
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
        new_frame = cv2.circle(new_frame, (int(a), int(b)), 5, color[i].tolist(), -1)
    img = cv2.add(new_frame, mask)
    return img


def main(input_video_file: str, output_video_file: str) -> None:
    # OpenCV video objects to work with
    cap = cv2.VideoCapture(input_video_file)
    fps = int(round(cap.get(5)))
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    print("Video FPS: {}, width: {}, height: {}".format(fps, frame_width, frame_height))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # note the lower case
    out = cv2.VideoWriter(output_video_file, fourcc, fps, (frame_height, frame_width), True)
    first_frame = None
    # while loop where the real work happens
    while cap.isOpened():
        ret, frame = cap.read()
        # print(frame.shape)
        if ret:
            if cv2.waitKey(28) & 0xFF == ord('q'):
                break
            # frame = cv2.resize(frame, (1280, 720))
            if between(cap, 0, 500):
                frame = add_text(frame, "Part 1")
            if between(cap, 500, 2500):
                frame = cv2.blur(frame, (10, 10))
                frame = add_text(frame, "Blur")
            if between(cap, 2500, 5000):
                frame = cv2.filter2D(frame, -1, sharpen_kernel)
                frame = add_text(frame, "Sharpen")
            if between(cap, 5000, 7500):
                frame = sobel(frame)
                frame = add_text(frame, "Sobel")
            if between(cap, 7500, 10000):
                canny_param = [255, 255 / 3, 3]
                frame = canny(frame, canny_param[0], canny_param[1], canny_param[2])
                frame = add_text(frame, "Canny, threshold 1: {}, threshold 2: {}, aperture size: {}".
                                 format(canny_param[0], canny_param[1], canny_param[2]))
            if between(cap, 10000, 12500):
                frame = DFT(frame, frame_height, frame_width)
                frame = add_text(frame, "Part 2: DFT Spectrum")
            if between(cap, 12500, 15000):
                frame = gaussian_low_pass_filter(frame, kernel_size_gaussian)
                frame = iDFT(frame, frame_height, frame_width)
                frame = add_text(frame, "Gaussian Low Pass Filter with kernel size {}".
                                 format(kernel_size_gaussian))
            if between(cap, 15000, 17500):
                frame = laplacian_high_pass_filter(frame, kernel_size_laplacian)
                frame = iDFT(frame, frame_height, frame_width)
                frame = add_text(frame, "Laplacian High Pass Filter with kernel size {}".
                                 format(kernel_size_laplacian))
            if between(cap, 17500, 20000):
                frame = band_pass_filter(frame, kernel_size_gaussian, kernel_size_laplacian)
                frame = iDFT(frame, frame_height, frame_width)
                frame = add_text(frame, "Band Pass Filter (Gaussian and Laplacian) with kernel size {} and {}".
                                 format(kernel_size_gaussian, kernel_size_laplacian))
            if between(cap, 20000, 29000):
                frame_sharpen = cv2.filter2D(frame, -1, sharpen_kernel)
                frame_gray = cv2.cvtColor(frame_sharpen, cv2.COLOR_BGR2GRAY)
                method = eval('cv2.TM_CCOEFF')
                gray_template = cv2.imread('template.png', cv2.IMREAD_GRAYSCALE)
                w, h = gray_template.shape[::-1]
                res = cv2.matchTemplate(frame_gray, gray_template, method)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                top_left = max_loc
                bottom_right = (top_left[0] + w, top_left[1] + h)
                frame = cv2.rectangle(frame, top_left, bottom_right, 255, 2)
                frame = add_text(frame, "Part 3: Template Matching")
            if between(cap, 29000, 30000):
                frame = add_text(frame, "Up Next: Optical Flow")
            if between(cap, 30000, 40000):
                if first_frame is None:
                    first_frame = frame
                    continue
                else:
                    frame = optical_flow(first_frame, frame)
                    first_frame = frame
            if between(cap, 40000, 42000):
                frame = add_text(frame, "Part 4: Freestyle. Get Ready!")
            if between(cap, 42000, 55000):
                if first_frame is None:
                    first_frame = frame
                    continue
                else:
                    frame = cv2.absdiff(first_frame, frame)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    frame = cv2.threshold(frame, 25, 255, cv2.THRESH_BINARY)[1]
                    frame = cv2.dilate(frame, None, iterations=2)
                    contours, hierarchy = cv2.findContours(frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    for c in contours:
                        if cv2.contourArea(c) < 500:
                            continue
                        (x, y, w, h) = cv2.boundingRect(c)
                        frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            if between(cap, 55000, 60000):
                frame = add_text(frame, "The end. Thank you for watching!", 960, 200, 2, 4)
            # write frame that you processed to output
            # print(frame)
            # frame = cv2.normalize(frame, frame, 0, 255, cv2.NORM_MINMAX)
            print(cap, frame.shape)
            # frame = np.uint8(frame)
            out.write(frame)

            # (optional) display the resulting frame
            cv2.imshow('Frame', frame)

            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        # Break the loop
        else:
            break

    # When everything done, release the video capture and writing object

    cap.release()
    out.release()

    # Closes all the frames
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OpenCV video processing')
    parser.add_argument('-i', "--input", help='full path to input video that will be processed')
    parser.add_argument('-o', "--output", help='full path for saving processed video output')
    args = parser.parse_args()

    if args.input is None or args.output is None:
        sys.exit("Please provide path to input and output video files! See --help")

    main(args.input, args.output)
