import numpy as np
from PIL import Image
from scipy import signal
import matplotlib.pyplot as plt

print('~ Welcome to my Coin Counter! ~ \n')


# ~~~~~~~ Functions ~~~~~~~~

def myArrayToImage(array):
    image = Image.fromarray(array)
    return image


def myImageToArray(image):
    imageToDo = Image.open(image)
    return np.array(imageToDo)


def myHoughTransformation(image, threshold, region, radius=None):
    (M, N) = image.shape
    [R_min, R_max] = radius

    R = R_max - R_min

    # Initializing accumulator array.
    # Allocating 2 times R_max to deal with overflow problems
    accumulator = np.zeros((R_max, M + 2 * R_max, N + 2 * R_max))

    # Initializing array with circles
    circles = np.zeros((R_max, M + 2 * R_max, N + 2 * R_max))

    # Angle discretization
    theta = 200
    # Take every position that represents an edge (not 0)
    edges = np.argwhere(image[:, :])

    for rad in range(R):
        r = R_min + rad

        # Creating the blueprint array
        blueprint = np.zeros((2 * (r + 1), 2 * (r + 1)))

        # The center of the blueprint
        (m, n) = (r + 1, r + 1)

        for angle in range(theta):
            x = int(np.round(r * np.cos(angle)))
            y = int(np.round(r * np.sin(angle)))
            blueprint[m + x, n + y] = 1
        constant = np.argwhere(blueprint).shape[0]

        # For every edge coordinate (possible circles)
        for x, y in edges:
            # Centering the blueprint circle over the edges
            # and updating the accumulator array
            X = [x - m + R_max, x + m + R_max]  # Computing the extreme X values
            Y = [y - n + R_max, y + n + R_max]  # Computing the extreme Y values

            accumulator[r, X[0]:X[1], Y[0]:Y[1]] += blueprint

        # Thresholding. Some kind of filtering the circles that are about to be made.
        accumulator[r][accumulator[r] < threshold * constant / r] = 0

    for r, x, y in np.argwhere(accumulator):
        tempAccum = accumulator[r - region:r + region, x - region:x + region, y - region:y + region]
        try:
            p, a, b = np.unravel_index(np.argmax(tempAccum), tempAccum.shape)
        except:
            continue
        circles[r + (p - region), x + (a - region), y + (b - region)] = 1
    # img = luminance()
    # myArrayToImage(img).show()
    return circles[:, R_max:-R_max, R_max:-R_max]


def showImageWithCircles(image):
    fig = plt.figure()
    image_start = luminance(imageNoFilter)
    plt.imshow(myArrayToImage(image_start))
    circleCoordinates = np.argwhere(image)
    circleCoordinates = filterCircles(circleCoordinates)

    twoEuro = 0
    oneEuro = 0
    fiftyCents = 0
    tenCents = 0

    circle = []
    for r, x, y in circleCoordinates:
        # 2 euros
        if r >= 49:
            circle.append(plt.Circle((y, x), r, color='red', fill=False))
            twoEuro = twoEuro + 1
        # 50 cents
        elif r >= 47:
            circle.append(plt.Circle((y, x), r, color='lime', fill=False))
            fiftyCents = fiftyCents + 1
        # 1 euro
        elif r >= 42:
            circle.append(plt.Circle((y, x), r, color='blue', fill=False))
            oneEuro = oneEuro + 1
        # 10 cents
        elif r != 0:
            circle.append(plt.Circle((y, x), r, color='fuchsia', fill=False))
            tenCents = tenCents + 1

        plt.gca().add_artist(circle[-1])

    # total = twoEuro + oneEuro + fiftyCents + tenCents
    # print('Total Coins: ', total)

    print('Found:')
    print('2-Euro: ', twoEuro)
    print('50-Cent: ', fiftyCents)
    print('1-Euro: ', oneEuro)
    print('10-Cent: ', tenCents)
    show = input('\nShow Coins? [Y/n] \n> ')
    if show == 'Y' or show == 'Υ':
        plt.show()


def filterCircles(circles):
    rows = circles.shape[0]
    cols = circles.shape[1]
    rightCircles = np.zeros((rows, cols))
    rightCirclesIndex = 0

    for k in range(0, rows):
        if circles[k, 0] != 0:
            sameCircles = np.zeros((rows, cols))
            sameCircles[k, 0] = circles[k, 0]
            sameCircles[k, 1] = circles[k, 1]
            sameCircles[k, 2] = circles[k, 2]
            circles[k, 0] = 0
            i_same = 1
            for i in range(0, rows):

                distance = np.sqrt((circles[k, 2] - circles[i, 2]) ** 2 + (circles[k, 1] - circles[i, 1]) ** 2)

                if distance <= 13:
                    if circles[i, 0] != 0:
                        sameCircles[i_same, 0] = circles[i, 0]
                        sameCircles[i_same, 1] = circles[i, 1]
                        sameCircles[i_same, 2] = circles[i, 2]
                        circles[i, 0] = 0
                        i_same = i_same + 1

            sum_radius = 0
            sum_height = 0
            sum_width = 0
            for j in range(rows):
                if sameCircles[j, 0] != 0:
                    sum_radius = sum_radius + sameCircles[j, 0]
                    sum_height = sum_height + sameCircles[j, 1]
                    sum_width = sum_width + sameCircles[j, 2]

            rightCircles[rightCirclesIndex, 0] = round(sum_radius / i_same)
            rightCircles[rightCirclesIndex, 1] = round(sum_height / i_same)
            rightCircles[rightCirclesIndex, 2] = round(sum_width / i_same)

            rightCirclesIndex = rightCirclesIndex + 1

    return rightCircles


def white(image):
    rows = image.shape[0]
    cols = image.shape[1]
    WHITES = np.zeros((rows, cols))
    for i in range(0, rows):
        for j in range(0, cols):
            if image[i, j] != 0:
                WHITES[i, j] = 255
    return WHITES


def clear(image):
    rows = image.shape[0]
    cols = image.shape[1]
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            if image[i, j] == 255:
                if image[i - 1, j] == 0 and image[i - 1, j - 1] == 0 and image[i + 1, j] == 0 and image[
                    i + 1, j + 1] == 0 and image[i, j - 1] == 0 and image[i, j + 1] == 0 and image[
                    i + 1, j - 1] == 0 and image[i - 1, j + 1] == 0:
                    image[i, j] == 0
    return image


def luminance(image):
    GRAY = 0.2989 * image[:, :, 0] + 0.5870 * image[:, :, 1] + 0.1140 * image[:, :, 2]
    return GRAY


def sobel(image):
    # Sobel filter kernel
    Gx = np.array([[-1, 0, 1, ], [-2, 0, 2], [-1, 0, 1]])
    Gy = np.array([[-1, -2, -1, ], [0, 0, 0], [1, 2, 1]])

    GRADIENT_X = signal.convolve2d(image, Gx, 'same')
    GRADIENT_Y = signal.convolve2d(image, Gy, 'same')
    SOBEL = abs(GRADIENT_X) + abs(GRADIENT_Y)
    return SOBEL


def choosenImage():
    fileName = input('File to open? \n> ')
    print()
    if fileName == '2' or fileName == '3' or fileName == '4' or fileName == '5' or fileName == '6' or fileName == '8':
        file = 'images/coins00' + fileName + '.tif'
        print('Opening file -->', file[7:], '...\n')
    else:
        print('Could not open file `coins00' + fileName + '.tif`')
        file = 'images/coins002.tif'
        print('Opening Default file -->', file[7:], '...\n')

    return file


# ~~~~~~~ Main ~~~~~~~~

imageNoFilter = myImageToArray(choosenImage())

# Make image grey
image = luminance(imageNoFilter)

# Make image only black and white (Better performance and better edge)
image = white(image)

# Clear some white pixels that don't belong in a circle
image = clear(image)

# Edge image
image = sobel(image)

# HT
createCircles = myHoughTransformation(image, 15, 37, radius=[38, 52])

# Show results
showImageWithCircles(createCircles)
