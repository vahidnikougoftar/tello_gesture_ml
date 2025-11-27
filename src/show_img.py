import os 
import argparse 
import cv2 

parser = argparse.ArgumentParser(description='enter full filepath')
parser.add_argument('filepath', type=str, help='Path to the input file.')
args = parser.parse_args()

input_filepath = args.filepath


if os.path.exists(input_filepath):
    filepath=input_filepath
else:
    print('file did not exist. showing IMG_4040 instead')
    filepath = r'../data/processed/train/takeoff/IMG_4040.JPG'

img = cv2.imread(filepath)

# Create a named window (optional, but good practice)
cv2.namedWindow("Display Image", cv2.WINDOW_NORMAL)

# Display the image
cv2.imshow("Display Image", img)

# Wait for a key press (0 means wait indefinitely)
cv2.waitKey(0)

# Destroy all OpenCV windows
cv2.destroyAllWindows()
