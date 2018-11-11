import cv2
import dlib
from imutils import face_utils
from imutils.face_utils import FaceAligner

PATH = 'shape_predictor_68_face_landmarks.dat'
predictor = dlib.shape_predictor(PATH)

def facial_landmarks(gray, rect):
    shape = predictor(gray, rect)
    shape = face_utils.shape_to_np(shape)
    return shape

def face_aligner(gray, rect, image, width, height):
    fa = FaceAligner(predictor, desiredFaceWidth=width, desiredFaceHeight = height)
    face_aligned = fa.align(image, gray, rect)
    return face_aligned
