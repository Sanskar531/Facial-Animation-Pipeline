import mediapipe as mp;
import numpy as np;
from scipy.spatial import procrustes;
import cv2;

def get_lm_as_np(meshes):
  """
    Changes mediapipe's landmark object to numpy array.
    @params:
      meshes: an array with all landmark objects
    returns:
      numpy array: a numpy array with all landmarks changed to numpy arrays.
  """
  all_processed = [];
  for i in meshes:
    curr_mesh = [];
    for j in range(468):
      curr_co = [];
      curr_co.append(i.multi_face_landmarks[0].landmark[j].x);
      curr_co.append(i.multi_face_landmarks[0].landmark[j].y);
      curr_co.append(i.multi_face_landmarks[0].landmark[j].z);
      curr_mesh.append(np.array(curr_co));
    all_processed.append(np.array(curr_mesh));
  return np.array(all_processed);

def align_mesh(basis, to_be_aligned):
  """
    @params:
      basis: Neutral/Basis face Mesh
      to_to_aligned: All the meshes that need to be aligned w.r.t basis
    returns:
      list: A list with aligned basis and aligned blendshapes
  """
  aligned_mesh = [];
  for i in to_be_aligned:
    data1, data2, _ = procrustes(basis, i);
    aligned_mesh.append(data2);
  return data1, aligned_mesh;

def generate_meshes(video_path):  
  """
    Generates Meshes for all frames in the video using the path of the video.
    @params:
      video_path: path to the video file.
    returns:
      fps: Frame rate of the video
      meshes: an array with all landmark object for all faces in the frames.
  """
  meshes = [];
  mp_drawing = mp.solutions.drawing_utils
  mp_drawing_styles = mp.solutions.drawing_styles
  mp_face_mesh = mp.solutions.face_mesh
  drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
  cap = cv2.VideoCapture(video_path);
  fps = cap.get(cv2.CAP_PROP_FPS);
  with mp_face_mesh.FaceMesh(
      max_num_faces=1,
      refine_landmarks=True,
      min_detection_confidence=0.5,
      min_tracking_confidence=0.5) as face_mesh:
    while cap.isOpened():
      success, image = cap.read()
      if not success:
        break;
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      results = face_mesh.process(image)
      meshes.append(results);
  cap.release()
  return fps, meshes;