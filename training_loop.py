import tensorflow as tf;
import numpy as np;
from tensorflow.python.client import device_lib;
import Assigner;

def get_gpu_count():
  """
    returns: Number of GPUs available in current runtime.
  """
  return len(tf.config.list_physical_devices('GPU'));

def get_gpu_names():
  """
    returns: Names of GPUs available in current runtime.
  """
  devices = device_lib.list_local_devices();
  gpus = [];
  for i in devices:
    if i.device_type == "GPU":
      gpus.append(i.name);
  return gpus;

def wrapper_with_assigner(frames, curr_frame_no, end_frame_no, blendshapes, basis_mesh, epochs, learning_rate, file_path, gpu_name, assigner):
  """
    wrapper function for call to assigner after thread finishes
    @params:
      file_path: path to where the optimized files should be saved.
      blendshapes: an array with mediapipe mesh blendshapes.
      frames: An array of target meshes.
      basis_mesh: Neutral/Basis face Mesh.
      epochs: Number of iterations.
      learning_rate: learning rate for gradient descent.
      sub_frame_limit: how many frames a single thread should be responsible for.
    returns:
      None
  """
  process_frames(frames, curr_frame_no, end_frame_no, blendshapes, basis_mesh, epochs, learning_rate, file_path, gpu_name)
  assigner.assign_frames(gpu_name);

def process_video(file_path, blendshapes, frames, basis_mesh, epochs, learning_rate, sub_frame_limit):
  """
    Processes a video based on available gpus.
    @params:
      file_path: path to where the optimized files should be saved.
      blendshapes: an array with mediapipe mesh blendshapes.
      frames: An array of target meshes.
      basis_mesh: Neutral/Basis face Mesh.
      epochs: Number of iterations.
      learning_rate: learning rate for gradient descent.
      sub_frame_limit: how many frames a single thread should be responsible for.
    returns:
      None
  """
  number_of_frames = 100;
  blendshapes = tf.Variable([i.flatten() for i in blendshapes]);
  basis_mesh = tf.Variable(basis_mesh.flatten());
  gpu_count = get_gpu_count();
  frame_no = 0;
  if get_gpu_count() > 1 :
    gpu_names = get_gpu_names();
    assigner = Assigner(file_path, blendshapes, frames, basis_mesh, epochs, learning_rate, sub_frame_limit);
    for i in gpu_names:
      assigner.assign_frames(i);

    #Loop that waits for the threads to join.
    while not assigner.check_all_thread_dead():
      for i in list(assigner.active_threads):
        if assigner.active_threads[i] is not None:
          if not assigner.active_threads[i].is_alive():
            pass;
          else:
            assigner.active_threads[i].join();
  else:
    for i in range(int(len(frames)/number_of_frames)):
      process_frames(frames, frame_no, frame_no+number_of_frames, blendshapes, basis_mesh, epochs, learning_rate, file_path);
      frame_no+=number_of_frames;

def process_frames(frames, start, end, blendshapes, basis_mesh, epochs, learning_rate, file_path, GPU = None):
  """
    Wrapper function assign gpus.
    @params:
      frames: all the meshes in the frames that need to be processed
      start: start idx inside frames
      end: end idx inside frames
      basis_mesh: neutral mesh
      epochs: number of iterations
      file_path: file path to save the optimized values
      GPU: which device to use to run the training loop
    returns:
      None
  """
  if len(frames)<=end:
    end = len(frames);
  if GPU is not None:
    with tf.device(GPU):
      optimize_key_values(frames, start, end, blendshapes, basis_mesh, epochs, learning_rate, file_path);
  else:
    optimize_key_values(frames, start, end, blendshapes, basis_mesh, epochs, learning_rate, file_path);  

def optimize_key_values(frames, start, end, blendshapes, basis_mesh, epochs, learning_rate, file_path):
  """
    @params:
      frames: all the meshes in the frames that need to be processed
      start: start idx inside frames
      end: end idx inside frames
      basis_mesh: neutral mesh
      epochs: number of iterations
      file_path: file path to save the optimized values
    returns:
      None
  """
  for i in range(start, end):
    if i%8 ==0:
      target = tf.Variable(frames[i].flatten());
      keys = tf.Variable(np.random.uniform(low=0.0, high=1.1, size=11).reshape(11,1));
      for j in range(epochs):
        with tf.GradientTape() as tape:
          tape.watch(keys);
          pred = prediction_function(keys, basis_mesh, blendshapes);
          y = tf.norm(tf.subtract(target,pred), ord=2);
          if j is 0:
            print("Frame No: {} Initial Loss: {} Device: {}".format(i, tf.reduce_sum(y), y.device));
        gradients = tape.gradient(y, keys);
        keys = keys - learning_rate*(gradients);
      print("Frame No: {} Final Loss: {} Device: {}".format(i, tf.reduce_sum(y), y.device));
      np.save(file_path+"/{}.npy".format(i), [target, -keys]);

def prediction_function(keys, basis_mesh, blendshapes):
  """
    Prediction/Hypothesis function
    @params:
      keys: weight values of each blendshape
      basis_mesh: neutral mesh
      blendshapes: array of all the blendshape meshes
    returns:
      Array: Predicted Mesh
  """
  return tf.add(basis_mesh, tf.matmul(keys, tf.subtract(basis_mesh, blendshapes), transpose_a=True))