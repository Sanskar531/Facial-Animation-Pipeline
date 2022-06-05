import tensorflow as tf;
from threading import Lock, Thread;
from training_loop import wrapper_with_assigner;

class Assigner():
  """
    Assigner Class which works as a Manager to assign frames to 
    threads. 
  """
  def __init__(self, file_path, blendshapes, frames, basis_mesh, epochs, learning_rate, sub_frame_limit):
    """
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
    self.file_path = file_path;
    self.blendshapes = tf.Variable([i.flatten() for i in blendshapes]);
    self.frames = frames;
    self.basis_mesh = tf.Variable(basis_mesh.flatten());
    self.epochs = epochs;
    self.learning_rate = learning_rate;
    self.lock = Lock();
    self.sub_frame_limit = sub_frame_limit
    self.curr_frame_no = 0;
    self.active_threads = {};
  
  def assign_frames(self, gpu_name):
    """
      Assigns frames to gpu names.
      @params:
        gpu_name: name of the gpu we should assign new frames to.
      returns:
        None
    """
    if(self.curr_frame_no is None):
      return;
    elif(self.curr_frame_no+ self.sub_frame_limit > len(self.frames)):
      with self.lock:
        self.active_threads[gpu_name] = (Thread(target=(wrapper_with_assigner), args=(self.frames, self.curr_frame_no, len(self.frames), self.blendshapes, self.basis_mesh, self.epochs, self.learning_rate, self.file_path, gpu_name, self)));
        self.curr_frame_no = None;
    else:
      with self.lock:
        print("New Thread Spawned");
        self.active_threads[gpu_name] = (Thread(target=(wrapper_with_assigner), args=(self.frames, self.curr_frame_no, self.curr_frame_no+self.sub_frame_limit, self.blendshapes, self.basis_mesh, self.epochs, self.learning_rate, self.file_path, gpu_name, self)));
        self.curr_frame_no = self.curr_frame_no + self.sub_frame_limit;
        self.active_threads[gpu_name].start();

  def check_all_thread_dead(self):
    """
      Checks if all threads has finished executing.
      returns:
        bool: value describing whether all threads are dead.
    """
    all_dead = True;
    for i in self.active_threads:
      if self.active_threads[i].is_alive():
        all_dead = False;
    return all_dead;