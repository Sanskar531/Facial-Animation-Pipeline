import math;
import numpy as np;

def get_blendshapes(meshes, fps):
  """
  Generate blendshapes using the predefined video structure mentioned in the demo workflow.
    @params:
      meshes: Blendshape Extraction video face meshes (expected length less than 2:05 seconds)
      fps: the FPS of the Blendshape Extraction video
    returns:
      Array: all blendshapes that are extracted from the video segment
  """
  frame_limit = math.ceil(fps)*10;
  sub_len = [int(7*fps), int(9*fps)];
  idx = 0;
  new_meshes = [];
  for i in range(int(math.floor(len(meshes)/frame_limit))):
    curr_shape_idx = (np.array(sub_len)+(frame_limit*idx));
    new_meshes.append(meshes[curr_shape_idx[0]:curr_shape_idx[1]]);
    idx+=1;
  for i in range(len(new_meshes)):
    new_meshes[i] = (np.sum(new_meshes[i],axis=0))/len(new_meshes[i]);
  return new_meshes;