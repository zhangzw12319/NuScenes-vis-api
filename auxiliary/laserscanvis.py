#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.
import os
import copy
import vispy
import nuscenes
import numpy as np


from vispy.scene import visuals, SceneCanvas
from matplotlib import pyplot as plt
from tqdm import tqdm


class LaserScanVis:
  """Class that creates and handles a visualizer for a pointcloud"""
  """
    gt_semantics: whether to show ground truth semantics
    gt_instances: whether to show ground truth instances
    pred_semantics: whether to show predicted semantics
    pred_instances: whether to show predicted instances
    classwise: whether to show classwise predictions
  """

  def __init__(self, raw_scan, gt_scan, pred_scan, scan_names,
               lidar_tokens, sample_tokens, scene_tokens,
               gt_label_names, pred_label_names,
               nusc, cam,
               offset=0, gt_semantics=True, gt_instances=False,
               pred_semantics=False, pred_instances=False,
               gt_classwise=False, pred_classwise=False,
               render_lidar=False,
               dark_mode=False,
               cfg=None):
    self.raw_scan = raw_scan
    self.gt_scan = gt_scan
    self.pred_scan = pred_scan
    self.scan_names = scan_names
    self.lidar_tokens = lidar_tokens
    self.sample_tokens = sample_tokens
    self.scene_tokens = scene_tokens

    self.gt_label_names = gt_label_names
    self.pred_label_names = pred_label_names

    self.offset = offset
    self.total = len(self.scan_names)
    self.gt_semantics = gt_semantics
    self.gt_instances = gt_instances
    self.pred_semantics = pred_semantics
    self.pred_instances = pred_instances
    self.gt_classwise = gt_classwise
    self.pred_classwise = pred_classwise
    
    self.nusc = nusc
    self.cam = cam
    self.render_lidar = render_lidar
    self.cfg = cfg
    
    if self.cfg != None:
      self.inv_learning_map = self.cfg["inverse_learning_map"]
    
    
    # color setting
    self.border_color = [1.0, 1.0, 1.0, 1.0]
    if dark_mode:
      self.bg_color = [0.0, 0.0, 0.0, 1.0]
    else:
      self.bg_color = [1.0, 1.0, 1.0, 1.0]
    
    self.selected_cls = 4

    # sanity check
    if not self.gt_semantics and self.gt_instances:
      print("Instances are only allowed in when semantics=True")
      raise ValueError
    
    if not self.pred_semantics and self.pred_instances:
      print("Instances are only allowed in when semantics=True")
      raise ValueError

    self.point_size = 3
    self.print_help()
    self.reset()
    self.update_scan()

  def reset(self):
    """ Reset. """
    # last key press (it should have a mutex, but visualization is not
    # safety critical, so let's do things wrong)
    self.action = "no"  # no, next, back, quit are the possibilities
    self.view_list = []
    self.linked = False

    ###############################################################

    # new canvas prepared for visualizing data
    self.canvas = SceneCanvas(keys='interactive', show=True)
    # interface (n next, b back, q quit, very simple)
    self.canvas.events.key_press.connect(self.key_press)
    self.canvas.events.draw.connect(self.draw)
    # grid
    self.grid = self.canvas.central_widget.add_grid()

    # laserscan part
    self.scan_view = vispy.scene.widgets.ViewBox(
        border_color=self.border_color, bgcolor=self.bg_color, parent=self.canvas.scene)
    self.grid.add_widget(self.scan_view, 0, 0)
    self.scan_vis = visuals.Markers()
    self.scan_view.camera = 'turntable'
    self.scan_view.add(self.scan_vis)
    visuals.XYZAxis(parent=self.scan_view.scene)
    self.view_list.append(self.scan_view)
    
    # add ground-truth point visualization
    if (self.gt_semantics) and (self.gt_label_names is not None):
      print("Using semantics in visualizer")
      self.sem_view = vispy.scene.widgets.ViewBox(
          border_color=self.border_color, bgcolor=self.bg_color, parent=self.canvas.scene)
      self.grid.add_widget(self.sem_view, 0, 1)
      self.sem_vis = visuals.Markers()
      self.sem_view.camera = 'turntable'
      self.sem_view.add(self.sem_vis)
      visuals.XYZAxis(parent=self.sem_view.scene)
      self.view_list.append(self.sem_view)

    if (self.gt_instances) and (self.gt_label_names is not None):
      print("Using instances in visualizer")
      self.inst_view = vispy.scene.widgets.ViewBox(
          border_color=self.border_color, bgcolor=self.bg_color, parent=self.canvas.scene)
      self.grid.add_widget(self.inst_view, 0, 2)
      self.inst_vis = visuals.Markers()
      self.inst_view.camera = 'turntable'
      self.inst_view.add(self.inst_vis)
      visuals.XYZAxis(parent=self.inst_view.scene)
      self.view_list.append(self.inst_view)

    # add predicted point visualization
    if (self.pred_semantics) and (self.pred_label_names is not None):
      print("Using predicted semantics in visualizer")
      self.pred_sem_view = vispy.scene.widgets.ViewBox(
          border_color=self.border_color, bgcolor=self.bg_color, parent=self.canvas.scene)
      self.grid.add_widget(self.pred_sem_view, 1, 1)
      self.pred_sem_vis = visuals.Markers()
      self.pred_sem_view.camera = 'turntable'
      self.pred_sem_view.add(self.pred_sem_vis)
      visuals.XYZAxis(parent=self.pred_sem_view.scene)
      self.view_list.append(self.pred_sem_view)    
   
    if (self.pred_instances) and (self.pred_label_names is not None):
      print("Using predicted instances in visualizer")
      self.pred_inst_view = vispy.scene.widgets.ViewBox(
          border_color='white', bgcolor=self.bg_color, parent=self.canvas.scene)
      self.grid.add_widget(self.pred_inst_view, 1, 2)
      self.pred_inst_vis = visuals.Markers()
      self.pred_inst_view.camera = 'turntable'
      self.pred_inst_view.add(self.pred_inst_vis)
      visuals.XYZAxis(parent=self.pred_inst_view.scene)
      self.view_list.append(self.pred_inst_view)
      

    ###############################################################

    # add another canvas for classwise visualization
    self.canvas2 = SceneCanvas(keys='interactive', show=True)
    # interface (n next, b back, q quit, plus 1-16 class)
    self.canvas2.events.key_press.connect(self.key_press)
    self.canvas2.events.draw.connect(self.draw)
    # grid
    self.grid2 = self.canvas2.central_widget.add_grid()
    
    # gt classwise
    if self.gt_classwise:
      self.gt_cls_view = vispy.scene.widgets.ViewBox(
          border_color=self.border_color, bgcolor=self.bg_color, parent=self.canvas2.scene)
      self.grid2.add_widget(self.gt_cls_view, 0, 0)
      self.gt_cls_vis = visuals.Markers()
      self.gt_cls_view.camera = 'turntable'
      self.gt_cls_view.add(self.gt_cls_vis)
      visuals.XYZAxis(parent=self.gt_cls_view.scene)
      self.view_list.append(self.gt_cls_view)
    
    # pred classwise
    if self.pred_classwise:
      self.pred_cls_view = vispy.scene.widgets.ViewBox(
          border_color=self.border_color, bgcolor=self.bg_color, parent=self.canvas2.scene)
      self.grid2.add_widget(self.pred_cls_view, 0, 1)
      self.pred_cls_vis = visuals.Markers()
      self.pred_cls_view.camera = 'turntable'
      self.pred_cls_view.add(self.pred_cls_vis)
      visuals.XYZAxis(parent=self.pred_cls_view.scene)
      self.view_list.append(self.pred_cls_view)

    ###############################################################

    # img canvas size
    self.multiplier = 1
    self.canvas_W = 1024
    self.canvas_H = 64
    if self.gt_semantics:
      self.multiplier += 1
    if self.gt_instances:
      self.multiplier += 1

    # new canvas for range img
    self.img_canvas = SceneCanvas(keys='interactive', show=True,
                                  size=(self.canvas_W, self.canvas_H * self.multiplier))
    # grid
    self.img_grid = self.img_canvas.central_widget.add_grid()
    # interface (n next, b back, q quit, very simple)
    self.img_canvas.events.key_press.connect(self.key_press)
    self.img_canvas.events.draw.connect(self.draw)

    # add a view for the depth
    self.img_view = vispy.scene.widgets.ViewBox(
        border_color=self.border_color, bgcolor=self.bg_color, parent=self.img_canvas.scene)
    self.img_grid.add_widget(self.img_view, 0, 0)
    self.img_vis = visuals.Image(cmap='viridis')
    self.img_view.add(self.img_vis)

    # add ground-truth semantics
    if (self.gt_semantics) and (self.gt_label_names is not None):
      self.sem_img_view = vispy.scene.widgets.ViewBox(
          border_color=self.border_color, bgcolor=self.bg_color, parent=self.img_canvas.scene)
      self.img_grid.add_widget(self.sem_img_view, 1, 0)
      self.sem_img_vis = visuals.Image(cmap='viridis')
      self.sem_img_view.add(self.sem_img_vis)

    # add ground-truth instances
    if (self.gt_instances) and (self.gt_label_names is not None):
      self.inst_img_view = vispy.scene.widgets.ViewBox(
          border_color=self.border_color, bgcolor=self.bg_color, parent=self.img_canvas.scene)
      self.img_grid.add_widget(self.inst_img_view, 2, 0)
      self.inst_img_vis = visuals.Image(cmap='viridis')
      self.inst_img_view.add(self.inst_img_vis)
      
    
    # init options of rendering lidar
    self.set_show_panoptic(True)
    self.set_filtering()
  
  
  def link_cameras(self):
    self.linked = True
    if len(self.view_list)>1:
      for view in self.view_list[1:]:
        self.view_list[0].camera.link(view.camera)
    
  # def unlink_cameras(self):
  #   self.linked = False
  #   if len(self.view_list)>1:
  #     for view in self.view_list[1:]:
  #       self.view_list[0].camera.unlink(view.camera)
  
  def reset_camera_view(self):
    for view in self.view_list:
      view.camera.reset()
    

  def get_mpl_colormap(self, cmap_name):
    cmap = plt.get_cmap(cmap_name)

    # Initialize the matplotlib color map
    sm = plt.cm.ScalarMappable(cmap=cmap)

    # Obtain linear color range
    color_range = sm.to_rgba(np.linspace(0, 1, 256), bytes=True)[:, 2::-1]

    return color_range.reshape(256, 3).astype(np.float32) / 255.0


  def update_scan(self):

    ###############################################################

    if self.gt_semantics:
      assert self.gt_scan is not None
      # first open data
      self.gt_scan.open_scan(self.scan_names[self.offset])
      # open label and colorize
      self.gt_scan.open_label(self.gt_label_names[self.offset])
      self.gt_scan.colorize()
      # plot semantics
      self.sem_vis.set_data(self.gt_scan.points,
                            face_color=self.gt_scan.sem_label_color[..., ::-1],
                            edge_color=self.gt_scan.sem_label_color[..., ::-1],
                            size=self.point_size)
      # plot instances
      if self.gt_instances:
        self.inst_vis.set_data(self.gt_scan.points,
                              face_color=self.gt_scan.inst_label_color[..., ::-1],
                              edge_color=self.gt_scan.inst_label_color[..., ::-1],
                              size=self.point_size)

    ###############################################################

    if self.pred_semantics:
      assert self.pred_scan is not None
      # first open data
      self.pred_scan.open_scan(self.scan_names[self.offset])
      self.pred_scan.open_label(self.pred_label_names[self.offset])
      self.pred_scan.colorize()
      self.pred_sem_vis.set_data(self.pred_scan.points,
                            face_color=self.pred_scan.sem_label_color[..., ::-1],
                            edge_color=self.pred_scan.sem_label_color[..., ::-1],
                            size=self.point_size)
      if self.pred_instances:
        self.pred_inst_vis.set_data(self.pred_scan.points,
                              face_color=self.pred_scan.inst_label_color[..., ::-1],
                              edge_color=self.pred_scan.inst_label_color[..., ::-1],
                              size=self.point_size)
    ###############################################################

    if self.gt_classwise:
      # can be commented out here to save time
      # first open data
      # self.gt_scan.open_scan(self.scan_names[self.offset])
      # self.gt_scan.open_label(self.gt_label_names[self.offset])
      # self.gt_scan.colorize() 
      print("gt_classwise!")
      mask = (self.gt_scan.sem_label == self.selected_cls)
      # print("self.selected_cls", self.selected_cls)
      # we have limited self.selected_cls range in def key_press
      if self.selected_cls <=10: 
        gt_cls_color = self.gt_scan.inst_label_color
        gt_cls_color[~mask] = np.array([0.1, 0.1, 0.1])
      else:
        gt_cls_color = self.gt_scan.sem_label_color
        gt_cls_color[~mask] = np.array([0.1, 0.1, 0.1])
      self.gt_cls_vis.set_data(self.gt_scan.points,
                          face_color=gt_cls_color[..., ::-1],
                          edge_color=gt_cls_color[..., ::-1],
                          size=self.point_size)
      
    if self.pred_classwise:
      # can be commented out here to save time
      # first open data
      # self.gt_scan.open_scan(self.scan_names[self.offset])
      # self.gt_scan.open_label(self.gt_label_names[self.offset])
      # self.gt_scan.colorize() 
      mask = (self.pred_scan.sem_label == self.selected_cls)
      if self.selected_cls <=10:
        pred_cls_color = self.pred_scan.inst_label_color
        pred_cls_color[~mask] = np.array([0.1, 0.1, 0.1])
      else:
        pred_cls_color = self.pred_scan.sem_label_color
        pred_cls_color[~mask] = np.array([0.1, 0.1, 0.1])
      self.pred_cls_vis.set_data(self.pred_scan.points,
                          face_color=pred_cls_color[..., ::-1],
                          edge_color=pred_cls_color[..., ::-1],
                          size=self.point_size)

    ###############################################################
    # then change names
    title = "scan " + str(self.offset)
    self.canvas.title = "Semantic and Instance Visualization:  " + title
    self.canvas2.title = "Classwise Comparison:  " + title
    self.img_canvas.title = "Range View:  " + title

    ###############################################################

    # plot range view scan
    self.raw_scan.open_scan(self.scan_names[self.offset])
    power = 16
    # print()
    range_data = np.copy(self.raw_scan.unproj_range)
    # print(range_data.max(), range_data.min())
    range_data = range_data**(1 / power)
    # print(range_data.max(), range_data.min())
    viridis_range = ((range_data - range_data.min()) /
                     (range_data.max() - range_data.min()) *
                     255).astype(np.uint8)
    viridis_map = self.get_mpl_colormap("viridis")
    viridis_colors = viridis_map[viridis_range]
    self.scan_vis.set_data(self.raw_scan.points,
                           face_color=viridis_colors[..., ::-1],
                           edge_color=viridis_colors[..., ::-1],
                           size=self.point_size)

    # now do all the range image stuff
    # plot range image
    data = np.copy(self.raw_scan.proj_range)
    # print(data[data > 0].max(), data[data > 0].min())
    data[data > 0] = data[data > 0]**(1 / power)
    data[data < 0] = data[data > 0].min()
    # print(data.max(), data.min())
    data = (data - data[data > 0].min()) / \
        (data.max() - data[data > 0].min())
    # print(data.max(), data.min())
    self.img_vis.set_data(data)
    self.img_vis.update()

    if self.gt_semantics:
      self.sem_img_vis.set_data(self.gt_scan.proj_sem_color[..., ::-1])
      self.sem_img_vis.update()

    if self.gt_instances:
      self.inst_img_vis.set_data(self.gt_scan.proj_inst_color[..., ::-1])
      self.inst_img_vis.update()
  
  def set_show_panoptic(self, allow=False):
    self.show_panoptic = allow
    self.show_lidarseg = not allow
    
  def set_filtering(self, array=None):
    assert (array is None) or (type(array) == list) and len(array) > 0
    if array is None:
      self.filtering_list = array
    else:
      self.filtering_list = np.vectorize(self.inv_learning_map.__getitem__)(array)
    

  # from https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/tutorials/nuscenes_lidarseg_panoptic_tutorial.ipynb
  def render_lidar_to_img_gt(self):
    flag = False
    if (self.render_lidar) and (type(self.nusc) == nuscenes.NuScenes):
      # render the sequence of scenes as video clip
  #         self.nusc.render_scene_channel_lidarseg(self.scene_tokens[self.offset], 
  #                                    'CAM_BACK', 
  # #                                    filter_lidarseg_labels=[18, 24, 28],
  #                                    verbose=True, 
  #                                    dpi=100,
  #                                    imsize=(1280, 720),
  #                                    render_mode='image',
  #                                    show_panoptic=True)
      # render gt
      self.nusc.render_sample(self.sample_tokens[self.offset], 
                              filter_lidarseg_labels=self.filtering_list,
                                # verbose=True, 
                                show_lidarseg=self.show_lidarseg,
                                show_panoptic=self.show_panoptic)
      flag = True
      # print("Token List", self.sample_tokens[self.offset])

    return flag
  
  
  def render_lidar_to_img_preds(self):
    flag = False
    if (self.render_lidar) and (type(self.nusc) == nuscenes.NuScenes):
      # render the sequence of scenes as video clip
  #         self.nusc.render_scene_channel_lidarseg(self.scene_tokens[self.offset], 
  #                                    'CAM_BACK', 
  # #                                    filter_lidarseg_labels=[18, 24, 28],
  #                                    verbose=True, 
  #                                    dpi=100,
  #                                    imsize=(1280, 720),
  #                                    render_mode='image',
  #                                    show_panoptic=True)
      # render pred
      if (self.pred_semantics):
        filepath_tmp = self.pred_label_names[self.offset]
        if self.show_lidarseg:
          suffix = "_original_lidarseg.bin"
        else:
          suffix = "_original_panoptic.npz"
        if not os.path.exists(filepath_tmp.replace(".npz", suffix)):
          print("Converting...")
          pred_label = np.load(filepath_tmp)['data'].reshape((-1))
          print(pred_label)
          print(pred_label.max())
          original_pred_sems = np.vectorize(self.inv_learning_map.__getitem__)(pred_label // 1000)
          filepath_tmp = filepath_tmp.replace(".npz", suffix)
          if self.show_lidarseg:
            original_pred_sems.astype(np.uint8).tofile(filepath_tmp)
          else:
            original_pred_pano = original_pred_sems * 1000 + pred_label % 1000
            np.savez_compressed(filepath_tmp, data=original_pred_pano.astype(np.uint16))
            
          print("Converting Complete! Saved to: ", filepath_tmp)
        else:
          filepath_tmp = filepath_tmp.replace(".npz", suffix)
          print("Directing to: ", filepath_tmp)
        
        self.nusc.render_sample(self.sample_tokens[self.offset], 
                              filter_lidarseg_labels=self.filtering_list,
                                lidarseg_preds_bin_path=filepath_tmp,
                                # verbose=True, 
                                show_lidarseg=self.show_lidarseg,
                                show_panoptic=self.show_panoptic)
        flag = True
        # print("Token List", self.sample_tokens[self.offset])
    return flag
    
    
  def print_path_info(self):
    print("LiDAR_PATH:", self.scan_names[self.offset])
    cam_order = ["CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_RIGHT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_FRONT_LEFT"]
    for cam_view in cam_order:
      cam_token = self.cam[self.offset][cam_view]["sample_data_token"]
      cam_path = self.nusc.get("sample_data", cam_token)["filename"]
      print(cam_view + ": " + cam_token + "\t" + cam_path)
    print()


  # interface
  def key_press(self, event):
    self.canvas.events.key_press.block()
    self.img_canvas.events.key_press.block()
    if event.key == 'N':
      self.offset += 1
      if self.offset >= self.total:
        self.offset = 0
      self.update_scan()

    elif event.key == 'B':
      self.offset -= 1
      if self.offset < 0:
        self.offset = self.total - 1
      self.update_scan()

    elif event.key == 'Q' or event.key == 'Escape':
      self.destroy()
      exit()

    elif event.key in ['1','2','3','4','5','6','7','8','9']:
      self.selected_cls = int(str(event.key)[6])
      self.update_scan()
      self.set_filtering([self.selected_cls])
      print(f"You select class {self.selected_cls}...")

    elif event.key in ['C','D','E','F','G','H','I']:
      self.selected_cls = ord(str(event.key)[6]) - ord('C') + 10
      self.update_scan()
      print(f"You select class {self.selected_cls}...")

    elif event.key == 'L':
      if not self.linked:
        self.link_cameras()
        print("Linking Cameras!")
      # else:
      #   self.unlink_cameras()
      #   print("Unlinking Cameras!")
    
    elif event.key == 'O':
      self.reset_camera_view()

    elif event.key == 'P':
      if self.find_nearest_cls():
        self.update_scan()
        print(f"You select class {self.selected_cls}...")

    elif event.key == 'R':
      flag = self.render_lidar_to_img_gt()
      if flag:
        print("Successfully Rendered gt!")
      else:
        print("Please set --render_lidar ")

    elif event.key == 'S':
      flag = self.render_lidar_to_img_preds()
      if flag:
        print("Successfully Rendered preds!")
      else:
        print("Please check that you have set --render_lidar and --pred_semantics, and then have provided predictions by --predictions path_to_prediction_folder.")
  
    elif event.key == 'T':
      self.print_path_info()

    elif event.key == 'U':
      self.set_show_panoptic(not self.show_panoptic)
      if self.show_lidarseg:
        print("Switch *on* show_lidarseg and switch *off* show_panoptic in rendering mode!")
      else:
        print("Switch *off* show_lidarseg and switch *on* show_panoptic in rendering mode!")

    elif event.key == 'V':
      if self.filtering_list is None:
        self.set_filtering([self.selected_cls])
        print("Switch *on* filtering list in rendering mode")
      else:
        self.filtering_list = None
        print("Switch *off* filtering in rendering mode")

    elif event.key == 'M':
        self.print_help()
    else:
      pass

  def draw(self, event):
    if self.canvas.events.key_press.blocked():
      self.canvas.events.key_press.unblock()
    if self.canvas2.events.key_press.blocked():
      self.canvas2.events.key_press.unblock()

    if self.img_canvas.events.key_press.blocked():
      self.img_canvas.events.key_press.unblock()

  def destroy(self):
    # destroy the visualization
    self.canvas.close()
    self.img_canvas.close()
    vispy.app.quit()


  def run(self):
    vispy.app.run()


  def find_nearest_cls(self):
    print(f"Search for class {self.selected_cls}...")
    if self.gt_semantics:
      temp_scan = copy.deepcopy(self.gt_scan)
    elif self.pred_semantics:
      temp_scan = copy.deepcopy(self.pred_scan)
    else:
      assert False, "Please set --gt_semantics or --pred_semantics"

    cnt = 0
    if (self.total>1000):
        msg = "It may search for a long time to complete.\
Please do not close, move window or press other keys.\
This progam use naive search and do not include blocking protection."
        print(msg)
    pbar = tqdm(total=self.total-1)
    current_offset = self.offset

    while cnt < self.total-1:
      cnt += 1
      current_offset = current_offset + 1
      if current_offset >= self.total:
          current_offset = 0

      temp_scan.open_scan(self.scan_names[current_offset])
      # check gt
      if self.gt_semantics:
        temp_scan.open_label(self.gt_label_names[current_offset])
        temp_sem_label = temp_scan.sem_label
        mask = (temp_sem_label == self.selected_cls)
        if mask.sum() > 14:
          self.offset = current_offset
          break
      
      # check pred
      if self.pred_semantics:
        temp_scan.open_label(self.pred_label_names[current_offset])
        temp_sem_label = temp_scan.sem_label
        mask = (temp_sem_label == self.selected_cls)
        if mask.sum() > 14:
          self.offset = current_offset
          break
      
      pbar.update(1)
    
    del temp_scan, pbar
    if cnt >= self.total-1:
      print(f"No more scans with class {self.selected_cls}!")
      return False
    else:
      return True
  
  def print_help(self):
    # print instructions
    print('*'*20)
    print("To navigate:")
    print("\tb: back (previous scan)")
    print("\tn: next (next scan)")
    print("\tq: quit (exit program)")
    print("\t1-9 and c-i: select class from 1-16")
    print("\tp: proceed to search the nearest keyframe which contains the selected class")

    print('*'*20)
    print("To render into surrounding cameras:")
    print("\tr: render lidar to surrounding images with gt labels")
    print("\ts: render lidar to surrounding images with pred labels")
    print("\tu: Switch show_lidarseg (reverse show_panoptic) in rendering mode!")
    print("\tv: Switch filtering list in rendering mode")

    print('*'*20)
    print("Mouse Movement:")
    print("\tScroll/Right mouse button: zoom in/out")
    print("\tLeft mouse button: rotates the view around its center point")
    print("\tShift + Left mouse button: Shifting")
    print("\to: reset camera view")
    print('*'*20)

    print("Other Functions:")
    print("\tt: print path and token infos")
    print("\tl: link cameras")
    print("\tm: print help message")
    print()