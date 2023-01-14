#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import copy
import vispy
from vispy.scene import visuals, SceneCanvas
import numpy as np
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

  def __init__(self, raw_scan, gt_scan, pred_scan, scan_names, gt_label_names, pred_label_names,
               offset=0, gt_semantics=True, gt_instances=False,
               pred_semantics=False, pred_instances=False,
               gt_classwise=False, pred_classwise=False):
    self.raw_scan = raw_scan
    self.gt_scan = gt_scan
    self.pred_scan = pred_scan
    self.scan_names = scan_names

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
    
    self.selected_cls = 4

    # sanity check
    if not self.gt_semantics and self.gt_instances:
      print("Instances are only allowed in when semantics=True")
      raise ValueError
    
    if not self.pred_semantics and self.pred_instances:
      print("Instances are only allowed in when semantics=True")
      raise ValueError

    self.reset()
    self.update_scan()

  def reset(self):
    """ Reset. """
    # last key press (it should have a mutex, but visualization is not
    # safety critical, so let's do things wrong)
    self.action = "no"  # no, next, back, quit are the possibilities

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
        border_color='white', parent=self.canvas.scene)
    self.grid.add_widget(self.scan_view, 0, 0)
    self.scan_vis = visuals.Markers()
    self.scan_view.camera = 'turntable'
    self.scan_view.add(self.scan_vis)
    visuals.XYZAxis(parent=self.scan_view.scene)
    
    # add ground-truth point visualization
    if (self.gt_semantics) and (self.gt_label_names is not None):
      print("Using semantics in visualizer")
      self.sem_view = vispy.scene.widgets.ViewBox(
          border_color='white', parent=self.canvas.scene)
      self.grid.add_widget(self.sem_view, 0, 1)
      self.sem_vis = visuals.Markers()
      self.sem_view.camera = 'turntable'
      self.sem_view.add(self.sem_vis)
      visuals.XYZAxis(parent=self.sem_view.scene)
      # self.sem_view.camera.link(self.scan_view.camera)

    if (self.gt_instances) and (self.gt_label_names is not None):
      print("Using instances in visualizer")
      self.inst_view = vispy.scene.widgets.ViewBox(
          border_color='white', parent=self.canvas.scene)
      self.grid.add_widget(self.inst_view, 0, 2)
      self.inst_vis = visuals.Markers()
      self.inst_view.camera = 'turntable'
      self.inst_view.add(self.inst_vis)
      visuals.XYZAxis(parent=self.inst_view.scene)
      # self.inst_view.camera.link(self.scan_view.camera)

    # add predicted point visualization
    if (self.pred_semantics) and (self.pred_label_names is not None):
      print("Using predicted semantics in visualizer")
      self.pred_sem_view = vispy.scene.widgets.ViewBox(
          border_color='white', parent=self.canvas.scene)
      self.grid.add_widget(self.pred_sem_view, 1, 1)
      self.pred_sem_vis = visuals.Markers()
      self.pred_sem_view.camera = 'turntable'
      self.pred_sem_view.add(self.pred_sem_vis)
      visuals.XYZAxis(parent=self.pred_sem_view.scene)
      # self.pred_sem_view.camera.link(self.scan_view.camera)    
   
    if (self.pred_instances) and (self.pred_label_names is not None):
      print("Using predicted instances in visualizer")
      self.pred_inst_view = vispy.scene.widgets.ViewBox(
          border_color='white', parent=self.canvas.scene)
      self.grid.add_widget(self.pred_inst_view, 1, 2)
      self.pred_inst_vis = visuals.Markers()
      self.pred_inst_view.camera = 'turntable'
      self.pred_inst_view.add(self.pred_inst_vis)
      visuals.XYZAxis(parent=self.pred_inst_view.scene)
      # self.pred_inst_view.camera.link(self.scan_view.camera)

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
          border_color='white', parent=self.canvas2.scene)
      self.grid2.add_widget(self.gt_cls_view, 0, 0)
      self.gt_cls_vis = visuals.Markers()
      self.gt_cls_view.camera = 'turntable'
      self.gt_cls_view.add(self.gt_cls_vis)
      visuals.XYZAxis(parent=self.gt_cls_view.scene)
    
    # pred classwise
    if self.pred_classwise:
      self.pred_cls_view = vispy.scene.widgets.ViewBox(
          border_color='white', parent=self.canvas2.scene)
      self.grid2.add_widget(self.pred_cls_view, 0, 1)
      self.pred_cls_vis = visuals.Markers()
      self.pred_cls_view.camera = 'turntable'
      self.pred_cls_view.add(self.pred_cls_vis)
      visuals.XYZAxis(parent=self.pred_cls_view.scene)

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
        border_color='white', parent=self.img_canvas.scene)
    self.img_grid.add_widget(self.img_view, 0, 0)
    self.img_vis = visuals.Image(cmap='viridis')
    self.img_view.add(self.img_vis)

    # add ground-truth semantics
    if (self.gt_semantics) and (self.gt_label_names is not None):
      self.sem_img_view = vispy.scene.widgets.ViewBox(
          border_color='white', parent=self.img_canvas.scene)
      self.img_grid.add_widget(self.sem_img_view, 1, 0)
      self.sem_img_vis = visuals.Image(cmap='viridis')
      self.sem_img_view.add(self.sem_img_vis)

    # add ground-truth instances
    if (self.gt_instances) and (self.gt_label_names is not None):
      self.inst_img_view = vispy.scene.widgets.ViewBox(
          border_color='white', parent=self.img_canvas.scene)
      self.img_grid.add_widget(self.inst_img_view, 2, 0)
      self.inst_img_vis = visuals.Image(cmap='viridis')
      self.inst_img_view.add(self.inst_img_vis)
      
    
    # new canvas for multi-view img
    # self.multiview_multiplier = 1
    # self.multiview_canvas_W = 600
    # self.multiview_canvas_H = 300
    
    # self.multi_img_canvas = SceneCanvas(keys='interactive', show=True,
    #                               size=(self.canvas_W * 3, (self.canvas_H * 2) * self.multiplier))
    

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
                            size=1)
      # plot instances
      if self.gt_instances:
        self.inst_vis.set_data(self.gt_scan.points,
                              face_color=self.gt_scan.inst_label_color[..., ::-1],
                              edge_color=self.gt_scan.inst_label_color[..., ::-1],
                              size=1)

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
                            size=1)
      if self.pred_instances:
        self.pred_inst_vis.set_data(self.pred_scan.points,
                              face_color=self.pred_scan.inst_label_color[..., ::-1],
                              edge_color=self.pred_scan.inst_label_color[..., ::-1],
                              size=1)
    ###############################################################

    if self.gt_classwise:
      # can be commented out here to save time
      # first open data
      # self.gt_scan.open_scan(self.scan_names[self.offset])
      # self.gt_scan.open_label(self.gt_label_names[self.offset])
      # self.gt_scan.colorize() 
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
                          size=1)
      
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
                          size=1)

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
                           size=1)

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
    elif event.key in ['1','2','3','4','5','6','7','8','9']:
      self.selected_cls = int(str(event.key)[6])
      self.update_scan()
      print(f"You select class {self.selected_cls}...")
    elif event.key in ['C','D','E','F','G','H','I']:
      self.selected_cls = ord(str(event.key)[6]) - ord('C') + 10
      self.update_scan()
      print(f"You select class {self.selected_cls}...")
    elif event.key == 'P':
      if self.find_nearest_cls():
        self.update_scan()
        print(f"You select class {self.selected_cls}...")

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
    temp_scan = copy.deepcopy(self.gt_scan)
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
      
      # check gt
      temp_scan.open_scan(self.scan_names[current_offset])
      temp_scan.open_label(self.gt_label_names[current_offset])
      temp_sem_label = temp_scan.sem_label
      mask = (temp_sem_label == self.selected_cls)
      if mask.sum() > 14:
        self.offset = current_offset
        break
      
      # check pred
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
    
    
    
    
    
