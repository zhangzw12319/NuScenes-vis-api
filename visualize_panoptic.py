#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import argparse
import os
import yaml
import pickle


from auxiliary.laserscan import LaserScan, SemLaserScan
from auxiliary.laserscanvis import LaserScanVis
from nuscenes import NuScenes

if __name__ == '__main__':
	parser = argparse.ArgumentParser("./visualize.py")
	parser.add_argument(
		'--dataset', '-d',
		type=str,
		required=True,
		help='Dataset to visualize. No Default',
	)
	parser.add_argument(
		'--config', '-c',
		type=str,
		required=False,
		default="config/nuscenes.yaml",
		help='Dataset config file. Defaults to %(default)s',
	)
	
	parser.add_argument(
		'--version', '-v',
		type=str,
		default="v1.0-mini",
		required=True,
		help='Which split v1.0-trainval|v1.0-mini|v1.0-test set to visualize. Defaults to %(default)s',
		choices=["v1.0-mini", "v1.0-trainval","v1.0-test"]
	)
	
	parser.add_argument(
	  '--split', '-s',
	  type=str,
	  default="val",
	  required=True,
	  help="train or validation set",
	  choices=["train", "val"]
	)
	
	parser.add_argument(
		'--predictions', '-p',
		type=str,
		default=None,
		required=False,
		help='Alternate location for labels, to use predictions folder. '
		'Must point to directory containing the predictions in the proper format '
		' (see readme)'
		'Defaults to %(default)s',
	)
	
	parser.add_argument(
		'--ignore_semantics', '-i',
		dest='ignore_semantics',
		default=False,
		action='store_true',
		help='Ignore semantics. Visualizes uncolored pointclouds.'
		'Defaults to %(default)s',
	)
	
	parser.add_argument(
		'--do_instances', '-di',
		dest='do_instances',
		default=False,
		action='store_true',
		help='Visualize instances too. Defaults to %(default)s',
	)
	
	parser.add_argument(
		'--ignore_safety',
		dest='ignore_safety',
		default=False,
		action='store_true',
		help='Normally you want the number of labels and ptcls to be the same,'
		', but if you are not done inferring this is not the case, so this disables'
		' that safety.'
		'Defaults to %(default)s',
	)
 
	parser.add_argument(
		'--pkl_path',
		type=str,
		help="For nuscenes, the directory to preprocessed pkl path file"
	)
	
	FLAGS, unparsed = parser.parse_known_args()
	
	# print summary of what we will do
	print("*" * 80)
	print("INTERFACE:")
	print("Dataset", FLAGS.dataset)
	print("Config", FLAGS.config)
	print("version", FLAGS.version)
	print("split", FLAGS.split)
	print("Predictions", FLAGS.predictions)
	print("ignore_semantics", FLAGS.ignore_semantics)
	print("do_instances", FLAGS.do_instances)
	print("ignore_safety", FLAGS.ignore_safety)
	print("*" * 80)
	
	
	# open config file
	try:
		print("Opening config file %s" % FLAGS.config)
		CFG = yaml.safe_load(open(FLAGS.config, 'r', errors='ignore'))
	except Exception as e:
		print(e)
		print("Error opening yaml file.")
		quit()
	
	# prepare nuscenes data
	nusc = NuScenes(version=FLAGS.version, dataroot=FLAGS.dataset, verbose=True)
	pkl_path = FLAGS.pkl_path
	if FLAGS.version == "v1.0-trainval" or FLAGS.version == "v1.0-mini":
		if FLAGS.split == 'train':
			pkl_path = os.path.join(pkl_path,  "nuscenes_infos_train.pkl")    
		else:
			pkl_path = os.path.join(pkl_path,  "nuscenes_infos_val.pkl")
	else:
		pkl_path = os.path.join(pkl_path, "nuscenes_infos_test.pkl")
	with open(pkl_path, 'rb') as f:
		data = pickle.load(f)
	nusc_infos = data['infos']	
	
	# collect lidar token list and sort
	sample_token_list = [entry["token"] for entry in nusc_infos]
	lidar_token_list = [nusc.get('sample', sample_token)['data']['LIDAR_TOP'] for sample_token in sample_token_list]
	# get lidar point file path
	scan_names = [os.path.join(nusc.dataroot, nusc.get('sample_data', lidar_token)['filename'] ) for lidar_token in lidar_token_list]

	# collect labels
	if not FLAGS.ignore_semantics:
		# collect predicted labels if there is a prediction folder
		if FLAGS.predictions is not None:
			label_paths = os.path.join(FLAGS.predictions)
			# open as npz file
			panoptic_pred_names = [os.path.join(label_paths, "%s_panoptic.npz" % token) for token in lidar_token_list]
			if not FLAGS.ignore_safety:
				assert(len(panoptic_pred_names) == len(lidar_token_list))
		
  		# collect ground-truth labels if verision is train|val
		if FLAGS.version != "v1.0-test":
			panoptic_gt_names = [
				os.path.join(nusc.dataroot, nusc.get('panoptic', token)['filename']) for token in lidar_token_list]
			if not FLAGS.ignore_safety:
				assert(len(panoptic_gt_names) == len(lidar_token_list))


	# create scans
		raw_scan = LaserScan(project=True)
		if FLAGS.ignore_semantics:
			if FLAGS.version != "v1.0-test":
				gt_scan = LaserScan(project=True)  # project all opened scans to spheric proj
			else:
				gt_scan = None
			if FLAGS.predictions is not None:
				pred_scan = LaserScan(project=True)
			else:
				pred_scan = None
		else:
			color_dict = CFG["color_map_17"]
			nclasses = len(color_dict)
			if FLAGS.version != "v1.0-test":
				gt_scan = SemLaserScan(nclasses, color_dict, project=True, learning_map=CFG["learning_map"])
			else:
				gt_scan = None
			if FLAGS.predictions is not None:
				pred_scan = SemLaserScan(nclasses, color_dict, project=True, learning_map=CFG["learning_map"])
			else:
				pred_scan = None

	# create a visualizer
	gt_semantics = not FLAGS.ignore_semantics and (FLAGS.version != "v1.0-test")
	gt_instances = gt_semantics and FLAGS.do_instances
	pred_semantics = not FLAGS.ignore_semantics and (FLAGS.predictions is not None)
	pred_instances = pred_semantics and FLAGS.do_instances

	if gt_semantics:
		gt_label_names = panoptic_gt_names
	else:		
		gt_label_names = None
	if pred_semantics:
		pred_label_names = panoptic_pred_names
	else:
		pred_label_names = None

	
	vis = LaserScanVis(raw_scan=raw_scan, gt_scan=gt_scan, pred_scan=pred_scan,
						scan_names=scan_names,
						gt_label_names=gt_label_names,
						pred_label_names=pred_label_names,
						offset=0,
						gt_semantics=gt_semantics, gt_instances=gt_instances,
						pred_semantics=pred_semantics, pred_instances=pred_instances,
						gt_classwise=False, pred_classwise=False)

	# print instructions
	print("To navigate:")
	print("\tb: back (previous scan)")
	print("\tn: next (next scan)")
	print("\tq: quit (exit program)")

	# run the visualizer
	vis.run()