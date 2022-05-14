#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt

import csv
import math
import random
import sys
import itertools

import rospy
from std_msgs.msg import String

import geometry_msgs as gm
from geometry_msgs.msg import PoseStamped

BOAST_CSV_PATH = 'boat.csv' # path
RADIUS = 1.5 # meters
MIN_RADIUS = 0.3 # meters
MIN_SELECTION_RADIUS = 0.3
NORMAL_OFFSET = 0.05 # meters

ROBOT_X = 0
ROBOT_Y = 0
ROBOT_Z = 0 # meters

class Point:
	def __init__(self, x, y, z):
		self.x = float(x)
		self.y = float(y)
		self.z = float(z)
		
	def magnitude(self):
		return math.sqrt((self.x ** 2) + (self.y ** 2) + (self.z ** 2))
		
	def cross(self, other):
		if isinstance(other, Point):
			x = (self.y * other.z) - (self.z * other.y)
			y = (self.z * other.x) - (self.x * other.z)
			z = (self.x * other.y) - (self.y * other.x)
			return Point(x, y, z)
		else:
			raise Exception(f'unexpected type in Point.cross: {type(other)}')
			
	def dot(self, other):
		if isinstance(other, Point):
			return (self.x * other.x) + (self.y * other.y) + (self.z * other.z)
		else:
			raise Exception(f'unexpected type in Point.cross: {type(other)}')
		
	def __add__(self, other):
		if isinstance(other, Point):
			return Point(self.x + other.x, self.y + other.y, self.z + other.z)
		else:
			raise Exception(f'unexpected type in Point.__add__: {type(other)}')
			
	def __sub__(self, other):
		if isinstance(other, Point):
			return Point(self.x - other.x, self.y - other.y, self.z - other.z)
		else:
			raise Exception(f'unexpected type in Point.__sub__: {type(other)}')
			
	def __mul__(self, other):
		if isinstance(other, float):
			return Point(self.x * other, self.y * other, self.z * other)
		else:
			raise Exception(f'unexpected type in Point.__mul__: {type(other)}')
			
	def __eq__(self, other):
		if isinstance(other, Point):
			# TODO: Consider comparing with range, `(x1 - x2).abs() < .0001`
			return self.x == other.x and self.y == other.y and self.z == other.z
		else:
			raise Exception(f'unexpected type in Point.__eq__: {type(other)}')
		
	def __repr__(self):
		return f'Point(x={self.x}, y={self.y}, z={self.z})'
		
class Quaternion:
	def __init__(self, x, y, z, w):
		self.x = float(x)
		self.y = float(y)
		self.z = float(z)
		self.w = float(w)
		
	def get_direction_vector(self):
		x = 2 * (self.x * self.z - self.w * self.y)
		y = 2 * (self.y * self.z + self.w * self.x)
		z = 1 - 2 * (self.x * self.x + self.y * self.y)
		return Point(x, y, z)

	def magnitude(self):
		return math.sqrt((self.x ** 2) + (self.y ** 2) + (self.z ** 2) + (self.w ** 2))

	def normalize(self):
		d = self.magnitude()
		self.x /= d
		self.y /= d
		self.z /= d
		self.w /= d
		
	def __repr__(self):
		return f'Quaternion(x={self.x}, y={self.y}, z={self.z}, w={self.w})'

def load_boat_data():
	ret = []
	with open(BOAST_CSV_PATH) as csvfile:
		csvreader = csv.reader(csvfile)
		next(csvreader, None)
		for row in csvreader:
			point = Point(row[0], row[1], row[2])
			normal = Quaternion(row[3], row[4], row[5], row[6])
			ret.append((point, normal))
	return ret
	
def quaternion_to_axis_angle(quaternion):
	angle = 2 * math.acos(quaternion.w)
	x = quaternion.x / math.sqrt(1 - quaternion.w * quaternion.w)
	y = quaternion.y / math.sqrt(1 - quaternion.w * quaternion.w)
	z = quaternion.z / math.sqrt(1 - quaternion.w * quaternion.w)
	return angle, Point(x, y, z)
	
def rotate_vector(vector, angle, unit_vector):
	return (vector * math.cos(angle)) + (vector.cross(unit_vector) * math.sin(angle)) + (unit_vector * unit_vector.dot(vector) * (1 - math.cos(angle)))
	
def visualize_points(points, groups=[], group_radius=1, normalized_boat_points=[]):
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	xs = []
	ys = []
	zs = []
	for point in points:
		xs.append(point.x)
		ys.append(point.y)
		zs.append(point.z)
	ax.scatter(xs,ys,zs)
	
	
	norm_xs = []
	norm_ys = []
	norm_zs = []
	for point in normalized_boat_points:
		norm_xs.append(point.x)
		norm_ys.append(point.y)
		norm_zs.append(point.z)
	ax.scatter(norm_xs, norm_ys, norm_zs)
	
	normal_line_xs = []
	normal_line_ys = []
	normal_line_zs = []
	for i in range(len(normalized_boat_points)):
		point = points[i]
		norm_point = normalized_boat_points[i]
		
		normal_line_xs.append([point.x, norm_point.x])
		normal_line_ys.append([point.y, norm_point.y])
		normal_line_zs.append([point.z, norm_point.z])
	
	for i in range(len(normal_line_xs)):
		ax.plot(normal_line_xs[i], normal_line_ys[i], normal_line_zs[i])
	
	for (group_point, points) in groups:
		r = group_radius
		u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
		x = r * (np.cos(u)*np.sin(v)) + group_point.x
		y = r * (np.sin(u)*np.sin(v)) + group_point.y
		z = r * np.cos(v)
		ax.plot_wireframe(x, y, z, color="r")
	
	ax.set_xlim(0, 4)
	ax.set_ylim(0, 4)
	ax.set_zlim(0, 4)
	
	plt.show()
	
def inside_point_cloud(point_cloud, test_point):
	has_gt_x = False
	has_lt_x = False
	has_gt_y = False
	has_lt_y = False
	
	for point in point_cloud:
		if point.x < test_point.x:
			has_lt_x = True
		if point.x > test_point.x:
			has_gt_x = True
		if point.y < test_point.y:
			has_lt_y = True
		if point.y > test_point.y:
			has_gt_y = True
	return has_gt_x and has_lt_x and has_gt_y and has_lt_y
	
def get_point_cloud_bb(point_cloud):
	min_x = point_cloud[0].x
	max_x = point_cloud[0].x
	
	max_y = point_cloud[0].y
	min_y = point_cloud[0].y
	
	for point in point_cloud:
		if point.x < min_x:
			min_x = point.x
		if point.x > max_x:
			max_x = point.x
			
		if point.y < min_y:
			min_y = point.y
		if point.y > max_y:
			max_y = point.y
		
	return (min_x, min_y, max_x, max_y)

	
def normalize_boat_data(boat_data):
	ret = []
	for (point, rot) in boat_data:
		ret.append(point - (rot.get_direction_vector() * NORMAL_OFFSET))
	return ret

def calc_dist_xyz(point1, point2):
	diff_x = point1.x - point2.x
	diff_y = point1.y - point2.y
	diff_z = point1.z - point2.z
	return math.sqrt((diff_x ** 2) + (diff_y ** 2) + (diff_z ** 2))

def get_points_in_radius(points, test_point, radius):
	return [point for point in points if calc_dist_xyz(point, test_point) < radius]

def group_points(point_cloud, radius, min_radius):
	bb = get_point_cloud_bb(point_cloud)
	bb_ext = 1 * radius
	bb = (bb[0] - bb_ext, bb[1] - bb_ext, bb[2] + bb_ext, bb[3] + bb_ext)
	
	groups = []
	
	remaining_point_cloud = point_cloud.copy()
	while len(remaining_point_cloud) > 0:
		rand_x = random.uniform(bb[0], bb[2])
		rand_y = random.uniform(bb[1], bb[3])
		rand_point = Point(rand_x, rand_y, ROBOT_Z)
		
		if inside_point_cloud(point_cloud, rand_point):
			continue
			
		if any(calc_dist_xyz(point, rand_point) < MIN_SELECTION_RADIUS for point, target_points in groups):
			continue
		
		group = get_points_in_radius(remaining_point_cloud, rand_point, radius)
		if len(get_points_in_radius(point_cloud, rand_point, min_radius)) > 0:
			continue
		
		if len(group) == 0:
			continue
		
		groups.append((rand_point, group))
		
		for point in group:
			remaining_point_cloud.remove(point)
			
	return groups
	
def travellingSalesmanPoints(initial_point, points):
	num_points = 1 + len(points)
	
	graph = [None] * num_points
	for i in range(num_points):
		graph[i] = ([None] * num_points)
		
		for j in range(num_points):
			point1 = points[i] if i != len(points) else initial_point
			point2 = points[j] if j != len(points) else initial_point
			graph[i][j] = calc_dist_xyz(point1, point2)
	
	verticies = list(range(len(points)))
	
	min_cost_path = sys.maxsize
	min_path = None
	for permutation in itertools.permutations(verticies):
		path_cost = 0
		path = []
		
		k = len(points)
		for i in permutation:
			path_cost += graph[k][i]
			path.append(i)
			k = i
		path_cost += graph[k][len(points)]
		path.append(len(points))
		
		if min_cost_path > path_cost:
			min_cost_path = path_cost
			min_path = path
		
	return min_path
	
def run_ros_node(instructions):
	nav_pub = rospy.Publisher('/rviz_2d_nav_goal', PoseStamped, queue_size=1)
	end_effector_pub = rospy.Publisher('/endeffector_goal', gm.msg.PointStamped, queue_size=1)
	rospy.init_node('project', anonymous=True)
	rate = rospy.Rate(10)

	requested_position, end_effector_points = None, []
	seq_id = 0
	last_time = rospy.get_rostime()
	sleep_time = 10
	while not rospy.is_shutdown():
		if rospy.get_rostime().secs - last_time.secs < sleep_time:
			rate.sleep()
			continue
			
		print('Ros loop tick')
		if requested_position is None and len(end_effector_points) == 0: 
			if len(instructions) == 0:
				break
			requested_position, end_effector_points = instructions.pop()

		if requested_position is not None:
			target_point = Point(end_effector_points[0].x, end_effector_points[0].y, 0)
			q = Quaternion(0, 0, 0, 0)
			a = requested_position.cross(target_point)
			q.x = a.x
			q.y = a.y
			q.z = a.z
			q.w = math.sqrt((requested_position.magnitude() ** 2) * (target_point.magnitude() ** 2)) + requested_position.dot(target_point)
			q.normalize()
	
			print(f'Moving to {requested_position} @ orientation {q}')
			pos_msg = PoseStamped()
			pos_msg.header.stamp = rospy.Time(0)
			pos_msg.header.frame_id = 'map'
			pos_msg.header.seq = seq_id
			pos_msg.pose.position = gm.msg.Point(requested_position.x, requested_position.y, requested_position.z)
			pos_msg.pose.orientation = gm.msg.Quaternion(q.x, q.y, q.z, q.w)

			nav_pub.publish(pos_msg)
			rate.sleep()
			requested_position = None
			seq_id += 1
			last_time = rospy.get_rostime()
			sleep_time = 30
		elif len(end_effector_points) != 0:
			point = end_effector_points.pop()

			print(f'Moving end effector to {point}')
		
			msg = gm.msg.PointStamped()
			msg.header.seq = seq_id
			msg.header.frame_id = 'map'
			msg.point = gm.msg.Point(point.x, point.y, point.z)

			end_effector_pub.publish(msg)
			rate.sleep()
			seq_id += 1
			last_time = rospy.get_rostime()
			sleep_time = 5
	
		rate.sleep()

def main():
	robot_location = Point(ROBOT_X, ROBOT_Y, ROBOT_Z)
	
	print('Loading boat data...')
	boat_data = load_boat_data()
	print(f'Loaded {len(boat_data)} boat data points')

	boat_points = [point for (point, rot) in boat_data]
	normalized_boat_points = normalize_boat_data(boat_data)

	print(f'Grouping points with radius={RADIUS} and min_radius={MIN_RADIUS}')
	point_groups = group_points(normalized_boat_points, RADIUS, MIN_RADIUS)
	print(f'Generated {len(point_groups)} point groups')
	
	points_point_groups = [group_point for group_point, points in point_groups]
	
	point_group_order = travellingSalesmanPoints(robot_location, points_point_groups)
	# print(point_group_order)
	
	instructions = []
	for index in point_group_order[:-1]:
		instructions.append(point_groups[index])
		
	instructions.reverse()
	print(instructions)

	# visualize_points(boat_points, point_groups, RADIUS, normalized_boat_points=normalized_boat_points)
	
	run_ros_node(instructions)
	
if __name__ == "__main__":
	main()
