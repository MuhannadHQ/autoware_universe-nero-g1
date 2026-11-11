# [nero camfront 2026-11-09] camera boxes allowed to LABEL lidar clusters: only the classes whose label makes the stack more careful
# (PEDESTRIAN by default; the param file says why not two-wheelers / vehicles). Vehicle boxes stay out of the cluster labelling: YOLOX calls the street's green
# garbage containers CAR at 3-4 m (11-03 13:04:57, logsim CF2), and a CAR label shrinks obstacle_stop's lateral margin (0.3 -> 0.1 m)
# and turns an obstacle into a "parked vehicle" for static avoidance. The full rois stay on rois0 for the confirm-only node.
# Also dropped: boxes with a score < min_score and boxes cut by the image border (border_px): a parked car's front cut by the left edge
# was a MOTORCYCLE at 0.39 (11-08 14:25:24.9, CF4 g8a); people score 0.75 (median of 410 boxes).
from rclpy.node import Node
import rclpy
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from tier4_perception_msgs.msg import DetectedObjectsWithFeature

LABELS = {"UNKNOWN": 0, "CAR": 1, "TRUCK": 2, "BUS": 3, "TRAILER": 4, "MOTORCYCLE": 5, "BICYCLE": 6, "PEDESTRIAN": 7}


class RoiClassFilter(Node):
    def __init__(self):
        super().__init__("roi_class_filter")
        keep = self.declare_parameter("keep_classes", ["PEDESTRIAN"]).value
        self.keep = {LABELS[k] for k in keep}
        self.min_score = self.declare_parameter("min_score", 0.5).value
        self.border = self.declare_parameter("border_px", 4).value
        self.w = self.declare_parameter("image_width", 1440).value
        self.h = self.declare_parameter("image_height", 1080).value
        q = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=5)
        self.pub = self.create_publisher(DetectedObjectsWithFeature, "~/output/rois", q)
        self.create_subscription(DetectedObjectsWithFeature, "~/input/rois", self.cb, q)
        self.get_logger().info(f"[NERO_CAMERA0] roi_class_filter keeps {sorted(keep)}, score >= {self.min_score}, "
                               f"not within {self.border} px of the {self.w}x{self.h} border")

    def cb(self, m):
        out = DetectedObjectsWithFeature()
        out.header = m.header
        b, w, h = self.border, self.w, self.h

        def ok(fo):
            r = fo.feature.roi
            return (fo.object.classification and fo.object.classification[0].label in self.keep
                    and fo.object.existence_probability >= self.min_score
                    and r.x_offset > b and r.y_offset > b and r.x_offset + r.width < w - b and r.y_offset + r.height < h - b)
        out.feature_objects = [fo for fo in m.feature_objects if ok(fo)]
        self.pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    n = RoiClassFilter()
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    n.destroy_node()
    rclpy.try_shutdown()
