# [nero camfront 2026-11-09] camera0 = the forward wide camera (recorded on the car as cross_left; ~120 deg fisheye).
#
# Input (one of):
#   input_mode "raw":  ~/input/image  sensor_msgs/Image from the camera driver (bgr8 / rgb8 / yuv422 / uyvy / yuyv / mono8)
#   input_mode "h265": ~/input/h265   sensor_msgs/CompressedImage with the driver's H.265 stream (decoded here with PyAV;
#                                     used by the logsim replays of recorded drives)
# Output:
#   ~/output/image        sensor_msgs/Image bgr8, the fisheye image rectified to a virtual pinhole camera (same optical frame,
#                         no rotation), one frame per lidar cycle (the frame nearest the lidar stamp phase)
#   ~/output/camera_info  sensor_msgs/CameraInfo of that pinhole image (plumb_bob, D = 0, K = P): what
#                         autoware_image_projection_based_fusion supports (it rejects the equidistant model)
# The header stamp is the camera's own stamp, unchanged; frame_id camera0/camera_optical_link.
import array
import math
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import CameraInfo, CompressedImage, Image

import cv2


def kb_project(P, fx, fy, cx, cy, k):
    x, y, z = P[..., 0], P[..., 1], P[..., 2]
    r = np.hypot(x, y)
    th = np.arctan2(r, z)
    t2 = th * th
    thd = th * (1 + t2 * (k[0] + t2 * (k[1] + t2 * (k[2] + t2 * k[3]))))
    s = np.where(r > 1e-12, thd / np.maximum(r, 1e-12), 1.0 / np.maximum(z, 1e-12))
    return fx * x * s + cx, fy * y * s + cy


def rect_maps(fx, fy, cx, cy, k, out_w, out_h, f_rect):
    cxr, cyr = (out_w - 1) / 2.0, (out_h - 1) / 2.0
    uu, vv = np.meshgrid(np.arange(out_w, dtype=np.float64), np.arange(out_h, dtype=np.float64))
    P = np.stack([(uu - cxr) / f_rect, (vv - cyr) / f_rect, np.ones_like(uu)], -1)
    mx, my = kb_project(P, fx, fy, cx, cy, k)
    m1, m2 = cv2.convertMaps(mx.astype(np.float32), my.astype(np.float32), cv2.CV_16SC2)
    return m1, m2, cxr, cyr


class Camera0Node(Node):
    def __init__(self):
        super().__init__("camera0_node")
        d = self.declare_parameter
        self.mode = d("input_mode", "raw").value
        self.frame_id = d("frame_id", "camera0/camera_optical_link").value
        fx = d("fisheye.fx", 921.7).value; fy = d("fisheye.fy", 921.7).value
        cx = d("fisheye.cx", 961.6).value; cy = d("fisheye.cy", 772.2).value
        k = [d("fisheye.k1", 0.0).value, d("fisheye.k2", 0.0).value, d("fisheye.k3", 0.0).value, d("fisheye.k4", 0.0).value]
        self.in_w = d("fisheye.width", 1920).value; self.in_h = d("fisheye.height", 1536).value
        ow = d("rectified.width", 1440).value; oh = d("rectified.height", 1080).value
        hfov = d("rectified.hfov_deg", 100.0).value
        self.period = d("cycle_sec", 0.1).value              # one frame per lidar cycle
        self.phase = d("cycle_phase_sec", -1.0).value        # fixed lidar stamp phase within the cycle; < 0: follow the topic below
        # the lidar cloud stamps (PublishedTime of the range-cropped cloud = the concatenated cloud's stamp): the phase is followed
        # with a circular mean of the last 20 clouds (the cameras drift ~20 ppm against the lidars); until known: every frame
        self.phase_topic = d("cycle_phase_topic", "/perception/obstacle_segmentation/range_cropped/pointcloud/debug/published_time").value
        self.ph_hist = []
        self.half = d("frame_half_window_sec", 1.0 / 60.0).value
        f_rect = (ow / 2.0) / math.tan(math.radians(hfov) / 2.0)
        self.m1, self.m2, cxr, cyr = rect_maps(fx, fy, cx, cy, k, ow, oh, f_rect)
        self.ow, self.oh = ow, oh
        ci = CameraInfo()
        ci.header.frame_id = self.frame_id
        ci.width, ci.height = ow, oh
        ci.distortion_model = "plumb_bob"
        ci.d = [0.0, 0.0, 0.0, 0.0, 0.0]
        ci.k = [f_rect, 0.0, cxr, 0.0, f_rect, cyr, 0.0, 0.0, 1.0]
        ci.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        ci.p = [f_rect, 0.0, cxr, 0.0, 0.0, f_rect, cyr, 0.0, 0.0, 0.0, 1.0, 0.0]
        self.ci = ci
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=5)
        qos_in = QoSProfile(reliability=ReliabilityPolicy.RELIABLE if self.mode == "h265" else ReliabilityPolicy.BEST_EFFORT,
                            history=HistoryPolicy.KEEP_LAST, depth=30)
        self.pub_img = self.create_publisher(Image, "~/output/image", qos)
        self.pub_ci = self.create_publisher(CameraInfo, "~/output/camera_info", qos)
        if self.phase < 0 and self.phase_topic:
            from autoware_internal_msgs.msg import PublishedTime
            self.create_subscription(PublishedTime, self.phase_topic, self.on_cloud_stamp,
                                     QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=5))
        if self.mode == "h265":
            import av  # noqa: delayed import, only the replay path needs it
            self.av = av
            self.ctx = av.CodecContext.create("hevc", "r")
            self.ctx.thread_type = "SLICE"   # frame threading (AUTO) buffers ~15 frames = 500 ms of delay: never
            self.started = False
            self.pending = {}
            self.n = 0
            self.create_subscription(CompressedImage, "~/input/h265", self.on_h265, qos_in)
        else:
            self.create_subscription(Image, "~/input/image", self.on_raw, qos_in)
        self.stat_t = time.monotonic(); self.stat_n = 0; self.stat_ms = 0.0
        self.get_logger().info(
            f"[NERO_CAMERA0] mode {self.mode}, fisheye f {fx:.1f}/{fy:.1f} c ({cx:.1f}, {cy:.1f}) k {k}; rectified {ow}x{oh} "
            f"hfov {hfov:.1f} f {f_rect:.1f}; cycle {self.period} s phase {self.phase if self.phase >= 0 else self.phase_topic}")

    # ---- frame selection: the frame nearest the lidar stamp phase (camera 30 Hz, lidar 10 Hz)
    def on_cloud_stamp(self, m):
        t = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
        a = 2 * math.pi * ((t % self.period) / self.period)
        self.ph_hist.append((math.cos(a), math.sin(a)))
        self.ph_hist = self.ph_hist[-20:]

    def cur_phase(self):
        if self.phase >= 0:
            return self.phase
        if len(self.ph_hist) < 3:
            return None
        c = sum(x for x, _ in self.ph_hist); s = sum(y for _, y in self.ph_hist)
        return (math.atan2(s, c) / (2 * math.pi)) % 1.0 * self.period

    def wanted(self, stamp):
        ph = self.cur_phase()
        if ph is None:
            return True
        t = stamp.sec + stamp.nanosec * 1e-9
        dphi = (t - ph) % self.period
        dphi = min(dphi, self.period - dphi)
        return dphi <= self.half

    def publish(self, bgr, stamp):
        t0 = time.monotonic()
        rect = cv2.remap(bgr, self.m1, self.m2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        msg = Image()
        msg.header.stamp = stamp
        msg.header.frame_id = self.frame_id
        msg.height, msg.width = rect.shape[:2]
        msg.encoding = "bgr8"
        msg.is_bigendian = 0
        msg.step = msg.width * 3
        msg.data = array.array("B", rect.tobytes())   # rclpy fast path (bytes are checked element by element)
        self.ci.header.stamp = stamp
        self.pub_img.publish(msg)
        self.pub_ci.publish(self.ci)
        self.stat_n += 1; self.stat_ms += (time.monotonic() - t0) * 1e3
        if time.monotonic() - self.stat_t > 10.0:
            ph = self.cur_phase()
            self.get_logger().info(f"[NERO_CAMERA0] {self.stat_n} frames in 10 s, rectify+publish {self.stat_ms / max(self.stat_n, 1):.1f} ms, "
                                   f"lidar phase {'-' if ph is None else f'{ph * 1e3:.1f} ms'}")
            self.stat_t = time.monotonic(); self.stat_n = 0; self.stat_ms = 0.0

    def on_raw(self, m):
        if not self.wanted(m.header.stamp):
            return
        buf = np.frombuffer(m.data, dtype=np.uint8)
        enc = m.encoding.lower()
        if enc == "bgr8":
            img = buf.reshape(m.height, m.step // 3 if m.step else m.width, 3)[:, :m.width]
        elif enc == "rgb8":
            img = cv2.cvtColor(buf.reshape(m.height, m.width, 3), cv2.COLOR_RGB2BGR)
        elif enc in ("yuv422", "uyvy"):
            img = cv2.cvtColor(buf.reshape(m.height, m.width, 2), cv2.COLOR_YUV2BGR_UYVY)
        elif enc in ("yuv422_yuy2", "yuyv"):
            img = cv2.cvtColor(buf.reshape(m.height, m.width, 2), cv2.COLOR_YUV2BGR_YUYV)
        elif enc == "mono8":
            img = cv2.cvtColor(buf.reshape(m.height, m.width), cv2.COLOR_GRAY2BGR)
        else:
            self.get_logger().error(f"[NERO_CAMERA0] unsupported encoding {m.encoding}", throttle_duration_sec=10.0)
            return
        self.publish(img, m.header.stamp)

    def on_h265(self, m):
        buf = bytes(m.data)
        if not self.started:
            if not _is_key(buf):
                return
            self.started = True
        pk = self.av.Packet(buf)
        pk.pts = self.n
        self.pending[self.n] = m.header.stamp
        self.n += 1
        try:
            frames = self.ctx.decode(pk)
        except Exception as e:  # a broken frame: wait for the next keyframe
            self.get_logger().warn(f"[NERO_CAMERA0] decode error {e}: waiting for a keyframe", throttle_duration_sec=5.0)
            self.ctx = self.av.CodecContext.create("hevc", "r"); self.started = False; self.pending.clear()
            return
        for fr in frames:
            st = self.pending.pop(fr.pts, None)
            if st is None or not self.wanted(st):
                continue
            self.publish(fr.to_ndarray(format="bgr24"), st)


def _is_key(buf):
    i, L, t = 0, min(len(buf), 4096), []
    while True:
        j = buf.find(b"\x00\x00\x01", i, L)
        if j < 0 or j + 3 >= L:
            break
        t.append((buf[j + 3] >> 1) & 0x3F)
        i = j + 3
    return any(x in (19, 20, 21) for x in t) and 32 in t


def main(args=None):
    rclpy.init(args=args)
    n = Camera0Node()
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    n.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
