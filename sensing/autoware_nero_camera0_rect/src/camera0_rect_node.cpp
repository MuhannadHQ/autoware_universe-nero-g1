// [nero camstack 2026-11-10] camera0 = the forward wide camera (recorded on the car as cross_left; ~120 deg fisheye), C++ image path
// (CHAIN CAMSETUP (1): no Python image path at 30 Hz on the Orin). Same outputs and parameters as autoware_nero_camera0's Python
// camera0_node (camfront_1109), which it replaces in the launch:
//   input_mode "raw":  ~/input/image  sensor_msgs/Image from the camera driver (bgr8 / rgb8 / yuv422 (uyvy) / yuyv / mono8 / nv12)
//   input_mode "h265": ~/input/h265   sensor_msgs/CompressedImage with the driver's H.265 stream, decoded with libavcodec on a worker
//                                     thread (decoder by name: "hevc" = software; e.g. "hevc_cuvid" / "hevc_nvv4l2dec" where built)
//   ~/output/image        sensor_msgs/Image bgr8, the fisheye image rectified to a virtual pinhole camera (same optical frame, no rotation),
//                         ONE frame per lidar cycle (the frame nearest the lidar stamp phase); header stamp = the camera's own stamp
//   ~/output/camera_info  sensor_msgs/CameraInfo of that pinhole image (plumb_bob, D = 0, K = P)
// Differences from the Python node (by design): frames that are not wanted are decoded but never converted / rectified / copied; the
// subscription callback only queues packets (the decoder runs on its own thread), so a slow frame never blocks the queue; a stats line
// every 10 s gives the decode and rectify times and how late the published frames are (wall clock, receive -> publish).
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <autoware_internal_msgs/msg/published_time.hpp>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
}

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace autoware::nero_camera0_rect
{
using sensor_msgs::msg::CameraInfo;
using sensor_msgs::msg::CompressedImage;
using sensor_msgs::msg::Image;
using Clock = std::chrono::steady_clock;

// Kannala-Brandt (equidistant) projection of rays (x, y, z) -> fisheye pixels; the same arithmetic (double) as the Python node.
static void kb_project(double x, double y, double z, double fx, double fy, double cx, double cy, const double k[4], double & u, double & v)
{
  const double r = std::hypot(x, y);
  const double th = std::atan2(r, z);
  const double t2 = th * th;
  const double thd = th * (1 + t2 * (k[0] + t2 * (k[1] + t2 * (k[2] + t2 * k[3]))));
  const double s = r > 1e-12 ? thd / std::max(r, 1e-12) : 1.0 / std::max(z, 1e-12);
  u = fx * x * s + cx;
  v = fy * y * s + cy;
}

static bool is_hevc_key(const uint8_t * buf, size_t n)
{
  // an IRAP picture (NAL 19 / 20 / 21) together with a VPS (32) in the first 4 KB: the decoder can start here
  bool irap = false, vps = false;
  const size_t L = std::min<size_t>(n, 4096);
  for (size_t i = 0; i + 3 < L; ++i) {
    if (buf[i] == 0 && buf[i + 1] == 0 && buf[i + 2] == 1) {
      const int t = (buf[i + 3] >> 1) & 0x3F;
      irap |= (t == 19 || t == 20 || t == 21);
      vps |= (t == 32);
      i += 2;
    }
  }
  return irap && vps;
}

class Camera0RectNode : public rclcpp::Node
{
public:
  explicit Camera0RectNode(const rclcpp::NodeOptions & options) : Node("camera0_node", options)
  {
    mode_ = declare_parameter<std::string>("input_mode", "raw");
    frame_id_ = declare_parameter<std::string>("frame_id", "camera0/camera_optical_link");
    const double fx = declare_parameter<double>("fisheye.fx", 921.7), fy = declare_parameter<double>("fisheye.fy", 921.7);
    const double cx = declare_parameter<double>("fisheye.cx", 961.6), cy = declare_parameter<double>("fisheye.cy", 772.2);
    const double k[4] = {
      declare_parameter<double>("fisheye.k1", 0.0), declare_parameter<double>("fisheye.k2", 0.0),
      declare_parameter<double>("fisheye.k3", 0.0), declare_parameter<double>("fisheye.k4", 0.0)};
    in_w_ = declare_parameter<int>("fisheye.width", 1920);
    in_h_ = declare_parameter<int>("fisheye.height", 1536);
    const int ow = declare_parameter<int>("rectified.width", 1440), oh = declare_parameter<int>("rectified.height", 1080);
    const double hfov = declare_parameter<double>("rectified.hfov_deg", 100.0);
    period_ = declare_parameter<double>("cycle_sec", 0.1);
    phase_ = declare_parameter<double>("cycle_phase_sec", -1.0);
    const auto phase_topic = declare_parameter<std::string>(
      "cycle_phase_topic", "/perception/obstacle_segmentation/range_cropped/pointcloud/debug/published_time");
    half_ = declare_parameter<double>("frame_half_window_sec", 1.0 / 60.0);
    decoder_name_ = declare_parameter<std::string>("decoder", "hevc");
    decoder_threads_ = declare_parameter<int>("decoder_threads", 2);
    queue_max_ = declare_parameter<int>("queue_max", 60);

    // the rectification maps: identical to the Python node (double rays -> float32 maps -> CV_16SC2)
    const double f_rect = (ow / 2.0) / std::tan(hfov * M_PI / 180.0 / 2.0);
    const double cxr = (ow - 1) / 2.0, cyr = (oh - 1) / 2.0;
    cv::Mat mx(oh, ow, CV_32FC1), my(oh, ow, CV_32FC1);
    for (int v = 0; v < oh; ++v) {
      for (int u = 0; u < ow; ++u) {
        double pu, pv;
        kb_project((u - cxr) / f_rect, (v - cyr) / f_rect, 1.0, fx, fy, cx, cy, k, pu, pv);
        mx.at<float>(v, u) = static_cast<float>(pu);
        my.at<float>(v, u) = static_cast<float>(pv);
      }
    }
    cv::convertMaps(mx, my, map1_, map2_, CV_16SC2);
    ow_ = ow;
    oh_ = oh;
    ci_.header.frame_id = frame_id_;
    ci_.width = ow;
    ci_.height = oh;
    ci_.distortion_model = "plumb_bob";
    ci_.d = {0.0, 0.0, 0.0, 0.0, 0.0};
    ci_.k = {f_rect, 0.0, cxr, 0.0, f_rect, cyr, 0.0, 0.0, 1.0};
    ci_.r = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
    ci_.p = {f_rect, 0.0, cxr, 0.0, 0.0, f_rect, cyr, 0.0, 0.0, 0.0, 1.0, 0.0};

    const auto qos = rclcpp::QoS(rclcpp::KeepLast(5)).best_effort();
    auto qos_in = rclcpp::QoS(rclcpp::KeepLast(30));
    if (mode_ == "h265") qos_in.reliable(); else qos_in.best_effort();
    pub_img_ = create_publisher<Image>("~/output/image", qos);
    pub_ci_ = create_publisher<CameraInfo>("~/output/camera_info", qos);
    if (phase_ < 0 && !phase_topic.empty()) {
      sub_phase_ = create_subscription<autoware_internal_msgs::msg::PublishedTime>(
        phase_topic, rclcpp::QoS(rclcpp::KeepLast(5)).reliable(),
        [this](autoware_internal_msgs::msg::PublishedTime::ConstSharedPtr m) { on_cloud_stamp(*m); });
    }
    if (mode_ == "h265") {
      open_decoder();
      sub_h265_ = create_subscription<CompressedImage>(
        "~/input/h265", qos_in, [this](CompressedImage::UniquePtr m) { on_h265(std::move(m)); });
      worker_ = std::thread([this] { decode_loop(); });
    } else {
      sub_raw_ = create_subscription<Image>(
        "~/input/image", qos_in, [this](Image::ConstSharedPtr m) { on_raw(m); });
    }
    stat_t_ = Clock::now();
    RCLCPP_INFO(
      get_logger(),
      "[NERO_CAMERA0] C++ node, mode %s (decoder %s, %d threads), fisheye f %.1f/%.1f c (%.1f, %.1f) k [%g, %g, %g, %g]; rectified %dx%d "
      "hfov %.1f f %.1f; cycle %.3f s phase %s",
      mode_.c_str(), decoder_name_.c_str(), decoder_threads_, fx, fy, cx, cy, k[0], k[1], k[2], k[3], ow, oh, hfov, f_rect, period_,
      phase_ >= 0 ? std::to_string(phase_).c_str() : phase_topic.c_str());
  }

  ~Camera0RectNode() override
  {
    {
      std::lock_guard<std::mutex> lk(q_mtx_);
      stop_ = true;
    }
    q_cv_.notify_all();
    if (worker_.joinable()) worker_.join();
    if (sws_) sws_freeContext(sws_);
    if (frame_) av_frame_free(&frame_);
    if (pkt_) av_packet_free(&pkt_);
    if (ctx_) avcodec_free_context(&ctx_);
  }

private:
  struct Packet
  {
    std::vector<uint8_t> data;
    builtin_interfaces::msg::Time stamp;
    Clock::time_point recv;
  };

  // ---- frame selection: the frame nearest the lidar stamp phase (camera 30 Hz, lidar 10 Hz)
  void on_cloud_stamp(const autoware_internal_msgs::msg::PublishedTime & m)
  {
    const double t = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9;
    const double a = 2 * M_PI * (std::fmod(t, period_) / period_);
    std::lock_guard<std::mutex> lk(ph_mtx_);
    ph_hist_.emplace_back(std::cos(a), std::sin(a));
    if (ph_hist_.size() > 20) ph_hist_.pop_front();
  }
  double cur_phase()
  {
    if (phase_ >= 0) return phase_;
    std::lock_guard<std::mutex> lk(ph_mtx_);
    if (ph_hist_.size() < 3) return -1.0;
    double c = 0, s = 0;
    for (const auto & p : ph_hist_) {
      c += p.first;
      s += p.second;
    }
    double ph = std::atan2(s, c) / (2 * M_PI);
    ph = ph - std::floor(ph);
    return ph * period_;
  }
  bool wanted(const builtin_interfaces::msg::Time & st)
  {
    // the input frame period (EMA of stamp steps): a driver publishing at the lidar rate or slower (publish_rate_divisor) has one frame
    // per cycle at an arbitrary phase -> every frame is taken (half window = half the input period), never all rejected
    const double t = st.sec + st.nanosec * 1e-9;
    if (last_in_t_ > 0 && t > last_in_t_ && t - last_in_t_ < 1.0) {
      in_period_ = in_period_ <= 0 ? (t - last_in_t_) : 0.9 * in_period_ + 0.1 * (t - last_in_t_);
    }
    last_in_t_ = t;
    const double ph = cur_phase();
    if (ph < 0) {
      // no lidar phase (start-up, or no cloud stamps): still at most one frame per lidar cycle, so the camera chain's load never
      // exceeds its 10 Hz budget (a 30 Hz driver would otherwise run YOLOX 3x on the GPU CenterPoint shares)
      if (nophase_last_t_ > 0 && t > nophase_last_t_ && t - nophase_last_t_ < 0.9 * period_) return false;
      nophase_last_t_ = t;
      return true;
    }
    double d = std::fmod(t - ph, period_);
    if (d < 0) d += period_;
    d = std::min(d, period_ - d);
    // a driver at the lidar rate or slower (input period >= 0.75 cycle): every frame; at 30 Hz the Python node's +-half window exactly
    if (in_period_ >= 0.75 * period_) return true;
    return d <= half_;
  }

  void publish(const cv::Mat & bgr, const builtin_interfaces::msg::Time & st, Clock::time_point recv)
  {
    const auto t0 = Clock::now();
    auto msg = std::make_unique<Image>();
    msg->header.stamp = st;
    msg->header.frame_id = frame_id_;
    msg->height = oh_;
    msg->width = ow_;
    msg->encoding = "bgr8";
    msg->is_bigendian = 0;
    msg->step = ow_ * 3;
    msg->data.resize(static_cast<size_t>(oh_) * ow_ * 3);
    cv::Mat out(oh_, ow_, CV_8UC3, msg->data.data(), msg->step);   // remap straight into the message buffer (no extra copy)
    cv::remap(bgr, out, map1_, map2_, cv::INTER_LINEAR, cv::BORDER_CONSTANT);
    auto ci = std::make_unique<CameraInfo>(ci_);
    ci->header.stamp = st;
    pub_img_->publish(std::move(msg));
    pub_ci_->publish(std::move(ci));
    const auto t1 = Clock::now();
    std::lock_guard<std::mutex> lk(stat_mtx_);
    stat_n_++;
    stat_rect_ms_ += std::chrono::duration<double, std::milli>(t1 - t0).count();
    lag_ms_.push_back(std::chrono::duration<double, std::milli>(t1 - recv).count());
    maybe_log_stats_locked();
  }

  void maybe_log_stats_locked()
  {
    const auto now = Clock::now();
    if (std::chrono::duration<double>(now - stat_t_).count() < 10.0) return;
    std::sort(lag_ms_.begin(), lag_ms_.end());
    const double p50 = lag_ms_.empty() ? 0.0 : lag_ms_[lag_ms_.size() / 2];
    const double mx = lag_ms_.empty() ? 0.0 : lag_ms_.back();
    const double ph = cur_phase();
    RCLCPP_INFO(
      get_logger(),
      "[NERO_CAMERA0] %d frames in 10 s, rectify+publish %.1f ms, decode %.1f ms/frame (%d decoded), receive->publish p50 %.1f max %.1f ms, "
      "queue max %zu, input period %.1f ms, lidar phase %s",
      stat_n_, stat_rect_ms_ / std::max(stat_n_, 1), stat_dec_ms_ / std::max(stat_dec_n_, 1), stat_dec_n_, p50, mx, stat_qmax_, in_period_ * 1e3,
      ph < 0 ? "-" : (std::to_string(ph * 1e3).substr(0, 5) + " ms").c_str());
    stat_t_ = now;
    stat_n_ = 0;
    stat_rect_ms_ = 0;
    stat_dec_ms_ = 0;
    stat_dec_n_ = 0;
    stat_qmax_ = 0;
    lag_ms_.clear();
  }

  // ---- raw input
  void on_raw(const Image::ConstSharedPtr & m)
  {
    if (!wanted(m->header.stamp)) return;
    const auto recv = Clock::now();
    const std::string enc = [&] { std::string e = m->encoding; std::transform(e.begin(), e.end(), e.begin(), ::tolower); return e; }();
    auto * d = const_cast<uint8_t *>(m->data.data());
    cv::Mat bgr;
    if (enc == "bgr8") {
      bgr = cv::Mat(m->height, m->width, CV_8UC3, d, m->step ? m->step : m->width * 3);
    } else if (enc == "rgb8") {
      cv::cvtColor(cv::Mat(m->height, m->width, CV_8UC3, d, m->step ? m->step : m->width * 3), bgr, cv::COLOR_RGB2BGR);
    } else if (enc == "yuv422" || enc == "uyvy") {
      cv::cvtColor(cv::Mat(m->height, m->width, CV_8UC2, d, m->step ? m->step : m->width * 2), bgr, cv::COLOR_YUV2BGR_UYVY);
    } else if (enc == "yuv422_yuy2" || enc == "yuyv") {
      cv::cvtColor(cv::Mat(m->height, m->width, CV_8UC2, d, m->step ? m->step : m->width * 2), bgr, cv::COLOR_YUV2BGR_YUYV);
    } else if (enc == "mono8") {
      cv::cvtColor(cv::Mat(m->height, m->width, CV_8UC1, d, m->step ? m->step : m->width), bgr, cv::COLOR_GRAY2BGR);
    } else if (enc == "nv12") {
      cv::cvtColor(cv::Mat(m->height * 3 / 2, m->width, CV_8UC1, d), bgr, cv::COLOR_YUV2BGR_NV12);
    } else {
      RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 10000, "[NERO_CAMERA0] unsupported encoding %s", m->encoding.c_str());
      return;
    }
    publish(bgr, m->header.stamp, recv);
  }

  // ---- H.265 input: the callback queues, the worker decodes every packet (P-frames need their references) and converts / rectifies
  // only the wanted ones
  void open_decoder()
  {
    const AVCodec * codec = avcodec_find_decoder_by_name(decoder_name_.c_str());
    if (!codec) {
      RCLCPP_WARN(get_logger(), "[NERO_CAMERA0] decoder %s not found, using the software hevc decoder", decoder_name_.c_str());
      codec = avcodec_find_decoder(AV_CODEC_ID_HEVC);
    }
    if (!codec) throw std::runtime_error("no HEVC decoder in libavcodec");
    ctx_ = avcodec_alloc_context3(codec);
    ctx_->thread_count = std::max(1, decoder_threads_);
    ctx_->thread_type = FF_THREAD_SLICE;   // frame threading buffers frames (~0.5 s with PyAV's AUTO): never
    ctx_->flags |= AV_CODEC_FLAG_LOW_DELAY;
    if (avcodec_open2(ctx_, codec, nullptr) < 0) throw std::runtime_error("avcodec_open2 failed");
    frame_ = av_frame_alloc();
    pkt_ = av_packet_alloc();
    started_ = false;
  }

  void on_h265(CompressedImage::UniquePtr m)
  {
    Packet p;
    p.data = std::move(m->data);
    p.stamp = m->header.stamp;
    p.recv = Clock::now();
    {
      std::lock_guard<std::mutex> lk(q_mtx_);
      q_.push_back(std::move(p));
      if (static_cast<int>(q_.size()) > queue_max_) {   // the decoder is far behind: drop to the next keyframe (never silently)
        q_.clear();
        resync_ = true;
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, "[NERO_CAMERA0] decoder %d packets behind: waiting for a keyframe", queue_max_);
      }
      std::lock_guard<std::mutex> ls(stat_mtx_);
      stat_qmax_ = std::max(stat_qmax_, q_.size());
    }
    q_cv_.notify_one();
  }

  void decode_loop()
  {
    std::map<int64_t, std::pair<builtin_interfaces::msg::Time, Clock::time_point>> pending;
    int64_t n = 0;
    cv::Mat bgr(in_h_, in_w_, CV_8UC3);
    while (true) {
      Packet p;
      {
        std::unique_lock<std::mutex> lk(q_mtx_);
        q_cv_.wait(lk, [this] { return stop_ || !q_.empty(); });
        if (stop_) return;
        p = std::move(q_.front());
        q_.pop_front();
        if (resync_) {
          started_ = false;
          resync_ = false;
          pending.clear();
          avcodec_flush_buffers(ctx_);
        }
      }
      if (!started_) {
        if (!is_hevc_key(p.data.data(), p.data.size())) continue;
        started_ = true;
      }
      const auto t0 = Clock::now();
      pkt_->data = p.data.data();
      pkt_->size = static_cast<int>(p.data.size());
      pkt_->pts = n;
      pending[n] = {p.stamp, p.recv};
      ++n;
      if (avcodec_send_packet(ctx_, pkt_) < 0) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, "[NERO_CAMERA0] decode error: waiting for a keyframe");
        avcodec_flush_buffers(ctx_);
        started_ = false;
        pending.clear();
        continue;
      }
      while (avcodec_receive_frame(ctx_, frame_) == 0) {
        const auto t1 = Clock::now();
        {
          std::lock_guard<std::mutex> ls(stat_mtx_);
          stat_dec_n_++;
          stat_dec_ms_ += std::chrono::duration<double, std::milli>(t1 - t0).count();
        }
        auto it = pending.find(frame_->pts);
        if (it == pending.end()) continue;
        const auto st = it->second.first;
        const auto recv = it->second.second;
        pending.erase(pending.begin(), std::next(it));
        if (!wanted(st)) continue;
        if (frame_->width != bgr.cols || frame_->height != bgr.rows) bgr.create(frame_->height, frame_->width, CV_8UC3);
        sws_ = sws_getCachedContext(
          sws_, frame_->width, frame_->height, static_cast<AVPixelFormat>(frame_->format), frame_->width, frame_->height,
          AV_PIX_FMT_BGR24, SWS_POINT, nullptr, nullptr, nullptr);
        uint8_t * dst[1] = {bgr.data};
        int dst_stride[1] = {static_cast<int>(bgr.step)};
        sws_scale(sws_, frame_->data, frame_->linesize, 0, frame_->height, dst, dst_stride);
        publish(bgr, st, recv);
      }
    }
  }

  std::string mode_, frame_id_, decoder_name_;
  int in_w_{1920}, in_h_{1536}, ow_{1440}, oh_{1080}, decoder_threads_{2}, queue_max_{60};
  double period_{0.1}, phase_{-1.0}, half_{1.0 / 60.0}, last_in_t_{-1.0}, in_period_{-1.0}, nophase_last_t_{-1.0};
  cv::Mat map1_, map2_;
  CameraInfo ci_;
  rclcpp::Publisher<Image>::SharedPtr pub_img_;
  rclcpp::Publisher<CameraInfo>::SharedPtr pub_ci_;
  rclcpp::Subscription<autoware_internal_msgs::msg::PublishedTime>::SharedPtr sub_phase_;
  rclcpp::Subscription<CompressedImage>::SharedPtr sub_h265_;
  rclcpp::Subscription<Image>::SharedPtr sub_raw_;
  std::mutex ph_mtx_;
  std::deque<std::pair<double, double>> ph_hist_;
  // decoder
  AVCodecContext * ctx_{nullptr};
  AVFrame * frame_{nullptr};
  AVPacket * pkt_{nullptr};
  SwsContext * sws_{nullptr};
  bool started_{false};
  std::thread worker_;
  std::mutex q_mtx_;
  std::condition_variable q_cv_;
  std::deque<Packet> q_;
  bool stop_{false}, resync_{false};
  // stats
  std::mutex stat_mtx_;
  Clock::time_point stat_t_;
  int stat_n_{0}, stat_dec_n_{0};
  double stat_rect_ms_{0}, stat_dec_ms_{0};
  size_t stat_qmax_{0};
  std::vector<double> lag_ms_;
};

}  // namespace autoware::nero_camera0_rect

RCLCPP_COMPONENTS_REGISTER_NODE(autoware::nero_camera0_rect::Camera0RectNode)
