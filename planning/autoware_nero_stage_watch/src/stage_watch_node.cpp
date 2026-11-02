// Copyright 2026 Nero. Apache License 2.0.
//
// NERO_STAGE_WATCH (trajstall job, QUEUE TRJ2, 2026-11-02): a read-only observer of the planning chain.
//
// On 11-01 one planning callback was busy or blocked 1.3-1.9 s (the trajectory monitor's MRM followed) and nothing in
// the captures could say which stage, or whether the thread was computing or waiting. This node subscribes to every
// stage's output as raw bytes (it reads only header.stamp: behavior_path_planner stamps its path with now() at the start
// of its cycle and every later stage copies that stamp), and to nothing else. When the final trajectory stops or arrives
// late it:
//   - snapshots, while the stall lasts, the threads of the planning processes from /proc (state, kernel wait channel,
//     CPU time, major page faults) and the system's pressure (load, PSI, dirty / writeback memory, swap);
//   - when the chain is current again, prints two node-level WARN lines (node loggers reach /rosout, so the bridge
//     copies them into the captures):
//       [NERO_STAGE_WATCH] stall ...: per stage the longest silence, the late stamp's lag at each stage, the first
//         stage whose own output was late (= the stalled callback), the stamp interval before it (timer fired late?);
//       [NERO_STAGE_WATCH] threads ...: per planning process the threads that were busy (CPU ms during the stall) or
//         waiting in the kernel on something other than the executor's idle waits, + the system numbers.
// It publishes nothing and changes nothing; unset NERO_STAGE_WATCH = the node is not launched (the car as before).

#include <rclcpp/rclcpp.hpp>
#include <rclcpp/serialized_message.hpp>

#include <dirent.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace nero_stage_watch
{
namespace
{
struct StageDef
{
  const char * name;
  const char * topic;
  const char * type;
};
// the lane-driving chain in order (tier4_planning_launch 0.44); the last one is what
// topic_state_monitor_scenario_planning_trajectory watches. A stage with no publisher on this car stays out of the lines.
constexpr std::array<StageDef, 9> STAGES{{
  {"bpp", "/planning/scenario_planning/lane_driving/behavior_planning/path_with_lane_id",
   "autoware_internal_planning_msgs/msg/PathWithLaneId"},
  {"bvp", "/planning/scenario_planning/lane_driving/behavior_planning/path", "autoware_planning_msgs/msg/Path"},
  {"eb", "/planning/scenario_planning/lane_driving/motion_planning/path_smoother/path",
   "autoware_planning_msgs/msg/Path"},
  {"po", "/planning/scenario_planning/lane_driving/motion_planning/path_optimizer/trajectory",
   "autoware_planning_msgs/msg/Trajectory"},
  {"mvp", "/planning/scenario_planning/lane_driving/motion_planning/motion_velocity_planner/trajectory",
   "autoware_planning_msgs/msg/Trajectory"},
  {"ld", "/planning/scenario_planning/lane_driving/trajectory", "autoware_planning_msgs/msg/Trajectory"},
  {"sel", "/planning/scenario_planning/scenario_selector/trajectory", "autoware_planning_msgs/msg/Trajectory"},
  {"vs", "/planning/scenario_planning/velocity_smoother/trajectory", "autoware_planning_msgs/msg/Trajectory"},
  {"val", "/planning/scenario_planning/trajectory", "autoware_planning_msgs/msg/Trajectory"},
}};
constexpr size_t N = STAGES.size();
constexpr size_t FINAL = N - 1;  // planning_validator's output = the monitored /planning/scenario_planning/trajectory
// the planning processes whose threads are snapshotted (matched against "__node:=<name>" in the command line)
const std::vector<std::string> PROCS{
  "behavior_planning_container", "motion_planning_container", "scenario_selector", "velocity_smoother_container",
  "planning_validator"};
// kernel wait channels of an idle executor / timer thread (not reported)
const std::vector<std::string> IDLE_WCHAN{
  "futex_wait_queue", "futex_wait_queue_me", "ep_poll", "do_epoll_wait", "hrtimer_nanosleep", "do_nanosleep",
  "poll_schedule_timeout", "do_select", "do_sys_poll", "pipe_read", "wait_woken", "sk_wait_data",
  "__skb_wait_for_more_packets", "__skb_recv_udp", "unix_stream_read_generic", "do_wait", "0", ""};

double steady_s()
{
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

std::string read_file(const std::string & path, const size_t max = 4096)
{
  FILE * f = std::fopen(path.c_str(), "r");
  if (!f) return {};
  std::string s(max, '\0');
  const size_t n = std::fread(&s[0], 1, max, f);
  std::fclose(f);
  s.resize(n);
  while (!s.empty() && (s.back() == '\n' || s.back() == '\0')) s.pop_back();
  return s;
}

// header.stamp of a CDR-serialized message whose first field is std_msgs/Header
bool stamp_of(const rclcpp::SerializedMessage & m, double & out)
{
  const auto & r = m.get_rcl_serialized_message();
  if (r.buffer_length < 12) return false;
  const uint8_t * b = r.buffer;
  const bool le = b[1] == 0x01 || b[1] == 0x03;
  auto u32 = [&](const size_t o) {
    uint32_t v;
    if (le) {
      v = static_cast<uint32_t>(b[o]) | (static_cast<uint32_t>(b[o + 1]) << 8) |
          (static_cast<uint32_t>(b[o + 2]) << 16) | (static_cast<uint32_t>(b[o + 3]) << 24);
    } else {
      v = (static_cast<uint32_t>(b[o]) << 24) | (static_cast<uint32_t>(b[o + 1]) << 16) |
          (static_cast<uint32_t>(b[o + 2]) << 8) | static_cast<uint32_t>(b[o + 3]);
    }
    return v;
  };
  const int32_t sec = static_cast<int32_t>(u32(4));
  const uint32_t nsec = u32(8);
  if (nsec >= 1000000000u) return false;
  out = static_cast<double>(sec) + 1e-9 * static_cast<double>(nsec);
  return true;
}

struct ThreadSnap
{
  char state{'?'};
  std::string comm, wchan;
  long long ticks{0};   // utime + stime
  long long majflt{0};
};
struct ProcSnap
{
  double t{0.0};
  std::map<int, std::map<int, ThreadSnap>> procs;  // pid -> tid -> snap
  std::string sys;                                  // load / psi / memory at that moment
  long long pgmajfault{0};
};
}  // namespace

class StageWatch : public rclcpp::Node
{
public:
  StageWatch() : Node("nero_stage_watch")
  {
    // NERO_STAGE_WATCH=1 launches this node (planning.launch.xml); the thresholds are parameters
    gap_s_ = std::max(0.1, declare_parameter<double>("gap_s", 0.5));
    lag_s_ = std::max(0.1, declare_parameter<double>("lag_s", 0.5));
    hz_ = static_cast<double>(sysconf(_SC_CLK_TCK));
    if (hz_ <= 0.0) hz_ = 100.0;
    // best effort, keep the last one: the watch never holds a publisher back
    const auto qos = rclcpp::QoS(rclcpp::KeepLast(1)).best_effort().durability_volatile();
    for (size_t i = 0; i < N; ++i) {
      try {
        subs_.push_back(create_generic_subscription(
          STAGES[i].topic, STAGES[i].type, qos,
          [this, i](std::shared_ptr<rclcpp::SerializedMessage> m) { on_msg(i, *m); }));
      } catch (const std::exception & e) {  // a missing message package must never take the watch (or launch) down
        RCLCPP_WARN(get_logger(), "[NERO_STAGE_WATCH] %s not watched: %s", STAGES[i].name, e.what());
      }
    }
    find_pids();
    timer_ = create_wall_timer(std::chrono::milliseconds(50), [this] { tick(); });
    std::ostringstream o;
    o << "[NERO_STAGE_WATCH] watching";
    for (const auto & s : STAGES) o << " " << s.name;
    o << " (silence > " << gap_s_ << " s or a final trajectory > " << lag_s_ << " s old = a stall); processes:";
    for (const auto & [name, pid] : pids_) o << " " << name << "=" << pid;
    RCLCPP_INFO(get_logger(), "%s", o.str().c_str());
  }

private:
  struct Stage
  {
    double last_recv{-1.0};   // steady s
    double last_stamp{-1.0};  // header stamp (ROS time)
    double prev_stamp{-1.0};
    double max_gap{0.0};      // longest silence in the open episode
  };
  struct StampRow
  {
    std::array<double, N> lag;
    double interval{0.0};  // bpp stamp minus the previous bpp stamp (the planner's timer)
  };

  void on_msg(const size_t i, const rclcpp::SerializedMessage & m)
  {
    const double ts = steady_s();
    const double now = this->now().seconds();
    double stamp = 0.0;
    if (!stamp_of(m, stamp)) return;
    auto & st = stages_[i];
    if (st.last_recv > 0.0) st.max_gap = std::max(st.max_gap, ts - st.last_recv);
    st.last_recv = ts;
    st.prev_stamp = st.last_stamp;
    st.last_stamp = stamp;
    const double lag = now - stamp;
    const auto key = static_cast<long long>(stamp * 1e3);  // ms
    auto it = rows_.find(key);
    if (it == rows_.end()) {
      StampRow r;
      r.lag.fill(-1.0);
      it = rows_.emplace(key, r).first;
      while (rows_.size() > 100) rows_.erase(rows_.begin());
    }
    it->second.lag[i] = lag;
    if (i == 0 && st.prev_stamp > 0.0) it->second.interval = stamp - st.prev_stamp;
    if (i == FINAL) {
      ever_final_ = true;
      if (lag > lag_s_ && !open_) open_episode(ts, "a final trajectory " + fmt(lag) + " s old");
      if (open_ && lag <= lag_s_ && ts > t_open_) close_episode(ts, now);
    }
  }

  void tick()
  {
    const double ts = steady_s();
    if (ts - last_pid_scan_ > 10.0) find_pids();
    if (!ever_final_) return;
    const auto & fin = stages_[FINAL];
    const double gap = ts - fin.last_recv;
    if (!open_ && gap > gap_s_ && !stopped_) {
      open_episode(ts, "no final trajectory for " + fmt(gap) + " s");
    }
    if (open_) {
      if (snaps_.size() < 4 && ts - snaps_.back().t >= 0.3) snaps_.push_back(snapshot(ts));
      if (ts - t_open_ > 10.0 && !stopped_) {
        // silent 10 s: planning stopped (goal reached, no route, Autoware stopping), not a stall: one INFO line, then
        // quiet until it publishes again (no report when it does)
        RCLCPP_INFO(
          get_logger(), "[NERO_STAGE_WATCH] no final trajectory for %.0f s: planning stopped (not a stall)",
          ts - stages_[FINAL].last_recv);
        stopped_ = true;
      }
    }
  }

  void open_episode(const double ts, const std::string & why)
  {
    open_ = true;
    stopped_ = false;
    t_open_ = ts;
    ros_open_ = this->now().seconds();
    why_ = why;
    for (auto & s : stages_) s.max_gap = s.last_recv > 0.0 ? ts - s.last_recv : 0.0;
    snaps_.clear();
    snaps_.push_back(snapshot(ts));
  }

  void close_episode(const double ts, const double /*now*/)
  {
    if (!stopped_) report(ts);
    open_ = false;
    stopped_ = false;
  }

  void report(const double ts)
  {
    // per stage: the longest silence in the episode
    std::ostringstream a;
    a << "[NERO_STAGE_WATCH] stall: " << why_ << "; lasted " << fmt(ts - t_open_)
      << " s; silence";
    for (size_t i = 0; i < N; ++i) {
      if (stages_[i].last_recv > 0.0) a << " " << STAGES[i].name << " " << fmt(stages_[i].max_gap);
    }
    // the stamps of the episode: the late ones, with each stage's lag; the first stage whose own output was late
    a << "; stamps (lag at each stage):";
    int shown = 0;
    std::string first_late;
    for (auto it = rows_.rbegin(); it != rows_.rend() && shown < 4; ++it) {
      // only the stamps of this episode: planned from the last one before it went silent / late on
      if (static_cast<double>(it->first) / 1e3 < ros_open_ - gap_s_ - 1.5) break;
      const auto & r = it->second;
      double worst = -1.0;
      for (const double l : r.lag) worst = std::max(worst, l);
      if (worst < lag_s_ && shown > 0) continue;
      ++shown;
      a << " " << fmt(static_cast<double>(it->first % 100000) / 1e3) << "[";
      bool sp = false;
      for (size_t i = 0; i < N; ++i) {
        if (stages_[i].last_recv < 0.0) continue;
        a << (sp ? " " : "") << STAGES[i].name << " " << (r.lag[i] < 0.0 ? std::string("-") : fmt(r.lag[i]));
        sp = true;
      }
      a << "; bpp timer +" << fmt(r.interval) << "]";
      if (first_late.empty()) {
        for (size_t i = 0; i < N; ++i) {
          if (r.lag[i] >= lag_s_) {
            first_late = STAGES[i].name;
            break;
          }
        }
      }
    }
    a << "; first late stage: " << (first_late.empty() ? "none (a silence, no late stamp)" : first_late);
    if (ts - t_open_ + gap_s_ > 3.0) a << "; long: a route change or a restart looks the same";
    RCLCPP_WARN(get_logger(), "%s", a.str().c_str());
    RCLCPP_WARN(get_logger(), "%s", threads_line().c_str());
  }

  std::string threads_line()
  {
    // per process: CPU used during the stall, the busiest threads (>= 50 ms, their states in each snapshot), and every
    // non-idle state / kernel wait seen (count of threads): busy computing = R with CPU ~ the stall; blocked = D or S on
    // something other than the executor's idle waits (pipe_write, balance_dirty_pages, a page fault, ...)
    std::ostringstream b;
    b << "[NERO_STAGE_WATCH] threads during the stall (" << snaps_.size() << " snapshots over "
      << fmt(snaps_.empty() ? 0.0 : snaps_.back().t - snaps_.front().t) << " s):";
    if (snaps_.empty()) return b.str();
    const auto & s0 = snaps_.front();
    const auto & s1 = snaps_.back();
    for (const auto & [name, pid] : pids_) {
      const auto p0 = s0.procs.find(pid);
      const auto p1 = s1.procs.find(pid);
      const std::string shortname = short_name(name);
      if (p0 == s0.procs.end() || p1 == s1.procs.end() || p1->second.empty()) {
        b << " " << shortname << " gone;";
        continue;
      }
      std::vector<std::pair<double, std::string>> busy;
      std::map<std::string, int> waits;
      long long total = 0, faults = 0;
      for (const auto & [tid, t1] : p1->second) {
        const auto t0 = p0->second.find(tid);
        const long long d = t0 == p0->second.end() ? 0 : t1.ticks - t0->second.ticks;
        total += d;
        faults += t0 == p0->second.end() ? 0 : t1.majflt - t0->second.majflt;
        std::string seq;
        std::vector<std::string> odd;
        for (const auto & s : snaps_) {
          const auto pp = s.procs.find(pid);
          if (pp == s.procs.end()) continue;
          const auto tt = pp->second.find(tid);
          if (tt == pp->second.end()) continue;
          const auto & w = tt->second.wchan;
          const std::string st = std::string(1, tt->second.state) + (w.empty() || w == "0" ? "" : ":" + w);
          seq += (seq.empty() ? "" : ",") + st;
          const bool idle = tt->second.state == 'S' &&
            std::find(IDLE_WCHAN.begin(), IDLE_WCHAN.end(), w) != IDLE_WCHAN.end();
          if (!idle && std::find(odd.begin(), odd.end(), st) == odd.end()) odd.push_back(st);
        }
        for (const auto & st : odd) ++waits[st];
        const double ms = 1e3 * static_cast<double>(d) / hz_;
        if (ms >= 50.0) busy.emplace_back(ms, t1.comm + "/" + std::to_string(tid) + " " + fmt0(ms) + "ms " + seq);
      }
      b << " " << shortname << " cpu " << fmt0(1e3 * static_cast<double>(total) / hz_) << "ms";
      std::sort(busy.rbegin(), busy.rend());
      for (size_t k = 0; k < busy.size() && k < 3; ++k) b << (k ? " | " : " [") << busy[k].second;
      if (!busy.empty()) b << "]";
      if (!waits.empty()) {
        std::vector<std::pair<int, std::string>> wv;
        for (const auto & [w, c] : waits) wv.emplace_back(c, w);
        std::sort(wv.rbegin(), wv.rend());
        b << " waits";
        for (size_t k = 0; k < wv.size() && k < 4; ++k) b << " " << wv[k].second << " x" << wv[k].first;
      }
      if (faults > 0) b << " majflt +" << faults;
      b << ";";
    }
    b << " sys at start: " << s0.sys << "; pgmajfault +" << (s1.pgmajfault - s0.pgmajfault);
    return b.str();
  }

  static std::string short_name(const std::string & n)
  {
    if (n == "behavior_planning_container") return "bpc";
    if (n == "motion_planning_container") return "mpc";
    if (n == "velocity_smoother_container") return "vsc";
    if (n == "scenario_selector") return "sel";
    if (n == "planning_validator") return "val";
    return n;
  }

  ProcSnap snapshot(const double ts) const
  {
    ProcSnap s;
    s.t = ts;
    for (const auto & [name, pid] : pids_) {
      auto & threads = s.procs[pid];
      const std::string base = "/proc/" + std::to_string(pid) + "/task";
      DIR * d = opendir(base.c_str());
      if (!d) continue;
      while (dirent * e = readdir(d)) {
        if (e->d_name[0] < '0' || e->d_name[0] > '9') continue;
        const int tid = std::atoi(e->d_name);
        const std::string tb = base + "/" + e->d_name;
        const std::string stat = read_file(tb + "/stat");
        const auto rp = stat.rfind(')');
        if (rp == std::string::npos) continue;
        ThreadSnap t;
        t.comm = stat.substr(stat.find('(') + 1, rp - stat.find('(') - 1);
        std::istringstream is(stat.substr(rp + 2));
        std::vector<std::string> f;
        std::string x;
        while (is >> x && f.size() < 14) f.push_back(x);
        if (f.size() < 14) continue;
        t.state = f[0].empty() ? '?' : f[0][0];   // field 3
        t.majflt = std::atoll(f[9].c_str());      // field 12
        t.ticks = std::atoll(f[11].c_str()) + std::atoll(f[12].c_str());  // fields 14 + 15
        t.wchan = read_file(tb + "/wchan", 64);
        threads[tid] = t;
      }
      closedir(d);
    }
    std::ostringstream sys;
    const std::string la = read_file("/proc/loadavg", 128);
    sys << "load " << la.substr(0, la.find(' '));
    for (const char * r : {"cpu", "memory", "io"}) {
      const std::string p = read_file(std::string("/proc/pressure/") + r, 256);
      const auto k = p.find("avg10=");
      if (k != std::string::npos) sys << " psi_" << r << " " << p.substr(k + 6, p.find(' ', k) - k - 6);
    }
    std::istringstream mi(read_file("/proc/meminfo", 8192));
    std::string line;
    while (std::getline(mi, line)) {
      for (const char * key : {"MemAvailable:", "Dirty:", "Writeback:", "SwapFree:", "SwapTotal:"}) {
        if (line.rfind(key, 0) == 0) {
          std::istringstream ls(line.substr(std::strlen(key)));
          long long kb = 0;
          ls >> kb;
          sys << " " << std::string(key, std::strlen(key) - 1) << " " << kb / 1024 << "MB";
        }
      }
    }
    s.sys = sys.str();
    std::istringstream vm(read_file("/proc/vmstat", 16384));
    while (std::getline(vm, line)) {
      if (line.rfind("pgmajfault ", 0) == 0) s.pgmajfault = std::atoll(line.c_str() + 11);
    }
    return s;
  }

  // the planning processes by node name; and every ROS process by namespace + node name: a node that runs in two
  // processes at once (leftovers of an earlier launch after a restart) is reported once per change
  void find_pids()
  {
    last_pid_scan_ = steady_s();
    std::map<std::string, int> found;
    std::map<std::string, std::vector<int>> by_node;
    std::map<int, int> parent;
    DIR * d = opendir("/proc");
    if (!d) return;
    const auto arg = [](const std::string & cmd, const std::string & key) {
      const auto k = cmd.find(key);
      if (k == std::string::npos) return std::string();
      const auto e = cmd.find(' ', k + key.size());
      return cmd.substr(k + key.size(), e - k - key.size());
    };
    while (dirent * e = readdir(d)) {
      if (e->d_name[0] < '0' || e->d_name[0] > '9') continue;
      std::string cmd = read_file(std::string("/proc/") + e->d_name + "/cmdline", 8192);
      std::replace(cmd.begin(), cmd.end(), '\0', ' ');
      cmd += ' ';
      const std::string node = arg(cmd, "__node:=");
      if (node.empty()) continue;
      by_node[arg(cmd, "__ns:=") + "/" + node].push_back(std::atoi(e->d_name));
      const std::string st = read_file(std::string("/proc/") + e->d_name + "/stat", 1024);
      const auto rp = st.rfind(')');
      int ppid = 0;
      if (rp != std::string::npos) std::sscanf(st.c_str() + rp + 2, "%*c %d", &ppid);
      parent[std::atoi(e->d_name)] = ppid;
      for (const auto & p : PROCS) {
        if (node == p) found[p] = std::atoi(e->d_name);
      }
    }
    closedir(d);
    pids_ = found;
    std::string dup;
    for (const auto & [name, pids] : by_node) {
      // a process whose parent carries the same node name is that node's wrapper's child (ros2 run, a shell): one node
      std::vector<int> roots;
      for (const int p : pids) {
        if (std::find(pids.begin(), pids.end(), parent[p]) == pids.end()) roots.push_back(p);
      }
      if (roots.size() < 2) continue;
      dup += (dup.empty() ? "" : "; ") + name + " x" + std::to_string(roots.size()) + " (pids";
      for (const int p : roots) dup += " " + std::to_string(p);
      dup += ")";
    }
    if (dup != dup_) {
      if (!dup.empty()) {
        RCLCPP_WARN(get_logger(), "[NERO_STAGE_WATCH] a ROS node runs in more than one process: %s", dup.c_str());
      } else if (!dup_.empty()) {
        RCLCPP_INFO(get_logger(), "[NERO_STAGE_WATCH] no ROS node runs in more than one process any more");
      }
      dup_ = dup;
    }
  }

  static std::string fmt(const double v)
  {
    char b[32];
    std::snprintf(b, sizeof(b), "%.2f", v);
    return b;
  }
  static std::string fmt0(const double v)
  {
    char b[32];
    std::snprintf(b, sizeof(b), "%.0f", v);
    return b;
  }

  double gap_s_{0.5}, lag_s_{0.5}, hz_{100.0};
  std::vector<rclcpp::GenericSubscription::SharedPtr> subs_;
  rclcpp::TimerBase::SharedPtr timer_;
  std::array<Stage, N> stages_{};
  std::map<long long, StampRow> rows_;
  std::map<std::string, int> pids_;
  std::string dup_;
  double last_pid_scan_{0.0};
  bool ever_final_{false}, open_{false}, stopped_{false};
  double t_open_{0.0}, ros_open_{0.0};
  std::string why_;
  std::vector<ProcSnap> snaps_;
};
}  // namespace nero_stage_watch

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  // one thread: subscriptions and the timer never run at the same time (no locks needed)
  rclcpp::spin(std::make_shared<nero_stage_watch::StageWatch>());
  rclcpp::shutdown();
  return 0;
}
