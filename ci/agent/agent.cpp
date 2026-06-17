// Copyright 2026 VinRobotics - Apache-2.0
//
// vla-ci-agent: the LAN control-plane daemon. Runs on each platform server
// (Linux or macOS) and replaces SSH for the CI control plane. Listens on a
// ZeroMQ REP socket and serves the vla_ctl protocol (see vla_ctl.proto):
// ping / put / exec / spawn / stop / get.
//
// Single-threaded request loop (the orchestrator drives it serially). Spawned
// servers are detached into their own session/group so `stop` can signal the
// whole group; dead detached children are reaped at the top of the loop.
//
// Security: with --token T (or env VLA_CI_TOKEN) every request must carry a
// matching token. Bind to a LAN address only - this runs arbitrary commands by
// design, so it is meant for a trusted local network, never the open internet.
//
//   vla-ci-agent --bind tcp://*:5600 [--token SECRET]
#include "vla_ctl.pb.h"

#include <zmq.hpp>

#include <fcntl.h>
#include <signal.h>
#include <sys/utsname.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using vla_ctl::Request;
using vla_ctl::Reply;

namespace {

std::map<std::string, pid_t> g_spawned;   // name -> session-leader pid

std::vector<char*> to_argv(const google::protobuf::RepeatedPtrField<std::string>& a) {
    std::vector<char*> v;
    v.reserve(a.size() + 1);
    for (const auto& s : a) v.push_back(const_cast<char*>(s.c_str()));
    v.push_back(nullptr);
    return v;
}

void apply_env(const google::protobuf::RepeatedPtrField<std::string>& env) {
    for (const auto& kv : env) {
        auto p = kv.find('=');
        if (p != std::string::npos) setenv(kv.substr(0, p).c_str(), kv.substr(p + 1).c_str(), 1);
    }
}

// Run argv to completion in cwd, capturing stdout+stderr. Returns exit code
// (128+sig if signalled, 127 if exec failed).
int run_capture(const std::string& cwd, const std::vector<std::string>& argv,
                const google::protobuf::RepeatedPtrField<std::string>& env, std::string& out) {
    int fd[2];
    if (pipe(fd) != 0) { out = "pipe() failed"; return -1; }
    pid_t pid = fork();
    if (pid == 0) {
        close(fd[0]);
        dup2(fd[1], STDOUT_FILENO);
        dup2(fd[1], STDERR_FILENO);
        close(fd[1]);
        if (!cwd.empty() && chdir(cwd.c_str()) != 0) { perror("chdir"); _exit(126); }
        apply_env(env);
        std::vector<char*> a;
        for (const auto& s : argv) a.push_back(const_cast<char*>(s.c_str()));
        a.push_back(nullptr);
        execvp(a[0], a.data());
        perror("execvp");
        _exit(127);
    }
    close(fd[1]);
    char buf[8192];
    ssize_t n;
    while ((n = read(fd[0], buf, sizeof buf)) > 0) out.append(buf, n);
    close(fd[0]);
    int st = 0;
    waitpid(pid, &st, 0);
    if (WIFEXITED(st)) return WEXITSTATUS(st);
    if (WIFSIGNALED(st)) return 128 + WTERMSIG(st);
    return -1;
}

// Launch a detached background process in its own session/group. Returns pid
// (== pgid), or -1 on fork failure.
pid_t spawn_detached(const std::string& cwd, const std::vector<std::string>& argv,
                     const google::protobuf::RepeatedPtrField<std::string>& env,
                     const std::string& logfile) {
    if (!logfile.empty()) {
        std::error_code ec;
        fs::create_directories(fs::path(logfile).parent_path(), ec);
    }
    pid_t pid = fork();
    if (pid == 0) {
        setsid();                              // new session+group: pgid == this pid
        if (!cwd.empty() && chdir(cwd.c_str()) != 0) _exit(126);
        int dn = open("/dev/null", O_RDONLY);
        if (dn >= 0) { dup2(dn, STDIN_FILENO); close(dn); }
        if (!logfile.empty()) {
            int lf = open(logfile.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
            if (lf >= 0) { dup2(lf, STDOUT_FILENO); dup2(lf, STDERR_FILENO); close(lf); }
        }
        apply_env(env);
        std::vector<char*> a;
        for (const auto& s : argv) a.push_back(const_cast<char*>(s.c_str()));
        a.push_back(nullptr);
        execvp(a[0], a.data());
        _exit(127);
    }
    return pid;
}

// SIGINT -> (grace) -> SIGTERM -> (grace) -> SIGKILL on the whole process group.
void kill_group(pid_t pid) {
    auto gone = [&] { return waitpid(pid, nullptr, WNOHANG) == pid; };
    killpg(pid, SIGINT);
    for (int i = 0; i < 20; ++i) { if (gone()) return; usleep(500000); }
    killpg(pid, SIGTERM);
    for (int i = 0; i < 10; ++i) { if (gone()) return; usleep(500000); }
    killpg(pid, SIGKILL);
    waitpid(pid, nullptr, 0);
}

bool is_protected(const std::string& rel, const google::protobuf::RepeatedPtrField<std::string>& prot) {
    if (rel == ".git" || rel.rfind(".git/", 0) == 0) return true;
    for (const auto& p : prot) {
        if (p.empty()) continue;
        if (rel == p || rel.rfind(p + "/", 0) == 0) return true;
    }
    return false;
}

// Extract a gzip-tar (already written to tarfile) into dst via the system tar,
// then (optionally) delete files under dst absent from the archive and not
// protected. Returns false on extract failure.
bool extract_and_prune(const std::string& tarfile, const std::string& dst, bool prune,
                       const google::protobuf::RepeatedPtrField<std::string>& protect,
                       uint32_t& n_files, uint32_t& n_pruned, std::string& err) {
    std::error_code ec;
    fs::create_directories(dst, ec);
    google::protobuf::RepeatedPtrField<std::string> noenv;
    std::string out;
    int rc = run_capture("", {"tar", "xzf", tarfile, "-C", dst}, noenv, out);
    if (rc != 0) { err = "tar xzf failed: " + out; return false; }

    // Member list (for both the file count and prune's keep-set).
    std::string list;
    run_capture("", {"tar", "tzf", tarfile}, noenv, list);
    std::vector<std::string> keep;
    n_files = 0;
    size_t s = 0;
    while (s < list.size()) {
        size_t e = list.find('\n', s);
        if (e == std::string::npos) e = list.size();
        std::string m = list.substr(s, e - s);
        s = e + 1;
        if (m.rfind("./", 0) == 0) m = m.substr(2);
        while (!m.empty() && m.back() == '\n') m.pop_back();
        if (m.empty()) continue;
        if (m.back() != '/') { keep.push_back(m); ++n_files; }   // file (dirs end with /)
    }
    n_pruned = 0;
    if (!prune) return true;

    auto in_keep = [&](const std::string& r) {
        for (const auto& k : keep) if (k == r) return true;
        return false;
    };
    for (auto it = fs::recursive_directory_iterator(dst, ec);
         it != fs::recursive_directory_iterator(); it.increment(ec)) {
        if (ec) break;
        std::string rel = fs::relative(it->path(), dst, ec).generic_string();
        if (it->is_directory(ec)) {
            if (is_protected(rel, protect)) it.disable_recursion_pending();   // never descend
            continue;
        }
        if (!it->is_regular_file(ec)) continue;
        if (is_protected(rel, protect) || in_keep(rel)) continue;
        std::error_code rmec;
        if (fs::remove(it->path(), rmec)) ++n_pruned;
    }
    return true;
}

void handle(const Request& req, Reply& rep) {
    rep.set_ok(true);
    switch (req.op_case()) {
        case Request::kPing: {
            utsname u{};
            uname(&u);
            auto* r = rep.mutable_ping();
            r->set_os(u.sysname);
            r->set_machine(u.machine);
            char cwd[4096];
            r->set_cwd(getcwd(cwd, sizeof cwd) ? cwd : "");
            break;
        }
        case Request::kPut: {
            const auto& p = req.put();
            std::string tmp = (fs::temp_directory_path() /
                               ("vla-ci-put-" + std::to_string(getpid()) + "-" +
                                std::to_string((long)time(nullptr)) + ".tgz")).string();
            { std::ofstream f(tmp, std::ios::binary); f.write(p.targz().data(), p.targz().size()); }
            uint32_t files = 0, pruned = 0;
            std::string err;
            bool ok = extract_and_prune(tmp, p.dst(), p.prune(), p.protect(), files, pruned, err);
            std::error_code ec; fs::remove(tmp, ec);
            if (!ok) { rep.set_ok(false); rep.set_error(err); break; }
            rep.mutable_put()->set_files(files);
            rep.mutable_put()->set_pruned(pruned);
            break;
        }
        case Request::kExec: {
            const auto& e = req.exec();
            std::vector<std::string> argv(e.argv().begin(), e.argv().end());
            if (argv.empty()) { rep.set_ok(false); rep.set_error("empty argv"); break; }
            std::string out;
            int code = run_capture(e.cwd(), argv, e.env(), out);
            rep.mutable_exec()->set_exit_code(code);
            rep.mutable_exec()->set_output(out);
            break;
        }
        case Request::kSpawn: {
            const auto& s = req.spawn();
            std::vector<std::string> argv(s.argv().begin(), s.argv().end());
            if (argv.empty()) { rep.set_ok(false); rep.set_error("empty argv"); break; }
            auto prev = g_spawned.find(s.name());
            if (prev != g_spawned.end()) { kill_group(prev->second); g_spawned.erase(prev); }
            pid_t pid = spawn_detached(s.cwd(), argv, s.env(), s.logfile());
            if (pid < 0) { rep.set_ok(false); rep.set_error("fork failed"); break; }
            g_spawned[s.name()] = pid;
            rep.mutable_spawn()->set_pid(pid);
            break;
        }
        case Request::kStop: {
            auto it = g_spawned.find(req.stop().name());
            if (it == g_spawned.end()) { rep.mutable_stop()->set_stopped(false); break; }
            kill_group(it->second);
            rep.mutable_stop()->set_stopped(true);
            rep.mutable_stop()->set_pid(it->second);
            g_spawned.erase(it);
            break;
        }
        case Request::kGet: {
            std::ifstream f(req.get().path(), std::ios::binary);
            if (!f) { rep.set_ok(false); rep.set_error("cannot open " + req.get().path()); break; }
            std::string data((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
            rep.mutable_get()->set_data(data);
            break;
        }
        case Request::kRev: {
            const std::string& path = req.rev().path();
            google::protobuf::RepeatedPtrField<std::string> noenv;
            std::string sha;
            int rc = run_capture("", {"git", "-C", path, "rev-parse", "HEAD"}, noenv, sha);
            while (!sha.empty() && (sha.back() == '\n' || sha.back() == '\r' || sha.back() == ' ')) sha.pop_back();
            if (rc != 0 || sha.empty()) {
                rep.set_ok(false);
                rep.set_error("git rev-parse HEAD failed at " + path + " (not a git repo?): " + sha);
                break;
            }
            std::string st;
            run_capture("", {"git", "-C", path, "status", "--porcelain"}, noenv, st);
            rep.mutable_rev()->set_commit(sha);
            rep.mutable_rev()->set_dirty(!st.empty());
            break;
        }
        default:
            rep.set_ok(false);
            rep.set_error("no op set");
    }
}

}  // namespace

int main(int argc, char** argv) {
    GOOGLE_PROTOBUF_VERIFY_VERSION;
    std::string bind = "tcp://*:5600";
    std::string token = std::getenv("VLA_CI_TOKEN") ? std::getenv("VLA_CI_TOKEN") : "";
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--bind" && i + 1 < argc) bind = argv[++i];
        else if (a == "--token" && i + 1 < argc) token = argv[++i];
        else if (a == "-h" || a == "--help") {
            std::printf("usage: %s [--bind tcp://*:5600] [--token SECRET]\n", argv[0]);
            return 0;
        } else { std::fprintf(stderr, "unknown arg: %s\n", a.c_str()); return 2; }
    }

    // A dropped peer must not kill us with SIGPIPE; detached children are reaped
    // at the top of the request loop instead.
    signal(SIGPIPE, SIG_IGN);

    zmq::context_t zctx(1);
    zmq::socket_t sock(zctx, zmq::socket_type::rep);
    sock.set(zmq::sockopt::linger, 0);
    sock.bind(bind);
    std::printf("vla-ci-agent: bound to %s. ready.%s\n", bind.c_str(),
                token.empty() ? "" : " (token auth on)");
    std::fflush(stdout);

    zmq::pollitem_t poll[] = {{static_cast<void*>(sock), 0, ZMQ_POLLIN, 0}};
    for (;;) {
        while (waitpid(-1, nullptr, WNOHANG) > 0) {}   // reap dead detached children
        try {
            zmq::poll(poll, 1, std::chrono::milliseconds(200));
        } catch (const zmq::error_t&) { continue; }
        if (!(poll[0].revents & ZMQ_POLLIN)) continue;

        zmq::message_t msg;
        auto rr = sock.recv(msg, zmq::recv_flags::none);
        if (!rr) continue;

        Request req;
        Reply rep;
        if (!req.ParseFromArray(msg.data(), msg.size())) {
            rep.set_ok(false);
            rep.set_error("malformed request");
        } else if (!token.empty() && req.token() != token) {
            rep.set_ok(false);
            rep.set_error("auth failed");
        } else {
            handle(req, rep);
        }
        std::string body;
        rep.SerializeToString(&body);
        sock.send(zmq::buffer(body), zmq::send_flags::none);
    }
}
