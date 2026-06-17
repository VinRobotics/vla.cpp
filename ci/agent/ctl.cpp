// Copyright 2026 VinRobotics - Apache-2.0
//
// vla-ci-ctl: orchestrator-side CLI for the vla-ci control plane. Speaks the
// vla_ctl protocol (protobuf over a ZeroMQ REQ socket) to a vla-ci-agent on a
// platform server. Used by ci/run_remote.sh in place of ssh/rsync/scp.
//
//   vla-ci-ctl --endpoint tcp://192.168.1.10:5600 [--token S] <cmd> ...
//     ping
//     put   --src DIR --dst DIR [--prune] [--protect P]...   (general; CI no longer deploys with it)
//     exec  --cwd DIR -- ARGV...                              (general; CI no longer builds with it)
//     build --cwd DIR [--flags "<cmake>"] [--jobs N] [--no-patch]   # build vla-server in the server's checkout
//     spawn --name N --cwd DIR --log FILE -- ARGV...
//     stop  --name N
//     get   --remote PATH --local PATH
//     rev   --path DIR        # prints `git -C DIR rev-parse HEAD` (server's real commit)
//
// `exec` prints the remote stdout+stderr and exits with the remote exit code, so
// a failed remote build fails the shell pipeline under `set -e`.
#include "vla_ctl.pb.h"

#include <zmq.hpp>

#include <sys/wait.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

using vla_ctl::Request;
using vla_ctl::Reply;

namespace {

// Run argv locally, capturing stdout (binary-safe). Used to build the tar.
int capture(const std::vector<std::string>& argv, std::string& out) {
    int fd[2];
    if (pipe(fd) != 0) return -1;
    pid_t pid = fork();
    if (pid == 0) {
        close(fd[0]);
        dup2(fd[1], STDOUT_FILENO);
        close(fd[1]);
        std::vector<char*> a;
        for (const auto& s : argv) a.push_back(const_cast<char*>(s.c_str()));
        a.push_back(nullptr);
        execvp(a[0], a.data());
        _exit(127);
    }
    close(fd[1]);
    char buf[65536];
    ssize_t n;
    while ((n = read(fd[0], buf, sizeof buf)) > 0) out.append(buf, n);
    close(fd[0]);
    int st = 0;
    waitpid(pid, &st, 0);
    return WIFEXITED(st) ? WEXITSTATUS(st) : -1;
}

[[noreturn]] void die(const std::string& m) { std::fprintf(stderr, "vla-ci-ctl: %s\n", m.c_str()); std::exit(2); }

// Send one request, receive the reply. timeout_ms bounds the wait for the reply.
Reply call(const std::string& endpoint, int timeout_ms, Request& req) {
    zmq::context_t ctx(1);
    zmq::socket_t sock(ctx, zmq::socket_type::req);
    sock.set(zmq::sockopt::linger, 0);
    sock.set(zmq::sockopt::rcvtimeo, timeout_ms);
    sock.set(zmq::sockopt::sndtimeo, 30000);
    sock.connect(endpoint);
    std::string body;
    req.SerializeToString(&body);
    sock.send(zmq::buffer(body), zmq::send_flags::none);
    zmq::message_t msg;
    auto rr = sock.recv(msg, zmq::recv_flags::none);
    if (!rr) die("no reply from " + endpoint + " (timeout) - is vla-ci-agent running there?");
    Reply rep;
    if (!rep.ParseFromArray(msg.data(), msg.size())) die("malformed reply");
    return rep;
}

}  // namespace

int main(int argc, char** argv) {
    GOOGLE_PROTOBUF_VERIFY_VERSION;
    std::string endpoint, token = std::getenv("VLA_CI_TOKEN") ? std::getenv("VLA_CI_TOKEN") : "";
    int timeout_s = 0;   // 0 -> per-command default

    int i = 1;
    for (; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--endpoint" && i + 1 < argc) endpoint = argv[++i];
        else if (a == "--token" && i + 1 < argc) token = argv[++i];
        else if (a == "--timeout-s" && i + 1 < argc) timeout_s = std::atoi(argv[++i]);
        else break;
    }
    if (endpoint.empty()) die("--endpoint tcp://HOST:PORT is required");
    if (i >= argc) die("missing command (ping|put|exec|spawn|stop|get)");
    std::string cmd = argv[i++];

    auto rest = [&](const std::string& flag) -> std::string {
        for (int j = i; j < argc; ++j) if (flag == argv[j] && j + 1 < argc) return argv[j + 1];
        return "";
    };
    auto has = [&](const std::string& flag) {
        for (int j = i; j < argc; ++j) if (flag == argv[j]) return true;
        return false;
    };
    auto after_ddash = [&]() {
        std::vector<std::string> v;
        bool seen = false;
        for (int j = i; j < argc; ++j) {
            if (!seen) { if (std::string(argv[j]) == "--") seen = true; continue; }
            v.emplace_back(argv[j]);
        }
        return v;
    };

    Request req;
    req.set_token(token);
    int def_timeout_ms = 120000;

    if (cmd == "ping") {
        req.mutable_ping();
    } else if (cmd == "put") {
        std::string src = rest("--src"), dst = rest("--dst");
        if (src.empty() || dst.empty()) die("put needs --src and --dst");
        auto* p = req.mutable_put();
        p->set_dst(dst);
        p->set_prune(has("--prune"));
        std::vector<std::string> tar = {"tar", "czf", "-", "-C", src, "--exclude=./.git"};
        for (int j = i; j < argc; ++j) {
            if (std::string(argv[j]) == "--protect" && j + 1 < argc) {
                std::string pp = argv[++j];
                p->add_protect(pp);
                tar.push_back("--exclude=./" + pp);
            }
        }
        tar.push_back(".");
        std::string targz;
        if (capture(tar, targz) != 0) die("local `tar czf` failed for " + src);
        p->set_targz(targz);
        def_timeout_ms = 1800000;   // 30 min: tar upload of a fresh tree
    } else if (cmd == "exec") {
        auto a = after_ddash();
        if (a.empty()) die("exec needs `-- ARGV...`");
        auto* e = req.mutable_exec();
        e->set_cwd(rest("--cwd"));
        for (auto& s : a) e->add_argv(s);
        def_timeout_ms = 3600000;   // 60 min: remote build
    } else if (cmd == "build") {
        // Convenience over `exec`: build vla-server in the server's own checkout.
        std::string cwd = rest("--cwd");
        if (cwd.empty()) die("build needs --cwd <server repo root>");
        std::string flags = rest("--flags");                 // cmake flags (may be empty -> Metal auto)
        std::string jobs = rest("--jobs");                   // optional -j value
        std::string jexpr = jobs.empty() ? "$(getconf _NPROCESSORS_ONLN)" : jobs;
        std::string patch = has("--no-patch") ? "" : "bash patches/patch.sh; ";
        std::string script = "set -e; " + patch +
            "cmake -B build -DCMAKE_BUILD_TYPE=Release " + flags + "; "
            "cmake --build build -j" + jexpr;
        auto* e = req.mutable_exec();
        e->set_cwd(cwd);
        e->add_argv("bash"); e->add_argv("-lc"); e->add_argv(script);
        def_timeout_ms = 3600000;   // 60 min: remote build
    } else if (cmd == "spawn") {
        auto a = after_ddash();
        if (a.empty()) die("spawn needs `-- ARGV...`");
        std::string name = rest("--name");
        if (name.empty()) die("spawn needs --name");
        auto* s = req.mutable_spawn();
        s->set_name(name);
        s->set_cwd(rest("--cwd"));
        s->set_logfile(rest("--log"));
        for (auto& x : a) s->add_argv(x);
    } else if (cmd == "stop") {
        std::string name = rest("--name");
        if (name.empty()) die("stop needs --name");
        req.mutable_stop()->set_name(name);
    } else if (cmd == "get") {
        std::string remote = rest("--remote");
        if (remote.empty()) die("get needs --remote");
        req.mutable_get()->set_path(remote);
    } else if (cmd == "rev") {
        std::string path = rest("--path");
        if (path.empty()) die("rev needs --path");
        req.mutable_rev()->set_path(path);
    } else {
        die("unknown command: " + cmd);
    }

    int timeout_ms = timeout_s > 0 ? timeout_s * 1000 : def_timeout_ms;
    Reply rep = call(endpoint, timeout_ms, req);

    if (cmd == "exec" || cmd == "build") {
        std::fwrite(rep.exec().output().data(), 1, rep.exec().output().size(), stdout);
        std::fflush(stdout);
        if (!rep.ok()) { std::fprintf(stderr, "vla-ci-ctl: %s\n", rep.error().c_str()); return 1; }
        return rep.exec().exit_code();
    }
    if (!rep.ok()) { std::fprintf(stderr, "vla-ci-ctl: %s\n", rep.error().c_str()); return 1; }

    if (cmd == "ping") {
        std::printf("ok: %s %s  cwd=%s\n", rep.ping().os().c_str(), rep.ping().machine().c_str(),
                    rep.ping().cwd().c_str());
    } else if (cmd == "put") {
        std::printf("ok: %u files, %u pruned -> %s\n", rep.put().files(), rep.put().pruned(),
                    req.put().dst().c_str());
    } else if (cmd == "spawn") {
        std::printf("ok: spawned pid=%d\n", rep.spawn().pid());
    } else if (cmd == "stop") {
        std::printf("ok: stopped=%d pid=%d\n", rep.stop().stopped(), rep.stop().pid());
    } else if (cmd == "get") {
        std::string local = rest("--local");
        if (local.empty()) die("get needs --local");
        std::ofstream f(local, std::ios::binary);
        if (!f) die("cannot write " + local);
        f.write(rep.get().data().data(), rep.get().data().size());
        std::printf("ok: %zu bytes -> %s\n", rep.get().data().size(), local.c_str());
    } else if (cmd == "rev") {
        std::printf("%s\n", rep.rev().commit().c_str());   // bare commit, capturable by callers
        if (rep.rev().dirty())
            std::fprintf(stderr, "vla-ci-ctl: warning: server repo has uncommitted changes\n");
    }
    return 0;
}
