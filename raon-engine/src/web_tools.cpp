#include "web_tools.h"

#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <cstring>
#include <filesystem>

extern char** environ;

using ojson = nlohmann::ordered_json;

WebTools::WebTools(std::string dir, std::string node, int timeout_s)
    : dir_(std::move(dir)), node_(std::move(node)), timeout_s_(timeout_s) {
    std::error_code ec;
    if (!std::filesystem::exists(std::filesystem::path(dir_) / "tools.ts", ec)) {
        error_ = dir_ + "/tools.ts not found";
        return;
    }
    if (!std::filesystem::exists(std::filesystem::path(dir_) / "node_modules", ec)) {
        error_ = dir_ + "/node_modules missing (run: make webtools)";
        return;
    }
    RunResult r = run({node_, dir_ + "/tools.ts", "list"}, "", 30, nullptr);
    if (r.status != 0) {
        error_ = "'" + node_ + " tools.ts list' failed: " + (r.err.empty() ? r.out : r.err);
        return;
    }
    try {
        list_ = ojson::parse(r.out);
        if (!list_.is_array()) throw std::runtime_error("not an array");
    } catch (const std::exception& e) {
        error_ = std::string("bad tool list: ") + e.what();
        list_ = ojson::array();
        return;
    }
    enabled_ = true;
}

bool WebTools::has(const std::string& name) const {
    for (const auto& t : list_)
        if (t.value("tool", "") == name) return true;
    return false;
}

ojson WebTools::call(const std::string& name, const ojson& params, const std::function<bool()>& should_stop) const {
    RunResult r = run({node_, dir_ + "/tools.ts", "call", name}, params.dump(), timeout_s_, should_stop);
    if (r.stopped) return {{"error", "tool call cancelled"}};
    if (r.timed_out) return {{"error", name + " timed out after " + std::to_string(timeout_s_) + " s"}};
    try {
        ojson j = ojson::parse(r.out);
        if (j.is_object() && (j.contains("plain_text_response") || j.contains("error"))) return j;
    } catch (...) {
    }
    std::string err = r.err.empty() ? r.out : r.err;
    if (err.size() > 2000) err = err.substr(err.size() - 2000);
    return {{"error", name + " failed (exit " + std::to_string(r.status) + "): " + err}};
}

WebTools::RunResult WebTools::run(const std::vector<std::string>& args, const std::string& input, int timeout_s,
                                  const std::function<bool()>& should_stop) const {
    RunResult res;
    int in_p[2], out_p[2], err_p[2];
    if (pipe2(in_p, O_CLOEXEC) || pipe2(out_p, O_CLOEXEC) || pipe2(err_p, O_CLOEXEC)) {
        res.err = std::string("pipe: ") + strerror(errno);
        return res;
    }
    posix_spawn_file_actions_t fa;
    posix_spawn_file_actions_init(&fa);
    posix_spawn_file_actions_adddup2(&fa, in_p[0], 0);
    posix_spawn_file_actions_adddup2(&fa, out_p[1], 1);
    posix_spawn_file_actions_adddup2(&fa, err_p[1], 2);
    posix_spawnattr_t attr;
    posix_spawnattr_init(&attr);
    posix_spawnattr_setflags(&attr, POSIX_SPAWN_SETPGROUP | POSIX_SPAWN_SETSIGMASK);
    posix_spawnattr_setpgroup(&attr, 0);  // own process group: a kill also reaches node's children (python, liteparse)
    sigset_t empty;
    sigemptyset(&empty);
    posix_spawnattr_setsigmask(&attr, &empty);

    std::vector<char*> argv;
    for (const auto& a : args) argv.push_back(const_cast<char*>(a.c_str()));
    argv.push_back(nullptr);
    pid_t pid = -1;
    int rc = posix_spawnp(&pid, argv[0], &fa, &attr, argv.data(), environ);
    posix_spawn_file_actions_destroy(&fa);
    posix_spawnattr_destroy(&attr);
    close(in_p[0]);
    close(out_p[1]);
    close(err_p[1]);
    if (rc != 0) {
        close(in_p[1]);
        close(out_p[0]);
        close(err_p[0]);
        res.err = "spawn " + args[0] + ": " + strerror(rc);
        return res;
    }

    // stdin: the params are small (a few KB at most), but write non-blocking alongside reading to be safe
    int in_fd = in_p[1];
    fcntl(in_fd, F_SETFL, O_NONBLOCK);
    size_t in_off = 0;
    if (input.empty()) { close(in_fd); in_fd = -1; }

    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(timeout_s);
    int fds_open = 2;
    bool out_open = true, err_open = true;
    char buf[65536];
    while (fds_open > 0) {
        pollfd p[3];
        int n = 0, i_out = -1, i_err = -1, i_in = -1;
        if (out_open) { p[n] = {out_p[0], POLLIN, 0}; i_out = n++; }
        if (err_open) { p[n] = {err_p[0], POLLIN, 0}; i_err = n++; }
        if (in_fd >= 0) { p[n] = {in_fd, POLLOUT, 0}; i_in = n++; }
        int pr = poll(p, n, 200);
        if (pr < 0 && errno != EINTR) break;
        if (should_stop && should_stop()) { res.stopped = true; break; }
        if (std::chrono::steady_clock::now() > deadline) { res.timed_out = true; break; }
        if (pr <= 0) continue;
        if (i_in >= 0 && (p[i_in].revents & (POLLOUT | POLLERR | POLLHUP))) {
            ssize_t w = write(in_fd, input.data() + in_off, input.size() - in_off);
            if (w > 0) in_off += (size_t)w;
            if (w < 0 && errno != EAGAIN) in_off = input.size();
            if (in_off >= input.size()) { close(in_fd); in_fd = -1; }
        }
        auto drain = [&](int idx, int fd, std::string& dst, bool& open) {
            if (idx < 0 || !(p[idx].revents & (POLLIN | POLLHUP | POLLERR))) return;
            ssize_t r = read(fd, buf, sizeof(buf));
            if (r > 0) dst.append(buf, (size_t)r);
            else if (r == 0 || errno != EINTR) { open = false; fds_open--; }
        };
        drain(i_out, out_p[0], res.out, out_open);
        drain(i_err, err_p[0], res.err, err_open);
    }
    if (in_fd >= 0) close(in_fd);
    close(out_p[0]);
    close(err_p[0]);
    if (res.stopped || res.timed_out) kill(-pid, SIGKILL);
    int st = 0;
    while (waitpid(pid, &st, 0) < 0 && errno == EINTR) {}
    res.status = WIFEXITED(st) ? WEXITSTATUS(st) : 128 + (WIFSIGNALED(st) ? WTERMSIG(st) : 0);
    return res;
}
