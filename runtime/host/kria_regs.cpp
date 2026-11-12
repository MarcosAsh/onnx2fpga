// A bounded register window over stdin/stdout, for local use or SSH.

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstdint>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {
constexpr size_t span = 0x10000;

uint64_t hex(const std::string& text) {
    if (text.empty() || text.find_first_not_of("0123456789abcdefABCDEF") != std::string::npos)
        throw std::runtime_error("expected an unsigned hexadecimal integer");
    size_t end = 0;
    const auto value = std::stoull(text, &end, 16);
    if (end != text.size()) throw std::runtime_error("invalid hexadecimal integer");
    return value;
}

void barrier() {
#if defined(__aarch64__)
    // Order device accesses across the AXI register transactions.
    asm volatile("dmb sy" ::: "memory");
#else
    __sync_synchronize();
#endif
}

class Registers {
public:
    Registers(const std::string& device, uint64_t base) {
        const long page = sysconf(_SC_PAGESIZE);
        if (page <= 0 || base % static_cast<uint64_t>(page) != 0 ||
            base > static_cast<uint64_t>(std::numeric_limits<off_t>::max()) - span)
            throw std::runtime_error("invalid register base");
        const int fd = open(device.c_str(), O_RDWR | O_SYNC | O_CLOEXEC);
        if (fd < 0) throw std::runtime_error("cannot open register device");
        struct stat info {};
        if (fstat(fd, &info) != 0) {
            close(fd);
            throw std::runtime_error("cannot stat register device");
        }
#if !defined(__aarch64__)
        if (!S_ISREG(info.st_mode)) {
            close(fd);
            throw std::runtime_error("physical register access requires AArch64");
        }
#endif
        if (S_ISREG(info.st_mode) && static_cast<uint64_t>(info.st_size) < base + span) {
            close(fd);
            throw std::runtime_error("register file is too short");
        }
        void* mapped = mmap(nullptr, span, PROT_READ | PROT_WRITE, MAP_SHARED,
                            fd, static_cast<off_t>(base));
        close(fd);
        if (mapped == MAP_FAILED) throw std::runtime_error("cannot map register device");
        words_ = static_cast<volatile uint32_t*>(mapped);
    }
    Registers(const Registers&) = delete;
    Registers& operator=(const Registers&) = delete;
    ~Registers() { munmap(const_cast<uint32_t*>(words_), span); }

    uint32_t read(uint64_t offset) const {
        check(offset);
        barrier();
        const uint32_t value = words_[offset / 4];
        barrier();
        return value;
    }
    void write(uint64_t offset, uint64_t value) {
        check(offset);
        if (value > UINT32_MAX) throw std::runtime_error("register value exceeds 32 bits");
        barrier();
        words_[offset / 4] = static_cast<uint32_t>(value);
        barrier();
    }

private:
    static void check(uint64_t offset) {
        if (offset >= span || offset % 4) throw std::runtime_error("invalid register offset");
    }
    volatile uint32_t* words_ = nullptr;
};
}  // namespace

int main(int argc, char** argv) {
    try {
        std::string device = "/dev/mem";
        uint64_t base = 0xa0000000;
        for (int i = 1; i < argc; i += 2) {
            if (i + 1 >= argc) throw std::runtime_error("missing option value");
            const std::string option = argv[i];
            if (option == "--mem") device = argv[i + 1];
            else if (option == "--base") base = hex(argv[i + 1]);
            else throw std::runtime_error("unknown option");
        }
        Registers bus(device, base);
        std::cout << "OTFREG1\n" << std::flush;
        std::string line;
        while (std::getline(std::cin, line)) {
            std::istringstream input(line);
            std::string op, address, value, extra;
            input >> op >> address;
            if (op == "r" && !(input >> extra)) {
                std::cout << "ok " << std::hex << bus.read(hex(address)) << '\n' << std::flush;
            } else if (op == "w" && (input >> value) && !(input >> extra)) {
                bus.write(hex(address), hex(value));
                std::cout << "ok\n" << std::flush;
            } else {
                throw std::runtime_error("invalid register command");
            }
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
