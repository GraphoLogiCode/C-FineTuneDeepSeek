#include "AIInterfaceKey.h"
#include <cstdlib>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

static unsigned long assertions = 0;
static void require(bool condition, const char* operation, std::size_t i, std::size_t j) {
    ++assertions;
    if (!condition) {
        std::cerr << "FAILED " << operation << " indices=" << i << "," << j
                  << " assertion=" << assertions << "\n";
        std::exit(7);
    }
}

int main() {
    const std::vector<std::string> names = {
        "", "A", "AI", "ai", "B", "name with spaces",
        std::string("\0", 1), std::string("A\0B", 3), std::string(512, 'x')
    };
    const std::vector<std::string> versions = {
        "", "0", "1", "01", "1.0", "2",
        std::string("\0", 1), std::string("v\0x", 3)
    };
    std::vector<std::pair<std::string, std::string>> expected;
    std::vector<AIInterfaceKey> keys;
    for (const auto& name : names) {
        for (const auto& version : versions) {
            expected.emplace_back(name, version);
            keys.emplace_back(name, version);
        }
    }
    AIInterfaceKey empty;
    require(empty.GetShortName().empty(), "default name", 0, 0);
    require(empty.GetVersion().empty(), "default version", 0, 0);
    require(empty.IsUnspecified(), "default unspecified", 0, 0);
    require(empty == AIInterfaceKey("", ""), "default identity", 0, 0);
    for (std::size_t i = 0; i < keys.size(); ++i) {
        const auto& key = keys[i];
        require(key.GetShortName() == expected[i].first, "get name", i, i);
        require(key.GetVersion() == expected[i].second, "get version", i, i);
        require(key.IsUnspecified() == expected[i].first.empty(), "unspecified", i, i);
        require(key.ToString() == expected[i].first + " " + expected[i].second, "string", i, i);
        const AIInterfaceKey copied(key);
        require(copied == key, "copy identity", i, i);
        for (std::size_t j = 0; j < keys.size(); ++j) {
            const auto& other = keys[j];
            require((key == other) == (expected[i] == expected[j]), "==", i, j);
            require((key != other) == (expected[i] != expected[j]), "!=", i, j);
            require((key < other) == (expected[i] < expected[j]), "<", i, j);
            require((key > other) == (expected[i] > expected[j]), ">", i, j);
            require((key <= other) == (expected[i] <= expected[j]), "<=", i, j);
            require((key >= other) == (expected[i] >= expected[j]), ">=", i, j);
        }
    }
    std::cout << "PASS assertions=" << assertions << " keys=" << keys.size()
              << " pairs=" << keys.size() * keys.size() << "\n";
    return assertions == 31468 ? 0 : 9;
}
