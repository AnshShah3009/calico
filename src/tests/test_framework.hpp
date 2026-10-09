#ifndef TEST_FRAMEWORK_HPP_
#define TEST_FRAMEWORK_HPP_

#include <iostream>
#include <string>
#include <vector>
#include <functional>
#include <cmath>

struct TestCase {
    std::string name;
    std::function<void()> func;
};

class TestRegistry {
public:
    static TestRegistry& instance() {
        static TestRegistry reg;
        return reg;
    }

    void add(const std::string& name, std::function<void()> func) {
        tests.push_back({name, func});
    }

    int run() {
        int passed = 0;
        int failed = 0;
        std::cout << "Running " << tests.size() << " unit test(s)..." << std::endl;
        for (const auto& t : tests) {
            std::cout << "  [ RUN      ] " << t.name << std::endl;
            try {
                t.func();
                std::cout << "  [       OK ] " << t.name << std::endl;
                passed++;
            } catch (const std::exception& e) {
                std::cerr << "  [  FAILED  ] " << t.name << ": " << e.what() << std::endl;
                failed++;
            } catch (...) {
                std::cerr << "  [  FAILED  ] " << t.name << ": unknown exception" << std::endl;
                failed++;
            }
        }
        std::cout << "========================================" << std::endl;
        std::cout << "Total: " << tests.size() << " | Passed: " << passed << " | Failed: " << failed << std::endl;
        return failed == 0 ? 0 : 1;
    }

private:
    std::vector<TestCase> tests;
};

#define TEST_CASE(name) \
    static void name(); \
    namespace { \
        struct Register_##name { \
            Register_##name() { \
                TestRegistry::instance().add(#name, name); \
            } \
        } reg_##name; \
    } \
    static void name()

#define ASSERT_TRUE(cond) \
    do { \
        if (!(cond)) { \
            throw std::runtime_error(std::string("Assertion failed: ") + #cond + " at " + __FILE__ + ":" + std::to_string(__LINE__)); \
        } \
    } while (0)

#define ASSERT_FALSE(cond) ASSERT_TRUE(!(cond))

#define ASSERT_EQ(a, b) \
    do { \
        if ((a) != (b)) { \
            throw std::runtime_error(std::string("Assertion failed: ") + #a + " == " + #b + " at " + __FILE__ + ":" + std::to_string(__LINE__)); \
        } \
    } while (0)

#define ASSERT_NEAR(a, b, eps) \
    do { \
        if (std::abs((a) - (b)) > (eps)) { \
            throw std::runtime_error(std::string("Assertion failed: |") + #a + " - " + #b + "| <= " + #eps + " at " + __FILE__ + ":" + std::to_string(__LINE__)); \
        } \
    } while (0)

#endif // TEST_FRAMEWORK_HPP_
