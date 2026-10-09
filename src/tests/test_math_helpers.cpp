#include "test_framework.hpp"
#include "../helper.hpp"
#include "../DirectoryFunctions.hpp"

TEST_CASE(TestToStringFromString) {
    ASSERT_EQ(ToString<int>(42), "42");
    ASSERT_EQ(ToString<int>(-100), "-100");
    ASSERT_EQ(FromString<int>("42"), 42);
    ASSERT_EQ(FromString<int>("-100"), -100);

    ASSERT_NEAR(FromString<float>("3.14159"), 3.14159f, 1e-4);
    ASSERT_NEAR(FromString<double>("0.000123"), 0.000123, 1e-6);
}

TEST_CASE(TestFormatWithCommas) {
    ASSERT_EQ(FormatWithCommas(100), "100");
    ASSERT_EQ(FormatWithCommas(1000), "1,000");
    ASSERT_EQ(FormatWithCommas(1000000), "1,000,000");
    ASSERT_EQ(FormatWithCommas(-54321), "-54,321");
}

TEST_CASE(TestEnsureDirHasTrailingBackslash) {
    string d1 = "/path/to/dir";
    EnsureDirHasTrailingBackslash(d1);
    ASSERT_EQ(d1, "/path/to/dir/");

    string d2 = "/already/terminated/";
    EnsureDirHasTrailingBackslash(d2);
    ASSERT_EQ(d2, "/already/terminated/");

    string d3 = "path";
    EnsureDirHasTrailingBackslash(d3);
    ASSERT_EQ(d3, "path/");
}
