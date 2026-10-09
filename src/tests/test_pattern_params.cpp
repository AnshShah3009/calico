#include "test_framework.hpp"
#include "../pattern-parameters.hpp"
#include <fstream>

TEST_CASE(TestNormalizeAprilFamily) {
    ASSERT_EQ(NormalizeAprilFamily("tag36h11"), "tag36h11");
    ASSERT_EQ(NormalizeAprilFamily("36h11"), "tag36h11");
    ASSERT_EQ(NormalizeAprilFamily("tag25h9"), "tag25h9");
    ASSERT_EQ(NormalizeAprilFamily("25h9"), "tag25h9");
    ASSERT_EQ(NormalizeAprilFamily("tag16h5"), "tag16h5");
    ASSERT_EQ(NormalizeAprilFamily("16h5"), "tag16h5");
    ASSERT_EQ(NormalizeAprilFamily("tag36h10"), "tag36h10");
    ASSERT_EQ(NormalizeAprilFamily("36h10"), "tag36h10");
}

TEST_CASE(TestGetArucoDictionarySize) {
    ASSERT_EQ(GetArucoDictionarySize(0), 50);   // DICT_4X4_50
    ASSERT_EQ(GetArucoDictionarySize(1), 100);  // DICT_4X4_100
    ASSERT_EQ(GetArucoDictionarySize(2), 250);  // DICT_4X4_250
    ASSERT_EQ(GetArucoDictionarySize(3), 1000); // DICT_4X4_1000
    ASSERT_EQ(GetArucoDictionarySize(8), 50);   // DICT_6X6_50
    ASSERT_EQ(GetArucoDictionarySize(10), 250); // DICT_6X6_250
    ASSERT_EQ(GetArucoDictionarySize(999), -1); // invalid code returns -1
}

TEST_CASE(TestReadAprilTagSpecification) {
    const std::string tmp_yaml = "/tmp/test_april_spec.yaml";
    std::ofstream ofs(tmp_yaml);
    ofs << "%YAML:1.0\n"
        << "squaresX: 6\n"
        << "squaresY: 6\n"
        << "squareLength: 80\n"
        << "margins: 10\n"
        << "tagSpace: 40\n"
        << "numberBoards: 2\n"
        << "type: 2\n"
        << "april_family: \"tag36h11\"\n";
    ofs.close();

    patternParameters pp;
    bool ok = readAprilTagSpecificationFile(tmp_yaml, pp);
    ASSERT_TRUE(ok);
    ASSERT_EQ(pp.squaresX, 6);
    ASSERT_EQ(pp.squaresY, 6);
    ASSERT_EQ(pp.squareLength, 80);
    ASSERT_NEAR(pp.tagSpace, 40.0f, 1e-4f);
    ASSERT_EQ(pp.numberBoards, 2);
    ASSERT_EQ(pp.april_family, "tag36h11");
}

TEST_CASE(TestReadCharucoSpecification) {
    const std::string tmp_yaml = "/tmp/test_charuco_spec.yaml";
    std::ofstream ofs(tmp_yaml);
    ofs << "%YAML:1.0\n"
        << "squaresX: 5\n"
        << "squaresY: 7\n"
        << "squareLength: 40\n"
        << "markerLength: 30\n"
        << "numberBoards: 1\n"
        << "arcCode: 10\n"
        << "margins: 5\n";
    ofs.close();

    patternParameters pp;
    bool ok = readCharucoSpecificationFile(tmp_yaml, pp);
    ASSERT_TRUE(ok);
    ASSERT_EQ(pp.squaresX, 5);
    ASSERT_EQ(pp.squaresY, 7);
    ASSERT_EQ(pp.squareLength, 40);
    ASSERT_EQ(pp.markerLength, 30);
    ASSERT_EQ(pp.numberBoards, 1);
    ASSERT_EQ(pp.arc_code, 10);
}

TEST_CASE(TestReadNonExistentSpecification) {
    patternParameters pp;
    bool ok_april = readAprilTagSpecificationFile("/tmp/does_not_exist_98765.yaml", pp);
    ASSERT_FALSE(ok_april);

    bool ok_charuco = readCharucoSpecificationFile("/tmp/does_not_exist_98765.yaml", pp);
    ASSERT_FALSE(ok_charuco);
}

TEST_CASE(TestValidatePatternParams) {
    patternParameters bad1;
    bad1.squaresX = 0;
    bad1.squaresY = 5;
    bad1.squareLength = 100;
    bad1.markerLength = 80;
    bad1.numberBoards = 1;
    ASSERT_FALSE(ValidatePatternParams(bad1, "spec.yaml"));

    patternParameters bad2;
    bad2.squaresX = 5;
    bad2.squaresY = 5;
    bad2.squareLength = 80;
    bad2.markerLength = 85; // marker > square!
    bad2.numberBoards = 1;
    ASSERT_FALSE(ValidatePatternParams(bad2, "spec.yaml"));

    patternParameters good;
    good.squaresX = 5;
    good.squaresY = 5;
    good.squareLength = 80;
    good.markerLength = 60;
    good.numberBoards = 1;
    good.type = charuco;
    good.arc_code = 0;
    ASSERT_TRUE(ValidatePatternParams(good, "spec.yaml"));
}
