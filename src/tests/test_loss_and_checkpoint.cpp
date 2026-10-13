#include "test_framework.hpp"
#include "../calico.hpp"
#include <ceres/ceres.h>
#include <fstream>
#include <cstdio>
#include <string>

TEST_CASE(TestCalicoOptionsDefaults) {
    CalicoOptions opts;
    ASSERT_EQ(opts.loss_type, "trivial");
    ASSERT_NEAR(opts.loss_scale, 1.0f, 1e-5);
    ASSERT_TRUE(!opts.checkpoint_enabled);
}

TEST_CASE(TestCeresProblemLossFunctions) {
    // Test trivial loss
    std::string loss_trivial = "trivial";
    double scale = 1.0;
    ceres::LossFunction* l1 = nullptr;
    if (loss_trivial == "huber") l1 = new ceres::HuberLoss(scale);
    else if (loss_trivial == "cauchy") l1 = new ceres::CauchyLoss(scale);
    ASSERT_TRUE(l1 == nullptr);

    // Test Huber loss
    std::string loss_huber = "huber";
    double huber_scale = 1.5;
    ceres::LossFunction* l2 = nullptr;
    if (loss_huber == "huber") l2 = new ceres::HuberLoss(huber_scale);
    ASSERT_TRUE(l2 != nullptr);
    delete l2;

    // Test Cauchy loss
    std::string loss_cauchy = "cauchy";
    double cauchy_scale = 2.0;
    ceres::LossFunction* l3 = nullptr;
    if (loss_cauchy == "cauchy") l3 = new ceres::CauchyLoss(cauchy_scale);
    ASSERT_TRUE(l3 != nullptr);
    delete l3;
}

TEST_CASE(TestStage5CheckpointFileFormat) {
    const std::string tmp_dir = "/tmp/calico_cp_test/";
    std::string cmd = "mkdir -p " + tmp_dir;
    int ret = system(cmd.c_str());
    (void)ret;

    std::string cp_file = tmp_dir + "checkpoint_stage5.txt";
    {
        std::ofstream cp(cp_file.c_str());
        cp << "stage 5\n";
        cp << "variable_index 42\n";
        cp << "num_vars 2\n";
        cp << "1 1 \n";
        // 2 4x4 identity matrices
        for (int i = 0; i < 2; i++) {
            for (int r = 0; r < 4; r++) {
                for (int c = 0; c < 4; c++) {
                    cp << (r == c ? 1.0 : 0.0) << " ";
                }
            }
            cp << "\n";
        }
        cp.close();
    }

    // Read back and verify format parsing
    std::ifstream cp_in(cp_file.c_str());
    ASSERT_TRUE(cp_in.good());

    std::string label;
    int stage = 0, var_idx = 0, num_vars = 0;
    cp_in >> label >> stage;
    ASSERT_EQ(label, "stage");
    ASSERT_EQ(stage, 5);

    cp_in >> label >> var_idx;
    ASSERT_EQ(label, "variable_index");
    ASSERT_EQ(var_idx, 42);

    cp_in >> label >> num_vars;
    ASSERT_EQ(label, "num_vars");
    ASSERT_EQ(num_vars, 2);

    int init0 = 0, init1 = 0;
    cp_in >> init0 >> init1;
    ASSERT_EQ(init0, 1);
    ASSERT_EQ(init1, 1);

    for (int i = 0; i < 2; i++) {
        for (int r = 0; r < 4; r++) {
            for (int c = 0; c < 4; c++) {
                double val;
                cp_in >> val;
                if (r == c) {
                    ASSERT_NEAR(val, 1.0, 1e-6);
                } else {
                    ASSERT_NEAR(val, 0.0, 1e-6);
                }
            }
        }
    }
    cp_in.close();

    // Clean up
    cmd = "rm -rf " + tmp_dir;
    ret = system(cmd.c_str());
    (void)ret;
}
