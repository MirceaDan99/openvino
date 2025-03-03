// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <common_test_utils/test_constants.hpp>
#include <single_layer_tests/rdft.hpp>
#include <vector>

namespace {

const std::vector<ngraph::helpers::DFTOpType> opTypes = {
    ngraph::helpers::DFTOpType::FORWARD,
    ngraph::helpers::DFTOpType::INVERSE,
};

const std::vector<InferenceEngine::Precision> inputPrecisions = {
    InferenceEngine::Precision::FP32,
    InferenceEngine::Precision::FP16,
};

const auto combine = [](const std::vector<InferenceEngine::SizeVector>& inputShapes,
                        const std::vector<std::vector<int64_t>>& axes,
                        const std::vector<std::vector<int64_t>>& signalSizes) {
    return testing::Combine(testing::ValuesIn(inputShapes),
                            testing::ValuesIn(inputPrecisions),
                            testing::ValuesIn(axes),
                            testing::ValuesIn(signalSizes),
                            testing::ValuesIn(opTypes),
                            testing::Values(ov::test::utils::DEVICE_GPU));
};

using namespace LayerTestsDefinitions;

// RDFT can support 1d
INSTANTIATE_TEST_SUITE_P(smoke_RDFT_1d,
                         RDFTLayerTest,
                         testing::Combine(testing::Values(InferenceEngine::SizeVector{10}),
                                          testing::ValuesIn(inputPrecisions),
                                          testing::Values(std::vector<int64_t>{0}),
                                          testing::Values(std::vector<int64_t>{}),
                                          testing::Values(ngraph::helpers::DFTOpType::FORWARD),
                                          testing::Values(ov::test::utils::DEVICE_GPU)),
                         RDFTLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_RDFT_3d,
                         RDFTLayerTest,
                         combine({{10, 4, 2}},    // input shapes
                                 {{0, 1}},        // axes
                                 {{}, {3, 10}}),  // signal sizes
                         RDFTLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_RDFT_4d,
                         RDFTLayerTest,
                         combine({{10, 4, 8, 2}},    // input shapes
                                 {{0, 1, 2}},        // axes
                                 {{}, {3, 10, 8}}),  // signal sizes
                         RDFTLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_RDFT_4d_negative_reversed_axes,
                         RDFTLayerTest,
                         combine({{10, 4, 8, 2}},    // input shapes
                                 {{-1, -2, -3}},     // axes
                                 {{}, {8, 10, 3}}),  // signal sizes
                         RDFTLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_RDFT_4d_single_axis,
                         RDFTLayerTest,
                         combine({{10, 4, 8, 2}},        // input shapes
                                 {{0}, {1}, {2}},        // axes
                                 {{}, {1}, {5}, {20}}),  // signal sizes
                         RDFTLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_RDFT_5d,
                         RDFTLayerTest,
                         combine({{10, 4, 8, 2, 2}},    // input shapes
                                 {{0, 1, 2, 3}},        // axes
                                 {{}, {3, 10, 8, 6}}),  // signal sizes
                         RDFTLayerTest::getTestCaseName);

// RDFT can support last axis
INSTANTIATE_TEST_SUITE_P(smoke_RDFT_5d_last_axis,
                         RDFTLayerTest,
                         testing::Combine(testing::Values(InferenceEngine::SizeVector{10, 4, 8, 2, 5}),
                                          testing::ValuesIn(inputPrecisions),
                                          testing::ValuesIn(std::vector<std::vector<int64_t>>{{{0, 1, 2, 3, 4}}}),
                                          testing::ValuesIn(std::vector<std::vector<int64_t>>{{}, {3, 10, 8, 6, 2}}),
                                          testing::Values(ngraph::helpers::DFTOpType::FORWARD),
                                          testing::Values(ov::test::utils::DEVICE_GPU)),
                         RDFTLayerTest::getTestCaseName);

// IRDFT can support 6d
INSTANTIATE_TEST_SUITE_P(smoke_RDFT_6d,
                         RDFTLayerTest,
                         testing::Combine(testing::Values(InferenceEngine::SizeVector{10, 4, 8, 2, 5, 2}),
                                          testing::ValuesIn(inputPrecisions),
                                          testing::ValuesIn(std::vector<std::vector<int64_t>>{{{0, 1, 2, 3, 4}}}),
                                          testing::ValuesIn(std::vector<std::vector<int64_t>>{{}, {3, 10, 8, 6, 2}}),
                                          testing::Values(ngraph::helpers::DFTOpType::INVERSE),
                                          testing::Values(ov::test::utils::DEVICE_GPU)),
                         RDFTLayerTest::getTestCaseName);

}  // namespace
