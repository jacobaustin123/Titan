if (TITAN_ENABLE_TEST)
    add_library(titan_gtest_main STATIC ${GTEST_DIR}/src/gtest-all.cc ${GTEST_DIR}/src/gtest_main.cc)
    target_compile_features(titan_gtest_main PRIVATE cxx_std_11)
    target_include_directories(titan_gtest_main PRIVATE ${GTEST_DIR}/include ${GTEST_DIR})
endif()

