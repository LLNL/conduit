///
/// file: conduit_mpi.cpp
///

#include "gtest/gtest.h"
#include <mpi.h>

TEST(conduit_mpi_smoke, conduit_mpi)
{
    // simple mpi test 
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    EXPECT_EQ(size, 2);
}

int main(int argc, char* argv[]) {
    int result = 0;

    ::testing::InitGoogleTest(&argc, argv);
    MPI_Init(&argc, &argv);
    result = RUN_ALL_TESTS();
    MPI_Finalize();

    return result;
}