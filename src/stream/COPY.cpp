//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-18, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-738930
//
// All rights reserved.
//
// This file is part of the RAJA Performance Suite.
//
// For details about use and distribution, please read raja-perfsuite/LICENSE.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "COPY.hpp"

#include "common/RAJAPerfSuite.hpp"

#include "common/DataUtils.hpp"

#include <iostream>

namespace rajaperf 
{
namespace stream
{


#define COPY_DATA_SETUP_CPU \
  ResReal_ptr a = m_a; \
  ResReal_ptr c = m_c;


COPY::COPY(const RunParams& params)
  : KernelBase(rajaperf::Stream_COPY, params)
{
   setDefaultSize(1000000);
   setDefaultReps(1800);
}

COPY::~COPY() 
{
}

void COPY::setUp(VariantID vid)
{
  allocAndInitData(m_a, getRunSize(), vid);
  allocAndInitDataConst(m_c, getRunSize(), 0.0, vid);
}

void COPY::runKernel(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getRunSize();

  switch ( vid ) {

    case Base_Seq : {

      COPY_DATA_SETUP_CPU;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (Index_type i = ibegin; i < iend; ++i ) {
          COPY_BODY;
        }

      }
      stopTimer();

      break;
    }

#ifdef RAJAPERF_ENABLE_RAJA
    case RAJA_Seq : {

      COPY_DATA_SETUP_CPU;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::forall<RAJA::simd_exec>(
          RAJA::RangeSegment(ibegin, iend), [=](Index_type i) {
          COPY_BODY;
        });

      }
      stopTimer();

      break;
    }
#endif // RAJPERF_ENABLE_RAJA
#ifdef RAJAPERF_ENABLE_KOKKOS
    case Kokkos_Lambda_Seq : {

      COPY_DATA_SETUP_CPU;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        Kokkos::parallel_for("perfsuite.stream.kokkos.seq.lambda.copy",
          Kokkos::RangePolicy<Kokkos::Serial>(ibegin, iend), [=](Index_type i) {
          COPY_BODY;
        });

      }
      stopTimer();

      break;
    }
#endif //RAJAPERF_ENABLE_KOKKOS
#if defined(RAJA_ENABLE_OPENMP)
    case Base_OpenMP : {

      COPY_DATA_SETUP_CPU;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        #pragma omp parallel for
        for (Index_type i = ibegin; i < iend; ++i ) {
          COPY_BODY;
        }

      }
      stopTimer();

      break;
    }

#ifdef RAJAPERF_ENABLE_RAJA
    case RAJA_OpenMP : {

      COPY_DATA_SETUP_CPU;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::forall<RAJA::omp_parallel_for_exec>(
          RAJA::RangeSegment(ibegin, iend), [=](Index_type i) {
          COPY_BODY;
        });

      }
      stopTimer();

      break;
    }
#endif // RAJPERF_ENABLE_RAJA
#ifdef RAJAPERF_ENABLE_KOKKOS
    case Kokkos_Lambda_OpenMP : {

      COPY_DATA_SETUP_CPU;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        Kokkos::parallel_for("perfsuite.stream.kokkos.openmp.lambda.copy",
          Kokkos::RangePolicy<Kokkos::OpenMP>(ibegin, iend), [=](Index_type i) {
          COPY_BODY;
        });

      }
      stopTimer();

      break;
    }
#endif //RAJAPERF_ENABLE_KOKKOS
#endif

#if defined(RAJA_ENABLE_TARGET_OPENMP)
    case Base_OpenMPTarget :
#ifdef RAJAPERF_ENABLE_RAJA
    case RAJA_OpenMPTarget :
    {
      runOpenMPTargetVariant(vid);
      break;
    }
#endif // RAJPERF_ENABLE_RAJA
#endif

#if defined(RAJA_ENABLE_CUDA)
    case Base_CUDA :
#ifdef RAJAPERF_ENABLE_RAJA
    case RAJA_CUDA :
    {
      runCudaVariant(vid);
      break;
    }
#endif // RAJPERF_ENABLE_RAJA
#endif

    default : {
      std::cout << "\n  COPY : Unknown variant id = " << vid << std::endl;
    }

  }

}

void COPY::updateChecksum(VariantID vid)
{
  checksum[vid] += calcChecksum(m_c, getRunSize());
}

void COPY::tearDown(VariantID vid)
{
  (void) vid;
  deallocData(m_a);
  deallocData(m_c);
}

} // end namespace stream
} // end namespace rajaperf
