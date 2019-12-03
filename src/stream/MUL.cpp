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

#include "MUL.hpp"

#include "common/RAJAPerfSuite.hpp"

#include "common/DataUtils.hpp"

#include <iostream>

namespace rajaperf 
{
namespace stream
{


#define MUL_DATA_SETUP_CPU \
  ResReal_ptr b = m_b; \
  ResReal_ptr c = m_c; \
  Real_type alpha = m_alpha;

struct MulFunctor {

  ResReal_ptr b;
  ResReal_ptr c;

  const Real_type alpha;

  MulFunctor(ResReal_ptr b,ResReal_ptr c, Real_type alpha) : b(b), c(c), alpha(alpha) {}

  void operator()(const Index_type i) const {
    MUL_BODY;
  }

};


MUL::MUL(const RunParams& params)
  : KernelBase(rajaperf::Stream_MUL, params)
{
   setDefaultSize(1000000);
   setDefaultReps(1800);
}

MUL::~MUL() 
{

}

void MUL::setUp(VariantID vid)
{
  allocAndInitDataConst(m_b, getRunSize(), 0.0, vid);
  allocAndInitData(m_c, getRunSize(), vid);
  initData(m_alpha, vid);
}

void MUL::runKernel(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getRunSize();

  switch ( vid ) {

    case Base_Seq : {

      MUL_DATA_SETUP_CPU;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (Index_type i = ibegin; i < iend; ++i ) {
          MUL_BODY;
        }

      }
      stopTimer();

      break;
    }

#ifdef RAJAPERF_ENABLE_RAJA
    case RAJA_Seq : {

      MUL_DATA_SETUP_CPU;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::forall<RAJA::simd_exec>(
          RAJA::RangeSegment(ibegin, iend), [=](Index_type i) {
          MUL_BODY;
        });

      }
      stopTimer();

      break;
    }
#endif // RAJPERF_ENABLE_RAJA
#ifdef RAJAPERF_ENABLE_KOKKOS
    case Kokkos_Lambda_Seq : {

      MUL_DATA_SETUP_CPU;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        Kokkos::parallel_for("put.profiling.string.here",
          Kokkos::RangePolicy<Kokkos::Serial>(ibegin, iend), [=](Index_type i) {
          MUL_BODY;
        });

      }
      stopTimer();

      break;
    }
    case Kokkos_Functor_Seq : {

      MUL_DATA_SETUP_CPU;
      MulFunctor mul_functor(b,c,alpha);
      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        Kokkos::parallel_for("put.profiling.string.here",
          Kokkos::RangePolicy<Kokkos::Serial>(ibegin, iend), mul_functor);

      }
      stopTimer();

      break;
    }
#endif // RAJPERF_ENABLE_KOKKOS

#if defined(RAJA_ENABLE_OPENMP)
    case Base_OpenMP : {

      MUL_DATA_SETUP_CPU;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        #pragma omp parallel for
        for (Index_type i = ibegin; i < iend; ++i ) {
          MUL_BODY;
        }

      }
      stopTimer();

      break;
    }

#ifdef RAJAPERF_ENABLE_RAJA
    case RAJA_OpenMP : {

      MUL_DATA_SETUP_CPU;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::forall<RAJA::omp_parallel_for_exec>(
          RAJA::RangeSegment(ibegin, iend), [=](Index_type i) {
          MUL_BODY;
        });

      }
      stopTimer();

      break;
    }
#endif // RAJPERF_ENABLE_RAJA
#ifdef RAJAPERF_ENABLE_KOKKOS
    case Kokkos_Lambda_OpenMP : {

      MUL_DATA_SETUP_CPU;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        Kokkos::parallel_for("put.profiling.string.here",
          Kokkos::RangePolicy<Kokkos::OpenMP>(ibegin, iend), [=](Index_type i) {
          MUL_BODY;
        });

      }
      stopTimer();

      break;
    }
    case Kokkos_Functor_OpenMP : {

      MUL_DATA_SETUP_CPU;
      MulFunctor mul_functor(b,c,alpha);
      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        Kokkos::parallel_for("put.profiling.string.here",
          Kokkos::RangePolicy<Kokkos::OpenMP>(ibegin, iend), mul_functor);

      }
      stopTimer();

      break;
    }
#endif // RAJPERF_ENABLE_KOKKOS
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
      std::cout << "\n  MUL : Unknown variant id = " << vid << std::endl;
    }

  }

}

void MUL::updateChecksum(VariantID vid)
{
  checksum[vid] += calcChecksum(m_b, getRunSize());
}

void MUL::tearDown(VariantID vid)
{
  (void) vid;
  deallocData(m_b);
  deallocData(m_c);
}

} // end namespace stream
} // end namespace rajaperf
