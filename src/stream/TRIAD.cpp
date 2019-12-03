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

#include "TRIAD.hpp"

#include "common/RAJAPerfSuite.hpp"

#include "common/DataUtils.hpp"

#include <iostream>

namespace rajaperf 
{
namespace stream
{


#define TRIAD_DATA_SETUP_CPU \
  ResReal_ptr a = m_a; \
  ResReal_ptr b = m_b; \
  ResReal_ptr c = m_c; \
  Real_type alpha = m_alpha;

struct TriadFunctor {

  ResReal_ptr a;
  ResReal_ptr b;
  ResReal_ptr c;

  const Real_type alpha;

  TriadFunctor(ResReal_ptr a,ResReal_ptr b,ResReal_ptr c, Real_type alpha) : a(a), b(b), c(c), alpha(alpha) {};
  void operator()(const Index_type i) const{
    TRIAD_BODY;
  }
};

TRIAD::TRIAD(const RunParams& params)
  : KernelBase(rajaperf::Stream_TRIAD, params)
{
   setDefaultSize(1000000);
   setDefaultReps(1000);
}

TRIAD::~TRIAD() 
{
}

void TRIAD::setUp(VariantID vid)
{
  allocAndInitDataConst(m_a, getRunSize(), 0.0, vid);
  allocAndInitData(m_b, getRunSize(), vid);
  allocAndInitData(m_c, getRunSize(), vid);
  initData(m_alpha, vid);
}

void TRIAD::runKernel(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getRunSize();

  switch ( vid ) {

    case Base_Seq : {

      TRIAD_DATA_SETUP_CPU;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (Index_type i = ibegin; i < iend; ++i ) {
          TRIAD_BODY;
        }

      }
      stopTimer();

      break;
    }

#ifdef RAJAPERF_ENABLE_RAJA
    case RAJA_Seq : {

      TRIAD_DATA_SETUP_CPU;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::forall<RAJA::simd_exec>(
          RAJA::RangeSegment(ibegin, iend), [=](Index_type i) {
          TRIAD_BODY;
        });

      }
      stopTimer();

      break;
    }
#endif // RAJPERF_ENABLE_RAJA
#ifdef RAJAPERF_ENABLE_KOKKOS
    case Kokkos_Lambda_Seq : {

      TRIAD_DATA_SETUP_CPU;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        Kokkos::parallel_for("put.profiling.string.here",
          Kokkos::RangePolicy<Kokkos::Serial>(ibegin, iend), [=](Index_type i) {
          TRIAD_BODY;
        });

      }
      stopTimer();

      break;
    }
    case Kokkos_Functor_Seq : {

      TRIAD_DATA_SETUP_CPU;
      TriadFunctor triad_functor(a,b,c,alpha);
      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        Kokkos::parallel_for("put.profiling.string.here",
          Kokkos::RangePolicy<Kokkos::Serial>(ibegin, iend), triad_functor);

      }
      stopTimer();

      break;
    }
#endif // RAJPERF_ENABLE_KOKKOS

#if defined(RAJA_ENABLE_OPENMP)
    case Base_OpenMP : {

      TRIAD_DATA_SETUP_CPU;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        #pragma omp parallel for
        for (Index_type i = ibegin; i < iend; ++i ) {
          TRIAD_BODY;
        }

      }
      stopTimer();

      break;
    }

#ifdef RAJAPERF_ENABLE_RAJA
    case RAJA_OpenMP : {

      TRIAD_DATA_SETUP_CPU;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::forall<RAJA::omp_parallel_for_exec>(
          RAJA::RangeSegment(ibegin, iend), [=](Index_type i) {
          TRIAD_BODY;
        });

      }
      stopTimer();

      break;
    }
#endif // RAJPERF_ENABLE_RAJA
#ifdef RAJAPERF_ENABLE_KOKKOS
    case Kokkos_Lambda_OpenMP : {

      TRIAD_DATA_SETUP_CPU;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        Kokkos::parallel_for("put.profiling.string.here",
          Kokkos::RangePolicy<Kokkos::OpenMP>(ibegin, iend), [=](Index_type i) {
          TRIAD_BODY;
        });

      }
      stopTimer();

      break;
    }
    case Kokkos_Functor_OpenMP : {

      TRIAD_DATA_SETUP_CPU;
      TriadFunctor triad_functor(a,b,c,alpha);
      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        Kokkos::parallel_for("put.profiling.string.here",
          Kokkos::RangePolicy<Kokkos::OpenMP>(ibegin, iend), triad_functor);

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
      std::cout << "\n  TRIAD : Unknown variant id = " << vid << std::endl;
    }

  }

}

void TRIAD::updateChecksum(VariantID vid)
{
  checksum[vid] += calcChecksum(m_a, getRunSize());
}

void TRIAD::tearDown(VariantID vid)
{
  (void) vid;
  deallocData(m_a);
  deallocData(m_b);
  deallocData(m_c);
}

} // end namespace stream
} // end namespace rajaperf
