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

#include "ADD.hpp"

#include "common/RajaPerfSuite.hpp"

#include "common/DataUtils.hpp"

#include <iostream>

namespace rajaperf 
{
namespace stream
{

 
#define ADD_DATA_SETUP_CPU \
  ResReal_ptr a = m_a; \
  ResReal_ptr b = m_b; \
  ResReal_ptr c = m_c;


ADD::ADD(const RunParams& params)
  : KernelBase(rajaperf::Stream_ADD, params)
{
   setDefaultSize(1000000);
   setDefaultReps(1000);
}

ADD::~ADD() 
{
}

void ADD::setUp(VariantID vid)
{
  allocAndInitData(m_a, getRunSize(), vid);
  allocAndInitData(m_b, getRunSize(), vid);
  allocAndInitDataConst(m_c, getRunSize(), 0.0, vid);
}

void ADD::runKernel(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getRunSize();

  switch ( vid ) {

    case Base_Seq : {

      ADD_DATA_SETUP_CPU;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (Index_type i = ibegin; i < iend; ++i ) {
          ADD_BODY;
        }

      }
      stopTimer();

      break;
    }

#ifdef RAJAPERF_ENABLE_RAJA
    case RAJA_Seq : {

      ADD_DATA_SETUP_CPU;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::forall<RAJA::simd_exec>(
          RAJA::RangeSegment(ibegin, iend), [=](Index_type i) {
          ADD_BODY;
        });

      }
      stopTimer();

      break;
    }
#endif // RAJPERF_ENABLE_RAJA
#ifdef RAJAPERF_ENABLE_KOKKOS
   case Kokkos_Lambda_Seq: {
     ADD_DATA_SETUP_CPU;
     startTimer();
     for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

       Kokkos::parallel_for("perfsuite.stream.add",
         Kokkos::RangePolicy<Kokkos::Serial>(ibegin, iend), [=](Index_type i) {
         ADD_BODY;
       });

     }
     
     stopTimer();
     break;
   }
#endif
#if defined(RAJA_ENABLE_OPENMP)
    case Base_OpenMP : {

      ADD_DATA_SETUP_CPU;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        #pragma omp parallel for
        for (Index_type i = ibegin; i < iend; ++i ) {
          ADD_BODY;
        }

      }
      stopTimer();

      break;
    }

#ifdef RAJAPERF_ENABLE_RAJA
    case RAJA_OpenMP : {

      ADD_DATA_SETUP_CPU;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::forall<RAJA::omp_parallel_for_exec>(
          RAJA::RangeSegment(ibegin, iend), [=](Index_type i) {
          ADD_BODY;
        });

      }
      stopTimer();

      break;
    }
#endif // RAJPERF_ENABLE_RAJA
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
      std::cout << "\n  ADD : Unknown variant id = " << vid << std::endl;
    }

  }

}

void ADD::updateChecksum(VariantID vid)
{
  checksum[vid] += calcChecksum(m_c, getRunSize());
}

void ADD::tearDown(VariantID vid)
{
  (void) vid;
  deallocData(m_a);
  deallocData(m_b);
  deallocData(m_c);
}

} // end namespace stream
} // end namespace rajaperf
