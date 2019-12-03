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

#include "DOT.hpp"

#include "common/RAJAPerfSuite.hpp"

#include "common/DataUtils.hpp"

#include <iostream>

namespace rajaperf 
{
namespace stream
{


#define DOT_DATA_SETUP_CPU \
  ResReal_ptr a = m_a; \
  ResReal_ptr b = m_b;

struct DotFunctor {
  ResReal_ptr a;
  ResReal_ptr b;

  DotFunctor (ResReal_ptr a, ResReal_ptr b) : a(a), b(b) {}
 
  void operator()(const Index_type i, Real_type& dot) const {
    DOT_BODY;
  }

};

DOT::DOT(const RunParams& params)
  : KernelBase(rajaperf::Stream_DOT, params)
{
   setDefaultSize(1000000);
   setDefaultReps(2000);
}

DOT::~DOT() 
{
}

void DOT::setUp(VariantID vid)
{
  allocAndInitData(m_a, getRunSize(), vid);
  allocAndInitData(m_b, getRunSize(), vid);

  m_dot = 0.0;
  m_dot_init = 0.0;
}

void DOT::runKernel(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getRunSize();

  switch ( vid ) {

    case Base_Seq : {

      DOT_DATA_SETUP_CPU;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        Real_type dot = m_dot_init;

        for (Index_type i = ibegin; i < iend; ++i ) {
          DOT_BODY;
        }

         m_dot += dot;

      }
      stopTimer();

      break;
    }

#ifdef RAJAPERF_ENABLE_RAJA
    case RAJA_Seq : {

      DOT_DATA_SETUP_CPU;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::ReduceSum<RAJA::seq_reduce, Real_type> dot(m_dot_init);

        RAJA::forall<RAJA::loop_exec>(
          RAJA::RangeSegment(ibegin, iend), [=](Index_type i) {
          DOT_BODY;
        });

        m_dot += static_cast<Real_type>(dot.get());

      }
      stopTimer();

      break;
    }
#endif // RAJPERF_ENABLE_RAJA
#ifdef RAJAPERF_ENABLE_KOKKOS
    case Kokkos_Lambda_Seq : {

      DOT_DATA_SETUP_CPU;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        //RAJA::ReduceSum<RAJA::seq_reduce, Real_type> dot(m_dot_init);
        Real_type dot;
        Kokkos::parallel_reduce("put.profiling.string.here",
          Kokkos::RangePolicy<Kokkos::Serial>(ibegin, iend), [=](const Index_type& i, Real_type& dot) {
          DOT_BODY;
        }, dot);

        m_dot += dot;

      }
      stopTimer();

      break;
    }
    case Kokkos_Functor_Seq : {

      DOT_DATA_SETUP_CPU;
      DotFunctor dot_functor(a,b);
      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        //RAJA::ReduceSum<RAJA::seq_reduce, Real_type> dot(m_dot_init);
        Real_type dot;
        Kokkos::parallel_reduce("put.profiling.string.here",
          Kokkos::RangePolicy<Kokkos::Serial>(ibegin, iend), dot_functor, dot);

        m_dot += dot;

      }
      stopTimer();

      break;
    }
#endif // RAJPERF_ENABLE_KOKKOS

#if defined(RAJA_ENABLE_OPENMP)
    case Base_OpenMP : {

      DOT_DATA_SETUP_CPU;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        Real_type dot = m_dot_init;

        #pragma omp parallel for reduction(+:dot)
        for (Index_type i = ibegin; i < iend; ++i ) {
          DOT_BODY;
        }

        m_dot += dot;

      }
      stopTimer();

      break;
    }

#ifdef RAJAPERF_ENABLE_RAJA
    case RAJA_OpenMP : {

      DOT_DATA_SETUP_CPU;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::ReduceSum<RAJA::omp_reduce, Real_type> dot(m_dot_init);

        RAJA::forall<RAJA::omp_parallel_for_exec>(
          RAJA::RangeSegment(ibegin, iend), [=](Index_type i) {
          DOT_BODY;
        });

        m_dot += dot;

      }
      stopTimer();

      break;
    }
#endif // RAJPERF_ENABLE_RAJA
#ifdef RAJAPERF_ENABLE_KOKKOS
    case Kokkos_Lambda_OpenMP : {

      DOT_DATA_SETUP_CPU;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        //RAJA::ReduceSum<RAJA::seq_reduce, Real_type> dot(m_dot_init);
        Real_type dot;
        Kokkos::parallel_reduce("put.profiling.string.here",
          Kokkos::RangePolicy<Kokkos::OpenMP>(ibegin, iend), [=](const Index_type& i, Real_type& dot) {
          DOT_BODY;
        }, dot);

        m_dot += dot;

      }
      stopTimer();

      break;
    }
    case Kokkos_Functor_OpenMP : {

      DOT_DATA_SETUP_CPU;
      DotFunctor dot_functor(a,b);
      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        Real_type dot;
        Kokkos::parallel_reduce("put.profiling.string.here",
          Kokkos::RangePolicy<Kokkos::OpenMP>(ibegin, iend), dot_functor, dot);

        m_dot += dot;

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
      std::cout << "\n  DOT : Unknown variant id = " << vid << std::endl;
    }

  }

}

void DOT::updateChecksum(VariantID vid)
{
  checksum[vid] += m_dot;
}

void DOT::tearDown(VariantID vid)
{
  (void) vid;
  deallocData(m_a);
  deallocData(m_b);
}

} // end namespace stream
} // end namespace rajaperf
