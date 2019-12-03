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

#include "LTIMES.hpp"

#include "common/RAJAPerfSuite.hpp"

#include "common/DataUtils.hpp"

#include <iostream>

namespace rajaperf 
{
namespace apps
{


#define LTIMES_DATA_SETUP_CPU \
  ResReal_ptr phidat = m_phidat; \
  ResReal_ptr elldat = m_elldat; \
  ResReal_ptr psidat = m_psidat; \
\
  Index_type num_d = m_num_d; \
  Index_type num_z = m_num_z; \
  Index_type num_g = m_num_g; \
  Index_type num_m = m_num_m;

struct LTimesFunctor {

  ResReal_ptr phidat;
  ResReal_ptr elldat;
  ResReal_ptr psidat;

  const Index_type num_d;
  const Index_type num_z;
  const Index_type num_g;
  const Index_type num_m;

  LTimesFunctor(ResReal_ptr phidat, ResReal_ptr elldat, ResReal_ptr psidat, Index_type num_d, Index_type num_z, Index_type num_g, Index_type num_m) : phidat(phidat), elldat(elldat), psidat(psidat), num_d(num_d), num_z(num_z), num_g(num_g), num_m(num_m){} 

  void operator()(Index_type m,Index_type d, Index_type g, Index_type z) const {
    LTIMES_BODY;
  }

};

LTIMES::LTIMES(const RunParams& params)
  : KernelBase(rajaperf::Apps_LTIMES, params)
{
  m_num_d_default = 64;
  m_num_z_default = 500;
  m_num_g_default = 32;
  m_num_m_default = 25;

  setDefaultSize(m_num_d_default * m_num_m_default * 
                 m_num_g_default * m_num_z_default);
  setDefaultReps(50);
}

LTIMES::~LTIMES() 
{
}

void LTIMES::setUp(VariantID vid)
{
  m_num_z = run_params.getSizeFactor() * m_num_z_default;
  m_num_g = m_num_g_default;  
  m_num_m = m_num_m_default;  
  m_num_d = m_num_d_default;  

  m_philen = m_num_m * m_num_g * m_num_z;
  m_elllen = m_num_d * m_num_m;
  m_psilen = m_num_d * m_num_g * m_num_z;

  allocAndInitDataConst(m_phidat, int(m_philen), Real_type(0.0), vid);
  allocAndInitData(m_elldat, int(m_elllen), vid);
  allocAndInitData(m_psidat, int(m_psilen), vid);
}

void LTIMES::runKernel(VariantID vid)
{
  const Index_type run_reps = getRunReps();

  switch ( vid ) {

    case Base_Seq : {

      LTIMES_DATA_SETUP_CPU;
  
      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (Index_type z = 0; z < num_z; ++z ) {
          for (Index_type g = 0; g < num_g; ++g ) {
            for (Index_type m = 0; m < num_m; ++m ) {
              for (Index_type d = 0; d < num_d; ++d ) {
                LTIMES_BODY;
              }
            }
          }
        }

      }
      stopTimer();

      break;
    } 
    #ifdef RAJAPERF_ENABLE_RAJA
    case RAJA_Seq : {

      LTIMES_DATA_SETUP_CPU;

      LTIMES_VIEWS_RANGES_RAJA;

      using EXEC_POL = RAJA::nested::Policy<
                             RAJA::nested::For<1, RAJA::loop_exec>,
                             RAJA::nested::For<2, RAJA::loop_exec>,
                             RAJA::nested::For<3, RAJA::loop_exec>,
                             RAJA::nested::For<0, RAJA::loop_exec> >;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {
      
        RAJA::nested::forall(EXEC_POL{},
                             RAJA::make_tuple(IDRange(0, num_d),
                                              IZRange(0, num_z),
                                              IGRange(0, num_g),
                                              IMRange(0, num_m)), 
          [=](ID d, IZ z, IG g, IM m) {
          LTIMES_BODY_RAJA;
        });

      }
      stopTimer(); 

      break;
    }
    #endif 
    #ifdef RAJAPERF_ENABLE_KOKKOS
    case Kokkos_Functor_Seq : {

      LTIMES_DATA_SETUP_CPU;
      Kokkos::MDRangePolicy<Kokkos::Serial,Kokkos::Rank<4>> policy({0,0,0,0},{num_m,num_d,num_g,num_z});
      LTimesFunctor ltimes_functor(phidat, elldat, psidat, num_d, num_z, num_g, num_m);
      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {
      
        //RAJA::nested::forall(EXEC_POL{},
        //                     RAJA::make_tuple(IDRange(0, num_d),
        //                                      IZRange(0, num_z),
        //                                      IGRange(0, num_g),
        //                                      IMRange(0, num_m)), 
        //  [=](ID d, IZ z, IG g, IM m) {
        //  LTIMES_BODY_RAJA;
        //});
        Kokkos::parallel_for( "put.profiling.string.here", policy,
          ltimes_functor);

      }
      stopTimer(); 

      break;
    }
    case Kokkos_Lambda_Seq : {

      LTIMES_DATA_SETUP_CPU;
      Kokkos::MDRangePolicy<Kokkos::Serial,Kokkos::Rank<4>> policy({0,0,0,0},{num_m,num_d,num_g,num_z});
      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {
      
        //RAJA::nested::forall(EXEC_POL{},
        //                     RAJA::make_tuple(IDRange(0, num_d),
        //                                      IZRange(0, num_z),
        //                                      IGRange(0, num_g),
        //                                      IMRange(0, num_m)), 
        //  [=](ID d, IZ z, IG g, IM m) {
        //  LTIMES_BODY_RAJA;
        //});
        Kokkos::parallel_for( "put.profiling.string.here", policy,
          [=](Index_type m, Index_type d, Index_type z, Index_type g) {
          LTIMES_BODY;
        });

      }
      stopTimer(); 

      break;
    }
    #endif

#if defined(RAJA_ENABLE_OPENMP)      
    case Base_OpenMP : {

      LTIMES_DATA_SETUP_CPU;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        #pragma omp parallel for
        for (Index_type z = 0; z < num_z; ++z ) {
          for (Index_type g = 0; g < num_g; ++g ) {
            for (Index_type m = 0; m < num_m; ++m ) {
              for (Index_type d = 0; d < num_d; ++d ) {
                LTIMES_BODY;
              }
            }
          }
        }  

      }
      stopTimer();

      break;
    }

    #ifdef RAJAPERF_ENABLE_RAJA
    case RAJA_OpenMP : {

      LTIMES_DATA_SETUP_CPU;

      LTIMES_VIEWS_RANGES_RAJA;

      using EXEC_POL = RAJA::nested::Policy<
                     RAJA::nested::For<1, RAJA::omp_parallel_for_exec>,
                     RAJA::nested::For<2, RAJA::loop_exec>,
                     RAJA::nested::For<3, RAJA::loop_exec>,
                     RAJA::nested::For<0, RAJA::loop_exec> >;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::nested::forall(EXEC_POL{},
                             RAJA::make_tuple(IDRange(0, num_d),
                                              IZRange(0, num_z),
                                              IGRange(0, num_g),
                                              IMRange(0, num_m)),
          [=](ID d, IZ z, IG g, IM m) {
          LTIMES_BODY_RAJA;
        });

      }
      stopTimer();

      break;
    }
    #endif
    #ifdef RAJAPERF_ENABLE_KOKKOS
    case Kokkos_Functor_OpenMP : {

      LTIMES_DATA_SETUP_CPU;
      Kokkos::MDRangePolicy<Kokkos::OpenMP,Kokkos::Rank<4>> policy({0,0,0,0},{num_m,num_d,num_g,num_z});
      LTimesFunctor ltimes_functor(phidat, elldat, psidat, num_d, num_z, num_g, num_m);
      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {
      
        //RAJA::nested::forall(EXEC_POL{},
        //                     RAJA::make_tuple(IDRange(0, num_d),
        //                                      IZRange(0, num_z),
        //                                      IGRange(0, num_g),
        //                                      IMRange(0, num_m)), 
        //  [=](ID d, IZ z, IG g, IM m) {
        //  LTIMES_BODY_RAJA;
        //});
        Kokkos::parallel_for( "put.profiling.string.here", policy,
          ltimes_functor);

      }
      stopTimer(); 

      break;
    }
    case Kokkos_Lambda_OpenMP : {

      LTIMES_DATA_SETUP_CPU;
      Kokkos::MDRangePolicy<Kokkos::OpenMP,Kokkos::Rank<4>> policy({0,0,0,0},{num_m,num_d,num_g,num_z});
      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {
      
        //RAJA::nested::forall(EXEC_POL{},
        //                     RAJA::make_tuple(IDRange(0, num_d),
        //                                      IZRange(0, num_z),
        //                                      IGRange(0, num_g),
        //                                      IMRange(0, num_m)), 
        //  [=](ID d, IZ z, IG g, IM m) {
        //  LTIMES_BODY_RAJA;
        //});
        Kokkos::parallel_for( "put.profiling.string.here", policy,
          [=](Index_type m, Index_type d, Index_type z, Index_type g) {
          LTIMES_BODY;
        });

      }
      stopTimer(); 

      break;
    }
    #endif
#endif

#if defined(RAJA_ENABLE_TARGET_OPENMP)
    case Base_OpenMPTarget :
    case RAJA_OpenMPTarget :
    {
      runOpenMPTargetVariant(vid);
      break;
    }
#endif

#if defined(RAJA_ENABLE_CUDA)
    case Base_CUDA :
    case RAJA_CUDA :
    {
      runCudaVariant(vid);
      break;
    }
#endif

    default : {
      std::cout << "\n LTIMES : Unknown variant id = " << vid << std::endl;
    }

  }
}

void LTIMES::updateChecksum(VariantID vid)
{
  checksum[vid] += calcChecksum(m_phidat, m_philen);
}

void LTIMES::tearDown(VariantID vid)
{
  (void) vid;
 
  deallocData(m_phidat);
  deallocData(m_elldat);
  deallocData(m_psidat);
}

} // end namespace apps
} // end namespace rajaperf
