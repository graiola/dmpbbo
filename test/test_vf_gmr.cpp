/**
 * @file   test_vf_gmr.cpp
 * @brief  GTest for VF GMR.
 * @author Gennaro Raiola
 *
 * This file is part of virtual-fixtures, a set of libraries and programs to create
 * and interact with a library of virtual guides.
 * Copyright (C) 2014-2016 Gennaro Raiola, ENSTA-ParisTech
 *
 * virtual-fixtures is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 2 of the License, or
 * (at your option) any later version.
 *
 * virtual-fixtures is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with virtual-fixtures.  If not, see <http://www.gnu.org/licenses/>.
 */


#include <ros/ros.h>
#include <ros/package.h>

#include <toolbox/toolbox.h>
#include <gtest/gtest.h>

#include "vf_gmr/FunctionApproximatorGMR.hpp"
#include "vf_gmr/MetaParametersGMR.hpp"
#include "vf_gmr/ModelParametersGMR.hpp"

using namespace DmpBbo;
using namespace Eigen;
using namespace tool_box;
using namespace std;

std::string pkg_path = ros::package::getPath("vf_gmr");
/*std::string file_path(pkg_path+"/test/");
std::string file_path_wrong(pkg_path+"/test/wrong");*/


TEST(VirtualMechanismGmrTest, TestGMR)
{
  // Gaussian Mixture Regression (GMR)
  //int dim_out = 3;
  int dim_in = 1;
  int number_of_gaussians = 10;
  //bool overwrite = false;
  MetaParametersGMR* meta_parameters_gmr = new MetaParametersGMR(dim_in,number_of_gaussians);
  FunctionApproximatorGMR* fa_ptr = new FunctionApproximatorGMR(meta_parameters_gmr);
  MatrixXd pos, phase, phase_predict, pos_out;

  std::string file_name_input;
  std::string file_name_output;
  std::string gmm_name_output;

  for (int i = 0; i<7; i++)
  {
      file_name_input = pkg_path+"/test/raw_pos_" + to_string(i+1);
      file_name_output = pkg_path+"/test/out_pos_" + to_string(i+1);
      gmm_name_output = pkg_path+"/test/gmm_" + to_string(i+1);

      ReadTxtFile(file_name_input,pos);

      phase.resize(pos.rows(),1);
      phase.col(0) = VectorXd::LinSpaced(pos.rows(), 0.0, 1.0);

      fa_ptr->trainIncremental(phase,pos);

      const ModelParametersGMR* model_parameters_GMR = static_cast<const ModelParametersGMR*>(fa_ptr->getModelParameters());
      model_parameters_GMR->saveGMMToMatrix(gmm_name_output, true);

      phase_predict.resize(phase.rows()*10,1);
      phase_predict.col(0) = VectorXd::LinSpaced(phase_predict.rows(), 0.0, 1.0);

      fa_ptr->predict(phase_predict,pos_out);

      WriteTxtFile(file_name_output,pos_out);
  }

  delete meta_parameters_gmr;
  delete fa_ptr;
/*
  // Load from txt

  ModelParametersGMR* model_parameters_gmr = ModelParametersGMR::loadGMMFromMatrix(gmm_name_output);
  FunctionApproximatorGMR* fa_ptr_new = new fa_t(model_parameters_gmr);
  fa_ptr_new->trainIncremental(inputs,targets);

  delete fa_ptr_new;

  //VirtualMechanismGmr<VMP_1ord_t> vm1(test_dim,K,B,Kf,Bf,fade_gain,fa_ptr);
  //VirtualMechanismGmr<VMP_2ord_t> vm2(test_dim,K,B,Kf,Bf,fade_gain,fa_ptr);*/
}

int main(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
