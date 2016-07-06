/**
 * \file demoTrainIncrementalGMR.cpp
 * \author Gennaro Raiola
 * \brief  Demonstrates how to initialize and train a function approximator..
 *
 * \ingroup Demos
 * \ingroup FunctionApproximators
 *
 * This file is part of DmpBbo, a set of libraries and programs for the 
 * black-box optimization of dynamical movement primitives.
 * Copyright (C) 2014 Freek Stulp, ENSTA-ParisTech
 * 
 * DmpBbo is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 2 of the License, or
 * (at your option) any later version.
 * 
 * DmpBbo is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 * 
 * You should have received a copy of the GNU Lesser General Public License
 * along with DmpBbo.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <time.h>
#include <boost/filesystem.hpp>

#include "functionapproximators/FunctionApproximatorGMR.hpp"
#include "functionapproximators/MetaParametersGMR.hpp"


#include "targetFunction.hpp"

using namespace std;
using namespace Eigen;
using namespace DmpBbo;

/** Compute mean absolute error for each column of two matrices.
 *  \param[in] mat1 First matrix of data 
 *  \param[in] mat2 Second matrix of data
 *  \return Mean absolute error between the matrices (one value for each column)
 */
VectorXd meanAbsoluteErrorPerOutputDimension(const MatrixXd& mat1, const MatrixXd& mat2)
{
  MatrixXd abs_error = (mat1.array()-mat2.array()).abs();
  VectorXd mean_abs_error_per_output_dim = abs_error.colwise().mean();
     
  cout << fixed << setprecision(5);
  cout << "         Mean absolute error ";
  if (mean_abs_error_per_output_dim.size()>1) cout << " (per dimension)";
  cout << ": " << mean_abs_error_per_output_dim.transpose();      
  cout << "   \t(range of target data is " << mat1.colwise().maxCoeff().array()-mat1.colwise().minCoeff().array() << ")";
  
  return mean_abs_error_per_output_dim;
}

/** Main function
 * \param[in] n_args Number of arguments
 * \param[in] args Arguments themselves
 * \return Success of exection. 0 if successful.
 */
int main(int n_args, char** args)
{
  // First argument may be optional directory to write data to
  string directory, directory_fa;
  if (n_args>1)
    directory = string(args[1]);

  // Generate training data 
  int n_input_dims = 1;
  VectorXi n_samples_per_dim = VectorXi::Constant(1,25);
  //if (n_input_dims==2)
  //  n_samples_per_dim = VectorXi::Constant(2,25);
    
  int n_targets = 5;
  MatrixXd inputs, outputs;
  vector<MatrixXd> targets(n_targets);
  targetFunction(n_samples_per_dim,inputs,targets[0]);

  // Add some noise
  for (int i_target = 1; i_target < n_targets; i_target++)
    targets[i_target] = targets[0] + 0.001*VectorXd::Random(targets[0].rows(),targets[0].cols());

  MatrixXd inputs_all, targets_all;

  // Hacky concatenate
  inputs_all.resize(inputs.rows()*n_targets,inputs.cols());
  inputs_all << inputs, inputs, inputs, inputs, inputs;
  targets_all.resize(inputs.rows()*n_targets,inputs.cols());
  targets_all << targets[0], targets[1], targets[2], targets[3], targets[4];


  FunctionApproximatorGMR* fa;
  
  // Gaussian Mixture Regression (GMR)
  int number_of_gaussians = pow(5,n_input_dims);
  MetaParametersGMR* meta_parameters_gmr = new MetaParametersGMR(n_input_dims,number_of_gaussians);
  fa = new FunctionApproximatorGMR(meta_parameters_gmr);
    
  cout << "_____________________________________" << endl << fa->getName() << endl;
  cout << "    Training GMM with all the data"  << endl;
  if (!directory.empty()) directory_fa =  directory+"/"+fa->getName();
  fa->train(inputs_all,targets_all);
  cout << "    Predicting" << endl;
  fa->predict(inputs_all,outputs);
  meanAbsoluteErrorPerOutputDimension(targets_all,outputs);
  cout << endl << endl;

  delete fa;

  fa = new FunctionApproximatorGMR(meta_parameters_gmr);
  for (int i_target = 4; i_target> 0; i_target--)
  {
      cout << "_____________________________________" << endl << fa->getName() << endl;
      cout << "    Training number: "<< i_target << "  "<< endl;
      //if (!directory.empty()) directory_fa =  directory+"/"+fa->getName();
      fa->train(inputs,targets[i_target]);
      cout << "    Predicting" << endl;
      fa->predict(inputs,outputs);
      meanAbsoluteErrorPerOutputDimension(targets[i_target],outputs);
      cout << endl << endl;
  }
  
  delete fa;
  delete meta_parameters_gmr;

  return 0;
}


