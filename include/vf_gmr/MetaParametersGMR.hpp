/**
 * @file   MetaParametersGMR.hpp
 * @brief  MetaParametersGMR class header file.
 * @author Freek Stulp, Thibaut Munzer
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

#ifndef METAPARAMETERSGMR_H
#define METAPARAMETERSGMR_H

#include "vf_gmr/MetaParameters.hpp"

namespace DmpBbo {

/** \brief Meta-parameters for the GMR function approximator
 * \ingroup FunctionApproximators
 * \ingroup GMR
 */
class MetaParametersGMR : public MetaParameters
{
  friend class FunctionApproximatorGMR;
  
public:

  /** Constructor for the algorithmic meta-parameters of the GMR function approximator
   *  \param[in] expected_input_dim Expected dimensionality of the input data
   *  \param[in] number_of_gaussians Number of gaussians
   */
	MetaParametersGMR(int expected_input_dim, int number_of_gaussians);
	
	MetaParametersGMR* clone(void) const;

    //std::string toStr(void) const;
  
private:
  /** Number of gaussians */
  int number_of_gaussians_;

};

}

#endif        //  #ifndef METAPARAMETERSGMR_H

