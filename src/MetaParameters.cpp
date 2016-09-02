/**
 * @file   MetaParameters.cpp
 * @brief  MetaParameters class source file.
 * @author Freek Stulp
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
#include <assert.h>

#include "vf_gmr/MetaParameters.hpp"

using namespace std;

namespace DmpBbo {

MetaParameters::MetaParameters(int expected_input_dim)
: expected_input_dim_(expected_input_dim)
{
  assert(expected_input_dim_>0);
}
                                                                          
MetaParameters::~MetaParameters(void) 
{
}

}
