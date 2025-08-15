/*
 * This file is part of Vlasiator.
 * Copyright 2010-2016 Finnish Meteorological Institute
 *
 * For details of usage, see the COPYING file and read the "Rules of the Road"
 * at http://www.physics.helsinki.fi/vlasiator/
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */

#include <cmath>
#include <cstdlib>
#include <iostream>

#include "../../backgroundfield/backgroundfield.h"
#include "../../common.h"
#include "../../object_wrapper.h"
#include "../../readparameters.h"

#include "Harris.h"

using namespace spatial_cell;

namespace projects {
Harris::Harris() : TriAxisSearch() {}
Harris::~Harris() {}

bool Harris::initialize(void) { return Project::initialize(); }

void Harris::addParameters() {
   typedef Readparameters RP;
   RP::add("Harris.currentSheetType", "Type of the current sheet initialization: 1. single layer, 2. double layer", 1);
   RP::add("Harris.scale", "Harris sheet scale size (m)", 150000.0);
   RP::add("Harris.Bx0", "Reference Magnetic field (T)", 8.33061003094e-8);
   RP::add("Harris.By0", "Reference Magnetic field (T)", 8.33061003094e-8);
   RP::add("Harris.Bz0", "Reference Magnetic field (T)", 8.33061003094e-8);
   RP::add("Harris.Psi0", "Perturbation of the magnetic flux (T*m)", 0.0);

   // Per-population parameters
   for (uint i = 0; i < getObjectWrapper().particleSpecies.size(); i++) {
      const std::string& pop = getObjectWrapper().particleSpecies[i].name;

      RP::add(pop + "_Harris.n", "Reference number density (m^-3)", 1.0e6);
      RP::add(pop + "_Harris.T", "Species temperature", 1.0);
      RP::add(pop + "_Harris.nSpaceSamples", "Number of sampling points per spatial dimension.", 2);
      RP::add(pop + "_Harris.nVelocitySamples", "Number of sampling points per velocity dimension.", 2);
   }
}

void Harris::getParameters() {
   Project::getParameters();
   typedef Readparameters RP;
   RP::get("Harris.currentSheetType", currentSheetType);
   RP::get("Harris.scale", lambda);
   RP::get("Harris.Bx0", Bx0);
   RP::get("Harris.By0", By0);
   RP::get("Harris.Bz0", Bz0);
   RP::get("Harris.Psi0", Psi0);

   if (currentSheetType != 1 && currentSheetType != 2) {
      int myRank;
      MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
      if (myRank == MASTER_RANK) {
         std::cerr << "unknown initialization type: " << currentSheetType << std::endl;
         MPI_Abort(MPI_COMM_WORLD, 1);
      }
   }

   // Per-population parameters
   for (uint i = 0; i < getObjectWrapper().particleSpecies.size(); i++) {
      const std::string& pop = getObjectWrapper().particleSpecies[i].name;
      HarrisSpeciesParameters sP;

      RP::get(pop + "_Harris.n", sP.n);
      RP::get(pop + "_Harris.T", sP.T);
      RP::get(pop + "_Harris.nSpaceSamples", sP.nSpaceSamples);
      RP::get(pop + "_Harris.nVelocitySamples", sP.nVelocitySamples);

      speciesParams.push_back(sP);
   }
}

Real Harris::getMaxwellian(creal& x, creal& y, creal& z, creal& vx, creal& vy, creal& vz, creal& dvx, creal& dvy,
                           creal& dvz, const uint popID) const {
   const HarrisSpeciesParameters& sP = speciesParams[popID];
   Real mass = getObjectWrapper().particleSpecies[popID].mass;
   auto KB = physicalconstants::K_B;

   Real f, n;

   if (currentSheetType == 1) {
      Real sech2 = sqr(1.0 / std::cosh(z / lambda));
      n = sP.n * sech2 + 0.2 * sP.n;
   } else if (currentSheetType == 2) {
      const Real Lx = P::xmax - P::xmin;
      const Real Lz = P::zmax - P::zmin;

      n = sP.n * (sqr(1.0 / std::cosh((z - 0.25 * Lz) / lambda)) + sqr(1.0 / std::cosh((z + 0.25 * Lz) / lambda))) +
          0.2 * sP.n;
   }

   f = n * std::pow(mass / (2.0 * M_PI * KB * sP.T), 1.5) *
       exp(-mass * (sqr(vx) + sqr(vy) + sqr(vz)) / (2.0 * KB * sP.T));

   return f;
}

Real Harris::getBiMaxwellian(const uint popID, creal rho, creal Tpar, creal Tperp, creal vpar, creal vperp) const {
   const Real MASS = getObjectWrapper().particleSpecies[popID].mass;
   Real c1 = MASS / (2.0 * physicalconstants::K_B);
   Real c2 = c1 / M_PI;
   Real f = rho / (sqrt(Tpar) * Tperp) * sqrt(c2) * c2 * exp(-c1 * (vperp * vperp) / Tperp - c1 * vpar * vpar / Tpar);
   return f;
}

Real Harris::calcPhaseSpaceDensity(creal& x, creal& y, creal& z, creal& dx, creal& dy, creal& dz, creal& vx, creal& vy,
                                   creal& vz, creal& dvx, creal& dvy, creal& dvz, const uint popID) const {
   const HarrisSpeciesParameters& sP = speciesParams[popID];

   if ((sP.nSpaceSamples > 1) && (sP.nVelocitySamples > 1)) {
      creal d_x = dx / (sP.nSpaceSamples - 1);
      creal d_y = dy / (sP.nSpaceSamples - 1);
      creal d_z = dz / (sP.nSpaceSamples - 1);
      creal d_vx = dvx / (sP.nVelocitySamples - 1);
      creal d_vy = dvy / (sP.nVelocitySamples - 1);
      creal d_vz = dvz / (sP.nVelocitySamples - 1);

      Real avg = 0.0;

      for (uint i = 0; i < sP.nSpaceSamples; ++i)
         for (uint j = 0; j < sP.nSpaceSamples; ++j)
            for (uint k = 0; k < sP.nSpaceSamples; ++k)
               for (uint vi = 0; vi < sP.nVelocitySamples; ++vi)
                  for (uint vj = 0; vj < sP.nVelocitySamples; ++vj)
                     for (uint vk = 0; vk < sP.nVelocitySamples; ++vk) {
                        avg += getMaxwellian(x + i * d_x, y + j * d_y, z + k * d_z, vx + vi * d_vx, vy + vj * d_vy,
                                             vz + vk * d_vz, dvx, dvy, dvz, popID);
                     }
      return avg / (sP.nSpaceSamples * sP.nSpaceSamples * sP.nSpaceSamples) /
             (sP.nVelocitySamples * sP.nVelocitySamples * sP.nVelocitySamples);
   } else {
      return getMaxwellian(x + 0.5 * dx, y + 0.5 * dy, z + 0.5 * dz, vx + 0.5 * dvx, vy + 0.5 * dvy, vz + 0.5 * dvz,
                           dvx, dvy, dvz, popID);
   }
}

void Harris::calcCellParameters(spatial_cell::SpatialCell* cell, creal& t) {}

std::vector<std::array<Real, 3>> Harris::getV0(creal x, creal y, creal z, const uint popID) const {
   std::vector<std::array<Real, 3>> V0;
   std::array<Real, 3> v = {{0.0, 0.0, 0.0}};
   V0.push_back(v);
   return V0;
}

void Harris::setProjectBField(FsGrid<std::array<Real, fsgrids::bfield::N_BFIELD>, FS_STENCIL_WIDTH>& perBGrid,
                              FsGrid<std::array<Real, fsgrids::bgbfield::N_BGB>, FS_STENCIL_WIDTH>& BgBGrid,
                              FsGrid<fsgrids::technical, FS_STENCIL_WIDTH>& technicalGrid) {
   setBackgroundFieldToZero(BgBGrid);

   const Real Lx = P::xmax - P::xmin;
   const Real Lz = P::zmax - P::zmin;

   if (!P::isRestart) {
      auto localSize = perBGrid.getLocalSize().data();

      if (currentSheetType == 1) {
#pragma omp parallel for collapse(3)
         for (int i = 0; i < localSize[0]; ++i) {
            for (int j = 0; j < localSize[1]; ++j) {
               for (int k = 0; k < localSize[2]; ++k) {
                  const std::array<Real, 3> x = perBGrid.getPhysicalCoords(i, j, k);
                  std::array<Real, fsgrids::bfield::N_BFIELD>* cell = perBGrid.get(i, j, k);

                  cell->at(fsgrids::bfield::PERBX) = Bx0 * tanh((x[2] + 0.5 * perBGrid.DZ) / lambda) +
                                                     Psi0 * (-M_PI / Lz) *
                                                         cos(2 * M_PI * (x[0] + 0.5 * perBGrid.DX) / Lx) *
                                                         sin(M_PI * (x[2] + 0.5 * perBGrid.DZ) / Lz);
                  cell->at(fsgrids::bfield::PERBY) = 0.0;
                  cell->at(fsgrids::bfield::PERBZ) = Psi0 * (2 * M_PI / Lx) *
                                                     sin(2 * M_PI * (x[0] + 0.5 * perBGrid.DX) / Lx) *
                                                     cos(M_PI * (x[2] + 0.5 * perBGrid.DZ) / Lz);
               }
            }
         }
      } else if (currentSheetType == 2) {
#pragma omp parallel for collapse(3)
         for (int i = 0; i < localSize[0]; ++i) {
            for (int j = 0; j < localSize[1]; ++j) {
               for (int k = 0; k < localSize[2]; ++k) {
                  const std::array<Real, 3> x = perBGrid.getPhysicalCoords(i, j, k);
                  std::array<Real, fsgrids::bfield::N_BFIELD>* cell = perBGrid.get(i, j, k);

                  cell->at(fsgrids::bfield::PERBX) =
                      Bx0 * (tanh((x[2] + 0.5 * perBGrid.DZ - 0.25 * Lz) / lambda) -
                             tanh((x[2] + 0.5 * perBGrid.DZ + 0.25 * Lz) / lambda) + 1.0) +
                      Psi0 * (-M_PI / Lz) * cos(2 * M_PI * (x[0] + 0.5 * perBGrid.DX) / Lx) *
                          sin(2 * M_PI * (x[2] + 0.5 * perBGrid.DZ) / Lz - 0.25);
                  cell->at(fsgrids::bfield::PERBY) = 0.0;
                  cell->at(fsgrids::bfield::PERBZ) = Psi0 * (-2 * M_PI / Lx) *
                                                     sin(2 * M_PI * (x[0] + 0.5 * perBGrid.DX) / Lx) *
                                                     cos(2 * M_PI * (x[2] + 0.5 * perBGrid.DZ) / Lz);
               }
            }
         }
      }
   }
}

} // namespace projects
