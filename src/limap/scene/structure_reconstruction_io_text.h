// Copyright (c), ETH Zurich and UNC Chapel Hill.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
//       its contributors may be used to endorse or promote products derived
//       from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#pragma once

#include "limap/scene/structure_reconstruction.h"

#include <filesystem>
#include <iostream>

#include <Eigen/Core>

namespace limap {

// Text IO for StructureReconstruction.
//
// Write functions only serialize active observations and associations:
// - Line3d tracks: inactive observations (from Line3dWithActiveLabels) are
//   skipped
// - Group3d associations: inactive point/line associations (from
//   Group3dWithActiveLabels) are skipped
// On read, all observations/associations default to active.

void ReadStructures2dText(StructureReconstruction &reconstruction,
                          std::istream &stream);
void ReadStructures2dText(StructureReconstruction &reconstruction,
                          const std::filesystem::path &path);

void ReadLines3DText(StructureReconstruction &reconstruction,
                     std::istream &stream);
void ReadLines3DText(StructureReconstruction &reconstruction,
                     const std::filesystem::path &path);

void ReadGroups3DText(StructureReconstruction &reconstruction,
                      std::istream &stream);
void ReadGroups3DText(StructureReconstruction &reconstruction,
                      const std::filesystem::path &path);

void ReadWireframeText(StructureReconstruction &reconstruction,
                       std::istream &stream);
void ReadWireframeText(StructureReconstruction &reconstruction,
                       const std::filesystem::path &path);

void WriteStructures2dText(const StructureReconstruction &reconstruction,
                           std::ostream &stream);
void WriteStructures2dText(const StructureReconstruction &reconstruction,
                           const std::filesystem::path &path);

void WriteLines3DText(const StructureReconstruction &reconstruction,
                      std::ostream &stream);
void WriteLines3DText(const StructureReconstruction &reconstruction,
                      const std::filesystem::path &path);

void WriteGroups3DText(const StructureReconstruction &reconstruction,
                       std::ostream &stream);
void WriteGroups3DText(const StructureReconstruction &reconstruction,
                       const std::filesystem::path &path);

void WriteWireframeText(const StructureReconstruction &reconstruction,
                        std::ostream &stream);
void WriteWireframeText(const StructureReconstruction &reconstruction,
                        const std::filesystem::path &path);

} // namespace limap
