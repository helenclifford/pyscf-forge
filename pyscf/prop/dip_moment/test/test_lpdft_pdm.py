#!/usr/bin/env python
# Copyright 2014-2024 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import numpy as np
from pyscf import gto, scf, mcscf
from pyscf import mcpdft
import unittest
from pyscf.fci.addons import fix_spin_

geom_h2o='''
O  0.00000000   0.08111156   0.00000000
H  0.78620605   0.66349738   0.00000000
H -0.78620605   0.66349738   0.00000000
'''
geom_furan= '''
C        0.000000000     -0.965551055     -2.020010585
C        0.000000000     -1.993824223     -1.018526668
C        0.000000000     -1.352073201      0.181141565
O        0.000000000      0.000000000      0.000000000
C        0.000000000      0.216762264     -1.346821565
H        0.000000000     -1.094564216     -3.092622941
H        0.000000000     -3.062658055     -1.175803180
H        0.000000000     -1.688293885      1.206105691
H        0.000000000      1.250242874     -1.655874372
'''
mol_h2o = gto.M(atom = geom_h2o, basis = 'aug-cc-pVDZ', symmetry='c2v', output='/dev/null', verbose=0)
mol_furan_cation = gto.M(atom = geom_furan, basis = 'sto-3g', charge=1, spin=1, symmetry=False, output='/dev/null', verbose=0)

# Three singlets all of A1 symmetry
def get_h2o_ftpbe(mol,iroots=3): 
    weights = [1/iroots]*iroots
    mf = scf.RHF(mol).run()
    mc = mcpdft.CASSCF(mf,'ftPBE', 4, 4, grids_level=9)
    fix_spin_(mc.fcisolver, ss=0)
    mc.fcisolver.wfnsym = 'A1'
    mc = mc.multi_state(weights, "lin")
    mo = mcscf.sort_mo(mc, mf.mo_coeff, [4,5,8,9])
    mc.conv_tol = 1e-12
    mc.conv_tol_grad = 1e-6
    mc.kernel(mo)
    return mc

def get_h2o_ftlda(mol,iroots=3):
    weights = [1/iroots]*iroots
    mf = scf.RHF(mol).run()
    mc = mcpdft.CASSCF(mf,'ftLDA', 4, 4, grids_level=9)
    fix_spin_(mc.fcisolver, ss=0)
    mc.fcisolver.wfnsym = 'A1'
    mc = mc.multi_state(weights, "lin")
    mo = mcscf.sort_mo(mc, mf.mo_coeff, [4,5,8,9])
    mc.conv_tol = 1e-12
    mc.conv_tol_grad = 1e-6
    mc.kernel(mo)
    return mc

# Three doublets of A2, B2, and A2 symmetries
def get_furan_cation_ftpbe(mol,iroots=3):
    weights = [1/iroots]*iroots
    mf = scf.RHF(mol).run()
    mc = mcpdft.CASSCF(mf,'ftPBE', 5, 5, grids_level=9)
    fix_spin_(mc.fcisolver, ss=0.75)
    mc = mc.multi_state(weights, 'lin')
    mc.max_cycle_macro = 200
    mo = mcscf.sort_mo(mc, mf.mo_coeff, [12,17,18,19,20])
    mc.conv_tol = 1e-12
    mc.conv_tol_grad = 1e-6
    mc.kernel(mo)
    return mc

def get_furan_cation_ftlda(mol,iroots=3):
    weights = [1/iroots]*iroots
    mf = scf.RHF(mol).run()
    mc = mcpdft.CASSCF(mf,'ftLDA', 5, 5, grids_level=9)
    fix_spin_(mc.fcisolver, ss=0.75)
    mc = mc.multi_state(weights, 'lin')
    mc.max_cycle_macro = 200
    mo = mcscf.sort_mo(mc, mf.mo_coeff, [12,17,18,19,20])
    mc.conv_tol = 1e-12
    mc.conv_tol_grad = 1e-6
    mc.kernel(mo)
    return mc

def tearDownModule():
    global mol_h2o, mol_furan_cation
    mol_h2o.stdout.close ()
    mol_furan_cation.stdout.close ()
    del mol_h2o, mol_furan_cation

class KnownValues(unittest.TestCase):
    '''
    The reference values were obtained by numeric differentiation of the energy 
    with respect to the electric field strength extrapolated to zero step size. 
    The fields were applied along each direction in the XYZ frame, and derivatives 
    were evaluated using 2-point central difference formula.   
    '''
    def test_h2o_lpdft_ftpbe_augccpvdz(self):
        dm_ref = np.array(\
            [[0.0000, 2.0184, 0.0000],  # State 0: x, y, z
            [ 0.0000,-1.4674, 0.0000],  # State 1: x, y, z
            [ 0.0000, 3.3430, 0.0000]]) # State 2: x, y, z
        delta = 0.001
        message = "Dipoles are not equal within {} D".format(delta)
        iroots=3
        mc = get_h2o_ftpbe(mol_h2o, iroots)
        for i in range(3):
            with self.subTest (i=i):
                dm_test = mc.dip_moment(unit='Debye', origin="Coord_center",state=i)
                for dmt,dmr in zip(dm_test,dm_ref[i]):
                    self.assertAlmostEqual(dmt, dmr, None, message, delta)
        mc.stdout.close()
        
    def test_h2o_lpdft_ftlda_augccpvdz(self):
        dm_ref = np.array(\
            [[0.0000, 1.8875, 0.0000],  # State 0: x, y, z
            [ 0.0000,-1.4480, 0.0000],  # State 1: x, y, z
            [ 0.0000, 3.3715, 0.0000]]) # State 2: x, y, z
        delta = 0.001
        message = "Dipoles are not equal within {} D".format(delta)
        iroots=3
        mc = get_h2o_ftlda(mol_h2o, iroots)
        for i in range(3):
            with self.subTest (i=i):
                dm_test = mc.dip_moment(unit='Debye', origin="Coord_center",state=i)
                for dmt,dmr in zip(dm_test,dm_ref[i]):
                    self.assertAlmostEqual(dmt, dmr, None, message, delta)
        mc.stdout.close()


    def test_furan_cation_cms3_ftpbe_sto3g(self):
        # Numerical ref from this software using 2-point central difference formula
        dm_ref = np.array(\
            [[0.0000, -0.8975, -0.9215],
            [ 0.0000, -1.2513, -1.2847],
            [ 0.0000, -0.6771, -0.6952]])
        delta = 0.001
        message = "Dipoles are not equal within {} D".format(delta)
        iroots=3
        mc = get_furan_cation_ftpbe(mol_furan_cation, iroots)
        for i in range(iroots):
            with self.subTest (i=i):
                dm_test = mc.dip_moment(unit='Debye', origin="charge_center",state=i)
                for dmt,dmr in zip(dm_test,dm_ref[i]):
                    self.assertAlmostEqual(dmt, dmr, None, message, delta)
   
    def test_furan_cation_cms3_ftlda_sto3g(self):
         # Numerical ref from this software using 2-point central difference formula
         dm_ref = np.array(\
             [[0.0000, -0.8332, -0.8555],
             [ 0.0000, -1.2069, -1.2392],
             [ 0.0000, -0.6569, -0.6745]])
         delta = 0.001
         message = "Dipoles are not equal within {} D".format(delta)
         iroots=3
         mc = get_furan_cation_ftlda(mol_furan_cation, iroots)
         for i in range(iroots):
             with self.subTest (i=i):
                 dm_test = mc.dip_moment(unit='Debye', origin="charge_center",state=i)
                 for dmt,dmr in zip(dm_test,dm_ref[i]):
                     self.assertAlmostEqual(dmt, dmr, None, message, delta)


if __name__ == "__main__":
    print("Test for L-PDFT permanent dipole moments")
    unittest.main()