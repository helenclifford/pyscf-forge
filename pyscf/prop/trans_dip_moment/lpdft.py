from pyscf.lib import logger
import numpy as np
from pyscf.data import nist
from pyscf import lib
from functools import reduce
from pyscf.prop.dip_moment import lpdft
from pyscf.grad import lpdft as lpdft_grad
from pyscf.prop.dip_moment.mcpdft import get_guage_origin
from pyscf.fci import direct_spin1
from pyscf.grad.mspdft import _unpack_state


def lpdft_trans_HellmanFeynman_dipole(mc, mo_coeff=None, state=None, ci=None, ci_bra=None, ci_ket=None, origin='Coord_Center'):
    if mo_coeff is None: mo_coeff = mc.mo_coeff
    if state is None: state   = self.state
    if ci is None: ci = mc.ci
    ket, bra = _unpack_state (state)
    if ci_bra is None: ci_bra = ci[:,bra]
    if ci_ket is None: ci_ket = ci[:,ket]
    if mc.frozen is not None:
        raise NotImplementedError
    
    mol = mc.mol
    ncore = mc.ncore
    ncas = mc.ncas
    nocc = ncas
    nocc = ncore + ncas
    nelecas = mc.nelecas

    #mo_core = mo_coeff[:,:nocc]
    mo_cas = mo_coeff[:,ncore:nocc]

    casdm1 = direct_spin1.trans_rdm12(ci_bra, ci_ket, ncas, nelecas)[0]
    casdm1 = 0.5 * (np.array(casdm1) + np.array(casdm1).T)
    
    #dm_core = np.dot(mo_core, mo_core.T) * 2
    dm_cas = reduce(np.dot, (mo_cas, casdm1, mo_cas.T))
    
    #tdm = dm_cas + dm_core
    tdm = dm_cas 

    center = get_guage_origin(mol,origin)
    with mol.with_common_orig(center):
        ao_dip = mol.intor_symmetric('int1e_r', comp=3)
    elec_term = -np.tensordot(ao_dip, tdm).real
    
    return elec_term

class TransitionDipole (lpdft.ElectricDipole):

    def convert_dipole (self, ham_response, LdotJnuc, mol_dip, unit='Debye'):
        val = np.linalg.norm(mol_dip)
        i   = self.state[0]
        j   = self.state[1]
        dif = abs(self.e_states[i]-self.e_states[j])
        osc = 2/3*dif*val**2
        if unit.upper() == 'DEBYE':
            for x in [ham_response, LdotJnuc, mol_dip]: x *= nist.AU2DEBYE
        log = lib.logger.new_logger(self, self.verbose)
        log.note('L-PDFT TDM <{}|mu|{}>          {:>10} {:>10} {:>10}'.format(i,j,'X','Y','Z'))
        log.note('Hamiltonian Contribution (%s) : %9.5f, %9.5f, %9.5f', unit, *ham_response)
        log.note('Lagrange Contribution    (%s) : %9.5f, %9.5f, %9.5f', unit, *LdotJnuc)
        log.note('Transition Dipole Moment (%s) : %9.5f, %9.5f, %9.5f', unit, *mol_dip)
        log.note('Oscillator strength  : %9.5f', osc)
        return mol_dip

    def get_ham_response(self, state=None, verbose=None, mo=None,
            ci=None, origin='Coord_Center', **kwargs):
        if state is None: state   = self.state
        if verbose is None: verbose = self.verbose
        if mo is None: mo      = self.base.mo_coeff
        if ci is None: ci      = self.base.ci

        fcasscf = self.make_fcasscf (state)
        fcasscf.mo_coeff = mo
        fcasscf.ci = ci

        elec_term = lpdft_trans_HellmanFeynman_dipole (fcasscf, mo_coeff=mo, state=state, ci_bra = ci[state[0]], ci_ket = ci[state[1]], origin=origin)
        return elec_term

