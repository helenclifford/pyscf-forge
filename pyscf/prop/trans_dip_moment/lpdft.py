from pyscf.lib import logger
import numpy as np
from pyscf.data import nist
from pyscf import lib
from functools import reduce
from pyscf.prop.dip_moment import lpdft
from pyscf.grad import lpdft as lpdft_grad
from pyscf.prop.dip_moment.mcpdft import get_guage_origin, nuclear_dipole
from pyscf.fci import direct_spin1
from pyscf.grad.mspdft import _unpack_state
from pyscf.nac.sacasscf import gen_g_hop_active
from pyscf.mcscf import newton_casscf
from pyscf.grad import sacasscf as sacasscf_grad

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
    nocc = ncore + ncas
    nelecas = mc.nelecas

    mo_core = mo_coeff[:,:nocc]
    mo_cas = mo_coeff[:,ncore:nocc]
    
    casdm1 = direct_spin1.trans_rdm12(ci[state[0]], ci[state[1]], ncas, nelecas)[0]
    casdm1 = 0.5 * (np.array(casdm1) + np.array(casdm1).T)
    
    dm_core = np.dot(mo_core, mo_core.T) * 2
    dm_cas = reduce(np.dot, (mo_cas, casdm1, mo_cas.T))
    
    tdm = dm_cas  #+ dm_core

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
        ket, bra = _unpack_state (state)

        #fcasscf = self.make_fcasscf_lpdft_trans(ket)
        fcasscf = self.make_fcasscf(state)
        fcasscf.mo_coeff = mo
        fcasscf.ci = ci

        elec_term = lpdft_trans_HellmanFeynman_dipole (fcasscf, mo_coeff=mo, state=state, ci=ci, ci_bra = ci[state[0]], ci_ket = ci[state[1]], origin=origin)
        return elec_term       

    def make_fcasscf_lpdft_trans (self, state=None, casscf_attr=None,fcisolver_attr=None):
        if state is None: state = self.state
        if casscf_attr is None: casscf_attr = {}
        if fcisolver_attr is None: fcisolver_attr = {}
        ket, bra = _unpack_state (state)
        ci, ncas, nelecas = self.base.ci, self.base.ncas, self.base.nelecas

        casdm1, casdm2 = direct_spin1.trans_rdm12 (ci[bra], ci[ket], ncas, nelecas)
        casdm1 = 0.5 * (casdm1 + casdm1.T)
        casdm2 = 0.5 * (casdm2 + casdm2.transpose (1,0,3,2))
        fcisolver_attr['make_rdm12'] = lambda *args, **kwargs : (casdm1, casdm2)
        fcisolver_attr['make_rdm1'] = lambda *args, **kwargs : casdm1
        fcisolver_attr['make_rdm2'] = lambda *args, **kwargs : casdm2
        return sacasscf_grad.Gradients.make_fcasscf (self,
            state=ket, casscf_attr=casscf_attr, fcisolver_attr=fcisolver_attr)   
   
    def get_wfn_response(self, state=None, verbose=None, mo=None, ci=None, feff1=None, feff2=None, **kwargs):
        if state is None: state = self.state
        if verbose is None: verbose = self.verbose
        if mo is None: mo = self.base.mo_coeff
        if ci is None: ci = self.base.ci
        if (feff1 is None) or (feff2 is None):
            feff1, feff2 = self.get_otp_gradient_response(mo, ci, state)
 
        log = logger.new_logger (self, verbose)
        ket, bra = _unpack_state (state)
       
        ndet = self.na_states[state[0]] * self.nb_states[state[0]]
        fcasscf = self.make_fcasscf_lpdft_trans(state)
        fcasscf_sa = self.make_fcasscf_sa()
        fcasscf.mo_coeff = mo
        #fcasscf.ci = ci[ket]
        fcasscf.ci = ci
        fcasscf.get_hcore = self.base.get_lpdft_hcore
        fcasscf_sa.get_hcore = lambda: feff1
       
        g_all_explicit = np.zeros(self.ngorb+self.nci)

        ncore = self.base.ncore
        moH = mo.conj ().T   
        vnocore = self.base.veff2.vhf_c.copy()
        vnocore[:,:ncore] = -moH @ fcasscf.get_hcore() @ mo[:,:ncore]
        with lib.temporary_env(self.base.veff2, vhf_c=vnocore):
            g_all_explicit = newton_casscf.gen_g_hop (fcasscf, mo, ci[state[1]], self.base.veff2, verbose)[0]       #ci[state[1]] or ci[state[0]]?
   
        g_all_implicit = newton_casscf.gen_g_hop (fcasscf_sa, mo, ci, feff2, verbose)[0]

        spin_states = np.asarray(self.spin_states)
        gmo_implicit, gci_implicit = self.unpack_uniq_var(g_all_implicit)
        for root in range(self.nroots):
            idx_spin = spin_states == spin_states[root]
            idx = np.where(idx_spin)[0]

            gci_root = gci_implicit[root].ravel()

            assert root in idx
            ci_proj = np.asarray([ci[i].ravel() for i in idx])
            gci_sa = np.dot(ci_proj, gci_root)
            gci_root -= np.dot(gci_sa, ci_proj)

            gci_implicit[root] = gci_root

        g_all = self.pack_uniq_var(gmo_implicit, gci_implicit)

        g_all[: self.ngorb] += g_all_explicit[: self.ngorb]

       # g_all = np.zeros(self.ngorb+self.nci)
       # g_all[: self.ngorb] += g_all_explicit[: self.ngorb]
       # g_all[: self.ngorb] += g_all_implicit[: self.ngorb]
       # g_all[self.ngorb :] += g_all_implicit[self.ngorb :]

        # Debug
        log.debug("g_all explicit orb:\n{}".format(g_all_explicit[: self.ngorb]))
        log.debug("g_all explicit ci:\n{}".format(g_all_explicit[self.ngorb :]))
        log.debug("g_all implicit orb:\n{}".format(g_all_implicit[: self.ngorb]))
        log.debug("g_all implicit ci:\n{}".format(g_all_implicit[self.ngorb :]))

        gorb, gci = self.unpack_uniq_var(g_all)
        log.debug("g_all orb:\n{}".format(gorb))
        log.debug("g_all ci:\n{}".format([c.ravel() for c in gci]))
        
        return g_all 


