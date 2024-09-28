from typing import Optional
import math
import torch
import torch.nn as nn
from e3nn import o3
from e3nn.util.jit import compile_mode

from scdp.common.constants import bohr2ang
from scdp.common.utils import scatter

EPSILON = 1e-8
MAX_L = 10

def well_tempered_basis_set():
    pass
            
def check_contraction(Ls, contraction):
    value_map = {}
    for a, b in zip(Ls, contraction):
        if b.item() in value_map and value_map[b.item()] != a.item():
            return False
        value_map[b.item()] = a.item()
    return True

def get_contraction_idx(Ls, contraction):
    con_idx_map = {}
    con_idx = []
    cur = 0
    for i in range(len(Ls)):
        con = contraction[i].item()
        if con not in con_idx_map:
            con_idx_map[con] = torch.arange(cur, cur + 2*Ls[i]+1)
            cur += 2*Ls[i]+1
        con_idx.append(con_idx_map[con])
    con_idx = torch.cat(con_idx)
    return con_idx
    
@compile_mode("script")
class GTOs(nn.Module):
    """
    Ensemble of gaussian typed orbitals.
    """
    def __init__(self, 
                 Ls: torch.Tensor, 
                 coeffs: torch.Tensor, 
                 expos: torch.Tensor, 
                 contraction : Optional[torch.Tensor] = None,
                 normalize: bool = True,
                 cutoff: Optional[float] = None,
                ):
        super(GTOs, self).__init__()
        assert len(Ls) == len(coeffs) == len(expos), "<len(Ls)> must equal <len(coeffs)> and <len(expos)>."
        if not isinstance(Ls, torch.Tensor):
            Ls = torch.tensor(Ls)
        if not isinstance(coeffs, torch.Tensor):
            coeffs = torch.tensor(coeffs, dtype=torch.float32)
        if not isinstance(expos, torch.Tensor):
            expos = torch.tensor(expos, dtype=torch.float32)
        
        self.register_buffer('Ls', Ls)
        self.Lmax = self.Ls.max().item()
        self.register_buffer('coeffs', coeffs)
        self.register_buffer('expos', expos)
        self.register_buffer('rad_idx', torch.repeat_interleave(torch.arange(len(Ls)), 2 * Ls + 1))
        self.register_buffer('sph_idx', torch.cat([torch.arange(i**2, i**2 + 2*i+1) for i in Ls]))
        
        # assumes Ls are sorted
        self.n_orbitals_per_L = scatter(torch.ones_like(self.Ls), self.Ls, MAX_L+1)
        self.outdim_per_L = self.n_orbitals_per_L * (2 * torch.arange(MAX_L+1) + 1)
        self.irreps = o3.Irreps([(int(x), (l, 1)) for l, x in enumerate(self.n_orbitals_per_L) if x > 0])
        
        if contraction is not None:
            if not isinstance(contraction, torch.Tensor):
                contraction = torch.FloatTensor(contraction)
            assert len(self.Ls) == len(contraction), "<len(Ls)> must equal <len(contraction)>."
            assert check_contraction(Ls, contraction), "cannot contract orbitals with different Ls."
            self.contraction = contraction
            self.register_buffer('con_idx', get_contraction_idx(Ls, contraction))
            self.outdim = self.con_idx.max() + 1
        else:
            self.contraction = None
            self.outdim = torch.sum(2 * Ls + 1)
            
        self.normalize = normalize
        if normalize:
            self.lognorm: torch.FloatTensor
            self.register_buffer('lognorm', self._generate_lognorm().view(-1))

        self.cutoff = cutoff
        
    def _generate_lognorm(self, expos=None):
        if expos is None:
            expos = self.expos
        power = (self.Ls + 1.5)
        numerator = power * torch.log(2 * expos) + math.log(2)
        denominator = torch.special.gammaln(power)
        lognorm = (numerator - denominator) / 2
        return lognorm
        
    def compute(self, vecs, expo_scaling=None, index_atom=None):
        r = vecs.norm(dim=-1, keepdim=True) + EPSILON
        spherical = o3.spherical_harmonics(
            list(range(self.Lmax+1)), vecs,
            normalize=True, normalization='norm'
        ) # N x (Lmax + 1)^2
                
        if expo_scaling is None:
            exponent = -self.expos * (r * r)
            normalization = torch.ones_like(r) * self.lognorm
        else:
            expos = self.expos.view(1, -1) * expo_scaling
            exponent = -expos[index_atom] * (r * r)            
            normalization =  self._generate_lognorm(expos)[index_atom]
        poly = self.Ls * torch.log(r)
        log = exponent + poly
        
        if self.normalize:
            radial = torch.exp(log + normalization) # * self.coeffs
        else:
            radial = self.coeffs * torch.exp(log)
            
        uncontracted = radial[:, self.rad_idx] * spherical[:, self.sph_idx]
        if self.contraction is not None:
            contracted = torch.zeros([vecs.shape[0], self.con_idx.max() + 1], device=vecs.device, dtype=uncontracted.dtype)            
            contracted.scatter_add_(1, self.con_idx.repeat(vecs.shape[0], 1), uncontracted)
            return contracted
        else:
            return uncontracted
        
    def forward(self, probe_coords, atom_coords, n_probes, n_atoms,
                coeffs=None, expo_scaling=None, reorder=True, pbc=False, cell=None):
        """
        batched forward pass.
        probe_coords:  (n_total_probes, 3)
        atom_coords:   (n_total_atoms, 3)
        n_probes:      (bsz, )
        n_atoms:       (bsz, )
        coeffs:        (n_total_atoms, outdim)
        expo_scaling:  (n_total_atoms, n_orbitals)
        """
        if expo_scaling is not None:
            assert expo_scaling.shape[1] == len(self.Ls), "expo_scaling must have the same number of orbitals as the basis."
            # expo_scaling = expo_scaling[:, :len(self.Ls)]        
        if coeffs is not None:
            assert coeffs.shape[1] == self.outdim, "coeffs must have the same number of orbitals as the basis."
            # coeffs = coeffs[:, :self.outdim]
        
        device = probe_coords.device
        n_pairs = n_probes * n_atoms
        n_total_pairs = n_pairs.sum()

        index_offset_p = torch.cumsum(n_probes, dim=0) - n_probes
        index_offset_p = torch.repeat_interleave(index_offset_p, n_pairs)
        index_offset_a = torch.cumsum(n_atoms, dim=0) - n_atoms
        index_offset_a = torch.repeat_interleave(index_offset_a, n_pairs)
        index_offset_pair = torch.cumsum(n_pairs, dim=0) - n_pairs
        index_offset_pair = torch.repeat_interleave(index_offset_pair, n_pairs)
        pair_count = torch.arange(n_total_pairs, device=device) - index_offset_pair
        n_atom_expand = torch.repeat_interleave(n_atoms, n_pairs)

        index_probe = torch.div(pair_count, n_atom_expand, rounding_mode='trunc').long() + index_offset_p
        index_atom = (pair_count % n_atom_expand).long() + index_offset_a
        if pbc:
            crossproducts = torch.cross(cell[:, [1,2,0]], cell[:, [2,0,1]], dim=-1)
            cell_vol = torch.sum(cell[:, 0] * crossproducts[:, 0], dim=-1, keepdim=True)
            n_rep = torch.ceil(
                self.cutoff * torch.norm(crossproducts / cell_vol[:, None], p=2, dim=-1)
                ).max(dim=0)[0].long()           
            _rep = lambda dim: torch.arange(-n_rep[dim], n_rep[dim] + 1)
            unit_cell = torch.tensor(
                [(x, y, z) for x in _rep(0) for y in _rep(1) for z in _rep(2)]
            ).to(cell.device).float()
            num_cells = len(unit_cell)
            unit_cell_batch = unit_cell.transpose(0, 1).unsqueeze(0).expand(len(cell), -1, -1)
            data_cell = torch.transpose(cell, 1, 2)
            pbc_offsets = torch.bmm(data_cell, unit_cell_batch)    
            pbc_offsets_per_pair = torch.repeat_interleave(
                pbc_offsets, n_pairs, dim=0
            )
            edge_atom_coords = atom_coords[index_atom].view(-1, 3, 1).expand(-1, -1, num_cells)
            edge_probe_coords = probe_coords[index_probe].view(-1, 3, 1).expand(-1, -1, num_cells)
            # num_pair -> num_pair x num_cells -> num_pair * num_cells
            index_atom = index_atom.view(-1, 1).repeat(1, num_cells).flatten()
            index_probe = index_probe.view(-1, 1).repeat(1, num_cells).flatten()
            # n_pair x 3 x num_cells
            vecs = edge_probe_coords + pbc_offsets_per_pair - edge_atom_coords
            # n_pair x num_cells x 3 -> n_pair * num_cells x 3
            vecs = vecs.transpose(1, 2).flatten(0, 1)
        else:
            vecs = probe_coords[index_probe] - atom_coords[index_atom]
        
        if self.cutoff is not None:
            mask = vecs.norm(dim=-1) < self.cutoff
            index_probe = index_probe[mask]
            index_atom = index_atom[mask]
            vecs = vecs[mask]
                
        if reorder:
            vecs = vecs[..., [1,2,0]] # z, x, y (e3nn) -> x, y, z
        vecs = vecs / bohr2ang # basis set length unit is a.u.

        if coeffs is not None:
            # n_total_probes x 1
            return scatter(
                (self.compute(vecs, expo_scaling, index_atom) * coeffs[index_atom]).sum(dim=1), 
                index_probe, n_probes.sum()
            )
        else:
            # n_total_pairs x outdim
            return scatter(self.compute(vecs, expo_scaling, index_atom), index_probe, n_probes.sum())
    
    def __repr__(self):
        return (
            f'{self.__class__.__name__}(Lmax={self.Lmax}, n_orbitals={len(self.Ls)}, '
            f'n_contracted={len(self.Ls) if self.contraction is None else self.contraction.max() + 1}, '
            f'outdim={self.outdim})'
        )
