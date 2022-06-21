############################################
# imports
############################################
import jax
import hashlib

import numpy as np
# import matplotlib.pyplot as plt


###########################################
# control gpu/cpu memory usage
###########################################
def rm(x):
    """remove data from device"""
    jax.tree_util.tree_map(lambda y: y.device_buffer.delete(), x)


def to(x, device="cpu"):
    """move data to device"""
    d = jax.devices(device)[0]
    return jax.tree_util.tree_map(lambda y: jax.device_put(y, d), x)


def clear_mem(device="gpu"):
    """remove all data from device"""
    backend = jax.lib.xla_bridge.get_backend(device)
    for buf in backend.live_buffers():
        buf.delete()


#########################################################################
# utils
#########################################################################
def get_hash(x):
    return hashlib.sha1(x.encode()).hexdigest()


def homooligomerize(msas, deletion_matrices, homooligomer=1):
    if homooligomer == 1:
        return msas, deletion_matrices
    else:
        new_msas = []
        new_mtxs = []
        for o in range(homooligomer):
            for msa, mtx in zip(msas, deletion_matrices):
                num_res = len(msa[0])
                L = num_res * o
                R = num_res * (homooligomer - (o + 1))
                new_msas.append(["-" * L + s + "-" * R for s in msa])
                new_mtxs.append([[0] * L + m + [0] * R for m in mtx])
        return new_msas, new_mtxs


# keeping typo for cross-compatibility
def homooliomerize(msas, deletion_matrices, homooligomer=1):
    return homooligomerize(msas, deletion_matrices, homooligomer=homooligomer)


def homooligomerize_heterooligomer(msas, deletion_matrices, lengths, homooligomers):
    """
    ----- inputs -----
    msas: list of msas
    deletion_matrices: list of deletion matrices
    lengths: list of lengths for each component in complex
    homooligomers: list of number of homooligomeric copies for each component
    ----- outputs -----
    (msas, deletion_matrices)
    """
    if max(homooligomers) == 1:
        return msas, deletion_matrices

    elif len(homooligomers) == 1:
        return homooligomerize(msas, deletion_matrices, homooligomers[0])

    else:
        frag_ij = [[0, lengths[0]]]
        for length in lengths[1:]:
            j = frag_ij[-1][-1]
            frag_ij.append([j, j + length])

        # for every msa
        mod_msas, mod_mtxs = [], []
        for msa, mtx in zip(msas, deletion_matrices):
            mod_msa, mod_mtx = [], []
            # for every sequence
            for n, (s, m) in enumerate(zip(msa, mtx)):
                # split sequence
                _s, _m, _ok = [], [], []
                for i, j in frag_ij:
                    _s.append(s[i:j])
                    _m.append(m[i:j])
                    _ok.append(max([o != "-" for o in _s[-1]]))

                if n == 0:
                    # if first query sequence
                    mod_msa.append("".join([x * h for x, h in zip(_s, homooligomers)]))
                    mod_mtx.append(sum([x * h for x, h in zip(_m, homooligomers)], []))

                elif sum(_ok) == 1:
                    # elif one fragment: copy each fragment to every homooligomeric copy
                    a = _ok.index(True)
                    for h_a in range(homooligomers[a]):
                        _blank_seq = [
                            ["-" * l] * h for l, h in zip(lengths, homooligomers)
                        ]
                        _blank_mtx = [
                            [[0] * l] * h for l, h in zip(lengths, homooligomers)
                        ]
                        _blank_seq[a][h_a] = _s[a]
                        _blank_mtx[a][h_a] = _m[a]
                        mod_msa.append("".join(["".join(x) for x in _blank_seq]))
                        mod_mtx.append(sum([sum(x, []) for x in _blank_mtx], []))
                else:
                    # else: copy fragment pair to every homooligomeric copy pair
                    for a in range(len(lengths) - 1):
                        if _ok[a]:
                            for b in range(a + 1, len(lengths)):
                                if _ok[b]:
                                    for h_a in range(homooligomers[a]):
                                        for h_b in range(homooligomers[b]):
                                            _blank_seq = [
                                                ["-" * l] * h
                                                for l, h in zip(lengths, homooligomers)
                                            ]
                                            _blank_mtx = [
                                                [[0] * l] * h
                                                for l, h in zip(lengths, homooligomers)
                                            ]
                                            for c, h_c in zip([a, b], [h_a, h_b]):
                                                _blank_seq[c][h_c] = _s[c]
                                                _blank_mtx[c][h_c] = _m[c]
                                            mod_msa.append(
                                                "".join(
                                                    ["".join(x) for x in _blank_seq]
                                                )
                                            )
                                            mod_mtx.append(
                                                sum(
                                                    [sum(x, []) for x in _blank_mtx], []
                                                )
                                            )
            mod_msas.append(mod_msa)
            mod_mtxs.append(mod_mtx)
        return mod_msas, mod_mtxs


def chain_break(idx_res, Ls, length=200):
    # Minkyung's code
    # add big enough number to residue index to indicate chain breaks
    L_prev = 0
    for L_i in Ls[:-1]:
        idx_res[L_prev + L_i :] += length
        L_prev += L_i
    return idx_res





##########################################################################
##########################################################################


def kabsch(a, b, weights=None, return_v=False):
    a = np.asarray(a)
    b = np.asarray(b)
    if weights is None:
        weights = np.ones(len(b))
    else:
        weights = np.asarray(weights)
    B = np.einsum("ji,jk->ik", weights[:, None] * a, b)
    u, s, vh = np.linalg.svd(B)
    if np.linalg.det(u @ vh) < 0:
        u[:, -1] = -u[:, -1]
    if return_v:
        return u
    else:
        return u @ vh

