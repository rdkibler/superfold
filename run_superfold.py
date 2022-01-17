#!/home/rdkibler/.conda/envs/pyroml/bin/python3.8

# most of the code is copied from krypton's colabfold https://colab.research.google.com/drive/1teIx4yDyrSm0meWvM9X9yYNLHGO7f_Zy#scrollTo=vJxiCSLc7IWD
# The initial guess stuff is from Nate Bennett with maybe some helper code from Adam Broerman
# pae code is lifted from Nate
# it contains alphafold2-multimer but don't use it
# krypton is basically lead author without knowing it

import time

time_checkpoint = time.time()
import argparse
import os

# from Bio import SeqIO
parser = argparse.ArgumentParser()


# This hack is probably unnecessary with AF2-multimer since they've switched to jax for feature processing
# tell Jax and Tensorflow to use the same memory. This allows us to run larger structures
os.environ["TF_FORCE_UNIFIED_MEMORY"] = "1"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "2.0"


# let's use a linkfile-like strategy for telling the script where to find stuff like data
SCRIPTDIR = os.path.dirname(os.path.realpath(__file__))
with open(f"{SCRIPTDIR}/alphafold_weights.pth", "r") as f:
    ALPHAFOLD_DATADIR = f.read().strip()
assert os.path.exists(ALPHAFOLD_DATADIR)


def validate_file(parser, path):
    """
    Check for file existance and read files first so that we can fail early before loading alphafold, etc

    https://stackoverflow.com/a/11541450
    """
    if not os.path.exists(path):
        parser.error("The file %s does not exist!" % path)
    else:
        if (
            path.endswith(".pdb")
            or path.endswith(".pdb.gz")
            or path.lower().endswith(".fasta")
            or path.lower().endswith(".fa")
            or path.lower().endswith(".silent")
        ):
            return path
        else:
            parser.error(
                "Only PDB files, silent files, and FASTA files are allowed. You supplied: %s"
                % path
            )


parser.add_argument(
    "input_files",
    metavar="PATH",
    nargs="+",
    type=lambda x: validate_file(parser, x),
    help="Paths to PDB files or FASTA files to run AlphaFold2 predictions on.",
)

parser.add_argument(
    "--pad_lengths",
    action="store_true",
    help="compile the model once to the longest input PDB and pad the remaining sequences. Ryan is unsure how this affects prediction accuracy, but it will speed up multiple prediction.",
)
parser.add_argument(
    "--mock_msa_depth",
    default=512,
    help="fake the msa. Default = 512. to go fast, use 1",
    type=int,
)
parser.add_argument(
    "--models",
    choices=["1", "2", "3", "4", "5", "all"],
    default="4",
    nargs="+",
    help="Deepmind provided five sets of weights/models. You can choose any combination of models to run. The model number 5 has been found (by aivan) to perform the best on single sequences so this is the default, but using multiple models might provide you with a relevent ensemble of structures.",
)
parser.add_argument(
    "--type",
    choices=["monomer", "monomer_ptm", "multimer"],
    default="monomer_ptm",
    help="The flavor of alphafold weights to use. 'monomer' is the original AF2. 'ptm' is the original AF2 with an extra head that predicts pTMscore. 'multimer' is AF2-Multimer. The use of multimer weights with standard AF2 probably won't work",
)
parser.add_argument(
    "--version",
    choices=["monomer", "multimer"],
    default="monomer",
    help="The version of AF2 Module to use. Both versions can predict both mulimers. When used to predict multimers, the 'monomer' version is equivalent to AF2-Gap. The 'multimer' version is equivalent to AF2-Multimer and should not be used with the monomer weight types.",
)


parser.add_argument(
    "--nstruct",
    help="Number of independent outputs to generate PER MODEL. It will make predictions with seeds starting at 'seed_start' and increasing by one until n outputs are generated (like seed_range = range(seed_start,seed_start + nstruct)). Default=1",
    default=1,
    type=int,
)
parser.add_argument(
    "--seed_start", type=int, help="Seed to start at. Default=0", default=0
)
parser.add_argument(
    "--num_ensemble",
    type=int,
    default=1,
    help="number of times to process the input features and combine. default = 1. Deepmind used 8 for casp. Expert Option.",
)
parser.add_argument(
    "--turbo",
    action="store_true",
    help="use the latest and greatest hacks to make it run fast fast fast.",
)
parser.add_argument(
    "--max_recycles",
    type=int,
    default=3,
    help="max number of times to run evoformer. Default is 3. Single domain proteins need fewer runs. Multidomain or PPI may need more",
)
parser.add_argument(
    "--recycle_tol",
    type=float,
    default=0.0,
    help="Stop recycling early if CA-RMSD difference between current output and previous is < recycle_tol. Default = 0.0 (no early stopping)",
)
parser.add_argument("--show_images", action="store_true")
parser.add_argument(
    "--output_pae",
    action="store_true",
    help="dump the PAE matrix to disk. This is useful for investigating interresidue relationships.",
)
parser.add_argument(
    "--save_intermediates",
    action="store_true",
    help="save intermediate structures between recycles. This is useful for making folding movies/trajectories",
)
parser.add_argument(
    "--amber_relax",
    action="store_true",
    help="use AMBER to relax the structure after prediction",
)
parser.add_argument(
    "--overwrite",
    action="store_true",
    help="overwrite existing files. Default is to skip predictions which would result in files that already exist. This is useful for checkpointing and makes the script more backfill friendly.",
)
parser.add_argument(
    "--initial_guess",
    action="store_true",
    help="use the initial guess from the input PDB file. This is useful for trying to focus predictions toward a known conformation.",
)
parser.add_argument(
    "--reference_pdb",
    type=str,
    help="reference PDB to use for RMSD calculations. Coordinates (after alignment) and chain order will be updated to that of this reference, unless the input_files are PDB files",
)
# sidechain_relax_parser = parser.add_mutually_exclusive_group(required=False)
# sidechain_relax_parser.add_argument("--amber_relax",help="run Amber relax on each output prediction")
# sidechain_relax_parser.add_argument("--rosetta_relax",help="run Rosetta relax (sidechain only) on each output prediction")

parser.add_argument(
    "--enable_dropout",
    action="store_true",
    help="Introduce structural diversity by enabling dropout",
)
parser.add_argument(
    "--pct_seq_mask",
    type=float,
    default=0.15,
    help="percent of sequence to make during inference. Default = 0.15. Setting to 0 might reduce prediction stocasticity.",
)
# parser.add_argument("--deepaccnet",action="store_true",help="Run DeepAccNet on the AlphaFold2 outputs.")

parser.add_argument(
    "--out_dir",
    type=str,
    default="output/",
    help="Directory to output models and data.",
)

args = parser.parse_args()


import os
import pymol
import silent_tools
from dataclasses import dataclass
from typing import Union, Tuple, Dict
import numpy as np
from matplotlib import pyplot as plt

plt.switch_backend("agg")
import time

os.makedirs(args.out_dir, exist_ok=True)


def renumber(pdbstr):
    previous_resid = None
    new_resid = 0
    new_atomid = 1

    lines = [line for line in pdbstr.split("\n") if line[:4] == "ATOM"]
    fixed_pdbstr = ""

    for line in lines:
        resid = int(line[22:26])
        if resid != previous_resid:
            new_resid += 1
        new_line = (
            line[:6]
            + f"{new_atomid: >5}"
            + line[11:22]
            + f"{str(new_resid): >4}"
            + line[26:]
        )
        previous_resid = resid
        fixed_pdbstr += new_line + "\n"
        new_atomid += 1
    return fixed_pdbstr


def pymol_renumber(sele="all"):
    pdbstr = pymol.cmd.get_pdbstr(sele)

    fixed_pdbstr = renumber(pdbstr)

    pymol.cmd.delete(sele)
    pymol.cmd.read_pdbstr(fixed_pdbstr, sele)


def pymol_apply_new_chains(pymol_object_name: str, new_chains: list) -> None:
    """
    Applies the new chains to a pymol object in the current order of chains

    i.e. if the pymol object has chains A, B, C, D, and the new_chains list is D, B, A, C
    it will turn current chain A into chain D, current chain B into chain B,
    ccurrent chain C into chain A, and current chain D into chain C.
    """
    import pymol

    # first, make selections for each current chain so we don't lose track of them
    chain_selections = []
    for i, chain in enumerate(pymol.cmd.get_chains(pymol_object_name)):
        chain_name = f"{pymol_object_name}_chain_{i}"
        pymol.cmd.select(chain_name, f"{pymol_object_name} and chain {chain}")
        chain_selections.append(chain_name)

    # print(pymol.cmd.get_names('all', 0))

    # now, apply the new chains
    for chain_selection, new_chain in zip(chain_selections, new_chains):
        pymol.cmd.alter(chain_selection, f'chain = "{new_chain}"')

    pymol.cmd.sort(pymol_object_name)


def get_chain_range_map(pdbstr):
    # load into pymol
    pymol.cmd.read_pdbstr(pdbstr, "chain_range_map_obj")

    space = {"chain_letters": []}
    # iterate over residues
    pymol.cmd.iterate(
        "chain_range_map_obj and n. CA", "chain_letters.append(chain)", space=space
    )
    # save indices where chains start and stop

    prev_chain = "A"
    chain_start = 0
    chain_range_map = {}
    for i, residue_chain in enumerate(space["chain_letters"]):
        if residue_chain != prev_chain:
            chain_stop = i
            chain_range_map[prev_chain] = (chain_start, chain_stop)
            chain_start = i
        prev_chain = residue_chain
    chain_stop = i
    chain_range_map[prev_chain] = (chain_start, chain_stop)
    # clean up
    pymol.cmd.delete("chain_range_map_obj")
    return chain_range_map


def get_chain_permutations(chains: list) -> list:
    """
    Gets all permutations of the chains.
    """
    import itertools

    return list(itertools.permutations(chains))


def pymol_multichain_align(
    model_pymol_name: str, reference_pymol_name: str, alignment_mode: str = "align"
) -> Tuple[float, str, list]:
    """
    Aligns two multichain models using pymol.
    Returns the RMSD and the aligned model.
    """
    import pymol
    import random
    import string

    # generate a random prefix so we don't overwrite anything else in the pymol session
    prefix = "".join(
        random.choice(string.ascii_uppercase + string.digits) for _ in range(5)
    )

    temp_pymol_name = f"{prefix}_temp"
    best_pymol_name = f"{prefix}_best"

    chains = pymol.cmd.get_chains(model_pymol_name)

    align_func = getattr(pymol.cmd, alignment_mode)

    best_rmsd = float("inf")
    best_order = None
    for new_order in get_chain_permutations(chains):
        # make a temporary object with the new order of chains
        pymol.cmd.delete(temp_pymol_name)
        pymol.cmd.create(temp_pymol_name, model_pymol_name)
        pymol_apply_new_chains(temp_pymol_name, new_order)
        rmsd = align_func(
            f"{temp_pymol_name} and n. CA",
            f"{reference_pymol_name} and n. CA",
            cycles=0,
        )[0]

        # debug: useful to see if alignment is working
        # print(f'{rmsd} {new_order}')

        if rmsd < best_rmsd:
            best_order = new_order
            best_rmsd = rmsd
            pymol.cmd.create(best_pymol_name, temp_pymol_name)

    best_pdbstr = pymol.cmd.get_pdbstr(best_pymol_name)

    # clean up
    pymol.cmd.delete(temp_pymol_name)
    pymol.cmd.delete(best_pymol_name)

    return best_rmsd, best_pdbstr, best_order


def convert_pdb_chainbreak_to_new_chain(pdbstring):
    previous_resid = 0
    chain_num = 0
    new_pdbstring = ""
    import string

    alphabet = string.ascii_uppercase
    for line in pdbstring.split("\n"):
        if line[:4] == "ATOM":
            resid = int(line[22:26])
            if resid - previous_resid > 1:
                chain_num += 1
            new_pdbstring += line[:21] + f"{alphabet[chain_num]: >1}" + line[22:] + "\n"
            previous_resid = resid
        else:
            new_pdbstring += line + "\n"
    return new_pdbstring


@dataclass(frozen=True)
class PredictionTarget:
    name: str
    seq: str
    pymol_obj_name: str = None

    def __lt__(self, other):
        return len(self) < len(other)

    def __len__(self):
        return len(self.seq.replace("/", ""))

    def padseq(self, pad_amt):
        return PredictionTarget(
            self.name, self.seq + "U" * pad_amt, self.pymol_obj_name
        )


def parse_fasta(path):
    if path.endswith(".gz"):
        import gzip

        filehandle = gzip.open(path, "rt")
    else:
        filehandle = open(path, "rt")

    outputs = []

    seq = ""
    name = ""
    for line in filehandle:
        if line.startswith(">"):
            if len(seq) > 0:
                outputs.append(PredictionTarget(name, seq))
            name = line[1:].strip()
            seq = ""
        else:
            seq += line.strip()
    if len(seq) > 0:
        # This should always be true for a well formatted fasta file
        outputs.append(PredictionTarget(name, seq))

    filehandle.close()

    return outputs


unique_name_counter = 0


def get_unique_name():
    global unique_name_counter
    unique_name_counter += 1
    return f"struct{unique_name_counter}"


def parse_pdb(path):
    name = path.split("/")[-1].split(".pdb")[0]
    pymol_obj_name = get_unique_name()
    pymol.cmd.load(path, object=pymol_obj_name)
    fastastring = pymol.cmd.get_fastastr(pymol_obj_name)
    # kinda obtuse, sorry. Basically, split the string on newlines and throw out the leading ">blahblahblah" line b/c we don't need it
    # then step through and (eventuallY) concat the normal seq lines. When it encounters a new ">blahbalbha" line, this signifies a chain
    # break and put in the chainbreak char instead
    seq = "".join(
        [line if not line.startswith(">") else "/" for line in fastastring.split()[1:]]
    )

    return [PredictionTarget(name, seq, pymol_obj_name)]


def parse_silent(path):
    outputs = []
    index = silent_tools.get_silent_index(path)

    tags = index["tags"]

    structures = silent_tools.get_silent_structures(path, index, tags)

    for name, structure in zip(tags, structures):

        chain_per_res = silent_tools.get_chain_ids(structure)

        # only gonna grab C-alphas
        seq = "".join(silent_tools.get_sequence_chunks(structure))
        # atoms = silent_tools.sketch_get_cas_protein_struct(structure)
        atoms = silent_tools.sketch_get_atoms(structure, 1)
        pdbstring = silent_tools.write_pdb_atoms(
            atoms, seq, ["CA"], chain_ids=chain_per_res
        )
        pymol_obj_name = get_unique_name()
        pymol.cmd.read_pdbstr("".join(pdbstring), oname=pymol_obj_name)

        # but you ask "why? you already have 'seq'!" Well, this one has chain breaks and I already wrote the code above for the pdb parsing
        fastastring = pymol.cmd.get_fastastr(pymol_obj_name)
        seq = "".join(
            [
                line if not line.startswith(">") else "/"
                for line in fastastring.split()[1:]
            ]
        )

        outputs.append(PredictionTarget(name, seq, pymol_obj_name))

    return outputs


def parse_file(path):
    targets = []
    if path.endswith(".gz"):
        filename = path[:-3]
    else:
        filename = path

    if filename.endswith(".silent"):
        targets.extend(parse_silent(path))
    elif filename.endswith(".fa") or filename.endswith(".fasta"):
        targets.extend(parse_fasta(path))
    elif filename.endswith(".pdb"):
        targets.extend(parse_pdb(path))

    return targets


query_targets = []
for file in args.input_files:
    query_targets.extend(parse_file(file))


from alphafold.model import model
from alphafold.model import config
from alphafold.model import data
from alphafold.common import protein

from alphafold.data import parsers

# I don't know if this is a good idea.
if args.version == "multimer":
    from alphafold.data import pipeline_multimer
from alphafold.data import pipeline

import colabfold as cf
from collections import defaultdict
import tqdm
import jax
from jax.lib import xla_bridge

device = xla_bridge.get_backend().platform
print("using ", device)


if args.amber_relax:
    from alphafold.relax import relax

    RELAX_MAX_ITERATIONS = 0
    RELAX_ENERGY_TOLERANCE = 2.39
    RELAX_STIFFNESS = 10.0
    RELAX_EXCLUDE_RESIDUES = []
    RELAX_MAX_OUTER_ITERATIONS = 3

    amber_relaxer = relax.AmberRelaxation(
        max_iterations=RELAX_MAX_ITERATIONS,
        tolerance=RELAX_ENERGY_TOLERANCE,
        stiffness=RELAX_STIFFNESS,
        exclude_residues=RELAX_EXCLUDE_RESIDUES,
        max_outer_iterations=RELAX_MAX_OUTER_ITERATIONS,
    )


longest = max([len(tgt) for tgt in query_targets])

if longest < 400 and device != "cpu":
    # catch the user's eye
    plural = "s are" if len(query_targets) > 1 else " is"
    print(
        "======================================================================================="
    )
    print(
        f"WARNING: Your query{plural} shorter than 400 residues. This is a very small protein."
    )
    print(
        "You may want to use the CPU to conserve GPU resources for those who need them."
    )
    print(
        "Remember that you can launch far more jobs in parallel on CPUs than you can on GPUs..."
    )
    print("See this example of how prediction time scales on CPU vs GPU: ")
    print(
        "https://docs.google.com/spreadsheets/d/1jTGITpIx6fJehAplUkXtePOp7me3Dpq_pPKHn68F7XY"
    )
    print(
        "======================================================================================="
    )

if args.pad_lengths:
    # I don't htink this is the best implememntation wrt the use of "U" to pad
    query_targets = [tgt.padseq(longest - len(tgt)) for tgt in query_targets]


length_dict = defaultdict(lambda: [])
for tgt in query_targets:
    length_dict[len(tgt)].append(tgt)

# sort so longer first so it fails early
lengths = sorted(length_dict.keys(), reverse=True)

if args.pad_lengths:
    assert len(lengths) == 1

prev_compile_settings = tuple()


seed_range = list(range(args.seed_start, args.seed_start + args.nstruct))

# initial guess and multimer are not compatible
if args.initial_guess and args.version == "multimer":
    print("WARNING: initial guess and multimer are not compatible. ")
    exit(1)


#######################################################################################################################
# Adapted from code by Nate Bennett for providing initial guess for the alphafold model
import jax.numpy as jnp
from alphafold.common import residue_constants
from alphafold.data import templates
import collections


def af2_get_atom_positions(pymol_object_name) -> Tuple[np.ndarray, np.ndarray]:
    """Gets atom positions and mask."""

    lines = pymol.cmd.get_pdbstr(pymol_object_name).splitlines()

    # indices of residues observed in the structure
    idx_s = [
        int(l[22:26]) for l in lines if l[:4] == "ATOM" and l[12:16].strip() == "CA"
    ]
    num_res = len(idx_s)

    all_positions = np.zeros([num_res, residue_constants.atom_type_num, 3])
    all_positions_mask = np.zeros(
        [num_res, residue_constants.atom_type_num], dtype=np.int64
    )

    residues = collections.defaultdict(list)
    # 4 BB + up to 10 SC atoms
    xyz = np.full((len(idx_s), 14, 3), np.nan, dtype=np.float32)
    for l in lines:
        if l[:4] != "ATOM":
            continue
        resNo, atom, aa = int(l[22:26]), l[12:16], l[17:20]

        residues[resNo].append(
            (atom.strip(), aa, [float(l[30:38]), float(l[38:46]), float(l[46:54])])
        )

    for resNo in residues:
        pos = np.zeros([residue_constants.atom_type_num, 3], dtype=np.float32)
        mask = np.zeros([residue_constants.atom_type_num], dtype=np.float32)

        for atom in residues[resNo]:
            atom_name = atom[0]
            x, y, z = atom[2]
            if atom_name in residue_constants.atom_order.keys():
                pos[residue_constants.atom_order[atom_name]] = [x, y, z]
                mask[residue_constants.atom_order[atom_name]] = 1.0
            elif atom_name.upper() == "SE" and res.get_resname() == "MSE":
                # Put the coordinates of the selenium atom in the sulphur column.
                pos[residue_constants.atom_order["SD"]] = [x, y, z]
                mask[residue_constants.atom_order["SD"]] = 1.0

        idx = idx_s.index(resNo)  # This is the order they show up in the pdb
        all_positions[idx] = pos
        all_positions_mask[idx] = mask
    # _check_residue_distances(
    #     all_positions, all_positions_mask, max_ca_ca_distance) # AF2 checks this but if we want to allow massive truncations we don't want to check this

    return all_positions, all_positions_mask


def af2_all_atom_pymol_object_name(pymol_object_name):
    template_seq = "".join(
        [
            line
            for line in pymol.cmd.get_fastastr(pymol_object_name).split()
            if not line.startswith(">")
        ]
    )

    all_atom_positions, all_atom_mask = af2_get_atom_positions(pymol_object_name)

    all_atom_positions = np.split(all_atom_positions, all_atom_positions.shape[0])

    templates_all_atom_positions = []

    # Initially fill will all zero values
    for _ in template_seq:
        templates_all_atom_positions.append(
            jnp.zeros((residue_constants.atom_type_num, 3))
        )

    for idx, i in enumerate(template_seq):
        templates_all_atom_positions[idx] = all_atom_positions[idx][
            0
        ]  # assign target indices to template coordinates

    return jnp.array(templates_all_atom_positions)


def mk_mock_template(query_sequence):
    # mock template features
    output_templates_sequence = []
    output_confidence_scores = []
    templates_all_atom_positions = []
    templates_all_atom_masks = []

    for _ in query_sequence:
        templates_all_atom_positions.append(
            np.zeros((templates.residue_constants.atom_type_num, 3))
        )
        templates_all_atom_masks.append(
            np.zeros(templates.residue_constants.atom_type_num)
        )
        output_templates_sequence.append("-")
        output_confidence_scores.append(-1)
    output_templates_sequence = "".join(output_templates_sequence)
    templates_aatype = templates.residue_constants.sequence_to_onehot(
        output_templates_sequence, templates.residue_constants.HHBLITS_AA_TO_ID
    )

    template_features = {
        "template_all_atom_positions": np.array(templates_all_atom_positions)[None],
        "template_all_atom_masks": np.array(templates_all_atom_masks)[None],
        "template_sequence": [f"none".encode()],
        "template_aatype": np.array(templates_aatype)[None],
        "template_confidence_scores": np.array(output_confidence_scores)[None],
        "template_domain_names": [f"none".encode()],
        "template_release_date": [f"none".encode()],
    }

    return template_features


#######################################################################################################################


#### load up reference pdb, if present
if args.reference_pdb is not None:
    reference_pdb_name = "REFERENCE"
    pymol.cmd.load(args.reference_pdb, reference_pdb_name)

######################################


with tqdm.tqdm(total=len(query_targets)) as pbar1:
    for length in lengths:
        cohort = length_dict[length]
        for target in cohort:
            pbar1.set_description(f"Input: {target.name}")

            # this is a lazy hack to avoid replacing var names.
            full_sequence = target.seq.replace("/", "")
            query_sequences = target.seq.split("/")
            name = target.name

            pad_mask = np.array([c != "U" for c in full_sequence])

            #############################
            # define input features
            #############################

            num_res = len(full_sequence)
            feature_dict = {}
            msas = [parsers.Msa([full_sequence], [[0] * len(full_sequence)], [name])]
            # deletion_matrices = [[[0]*len(full_sequence)]]

            if args.version == "multimer":

                feature_dict = pipeline_multimer.DataPipelineFaker().process(
                    query_sequences
                )
            else:
                feature_dict.update(
                    pipeline.make_sequence_features(full_sequence, name, num_res)
                )
                feature_dict.update(pipeline.make_msa_features(msas))

                Ls = [len(chain_seq) for chain_seq in query_sequences]
                Ls_plot = sum([[len(seq)] for seq in query_sequences], [])
                # this introduces a bug where the plot just doesn't work for multimer version

                feature_dict["residue_index"] = cf.chain_break(
                    feature_dict["residue_index"], Ls
                )

                if args.initial_guess:
                    feature_dict.update(mk_mock_template(query_sequences))

            ###########################
            # run alphafold
            ###########################
            def parse_results(prediction_result, processed_feature_dict):

                b_factors = (
                    prediction_result["plddt"][:, None]
                    * prediction_result["structure_module"]["final_atom_mask"]
                )
                dist_bins = jax.numpy.append(
                    0, prediction_result["distogram"]["bin_edges"]
                )
                dist_mtx = dist_bins[
                    prediction_result["distogram"]["logits"].argmax(-1)
                ]
                contact_mtx = jax.nn.softmax(prediction_result["distogram"]["logits"])[
                    :, :, dist_bins < 8
                ].sum(-1)

                out = {
                    "unrelaxed_protein": protein.from_prediction(
                        processed_feature_dict,
                        prediction_result,
                        b_factors=b_factors,
                        remove_leading_feature_dimension=args.type != "multimer",
                    ),
                    "plddt": prediction_result["plddt"],
                    "mean_plddt": prediction_result["plddt"].mean(),
                    "dists": dist_mtx,
                    "adj": contact_mtx,
                }

                if "ptm" in prediction_result:
                    out.update(
                        {
                            "pae": prediction_result["predicted_aligned_error"],
                            "pTMscore": prediction_result["ptm"],
                        }
                    )
                if args.type == "multimer":
                    out.update(
                        {
                            "pTMscore": prediction_result["ptm"],
                            "pae": prediction_result["predicted_aligned_error"],
                            "iptm": prediction_result["iptm"],
                        }
                    )
                return out

            if args.models[0] == "all":
                model_names = ["model_1", "model_2", "model_3", "model_4", "model_5"]
            else:
                model_names = [f"model_{model_num}" for model_num in args.models]

            total = len(model_names) * len(seed_range)

            with tqdm.tqdm(total=total) as pbar2:
                #######################################################################
                # precompile model and recompile only if length changes
                #######################################################################
                if args.turbo:
                    if args.type == "multimer":
                        model_name = "model_5_multimer"
                    else:
                        model_name = (
                            "model_5_ptm" if args.type == "monomer_ptm" else "model_5"
                        )

                    N = len(feature_dict["msa"])
                    L = len(feature_dict["residue_index"])
                    compile_settings = (
                        N,
                        L,
                        args.type,
                        args.max_recycles,
                        args.recycle_tol,
                        args.num_ensemble,
                        args.mock_msa_depth,
                        args.enable_dropout,
                        args.pct_seq_mask,
                        args.initial_guess,
                    )

                    recompile = prev_compile_settings != compile_settings

                    if recompile:
                        cf.clear_mem(device)  ##is this ok?
                        cfg = config.model_config(model_name)

                        if args.version == "multimer":
                            cfg.model.num_ensemble_eval = args.num_ensemble
                            # cfg.model.embeddings_and_evoformer.num_extra_msa = args.mock_msa_depth
                            cfg.model.embeddings_and_evoformer.masked_msa.replace_fraction = (
                                args.pct_seq_mask
                            )

                            # templates are enabled by default, but I'm not supplying them, so disable
                            cfg.model.embeddings_and_evoformer.template.enabled = False

                        else:
                            cfg.data.eval.num_ensemble = args.num_ensemble
                            cfg.data.eval.max_msa_clusters = args.mock_msa_depth
                            cfg.data.common.max_extra_msa = args.mock_msa_depth
                            cfg.data.eval.masked_msa_replace_fraction = (
                                args.pct_seq_mask
                            )
                            cfg.data.common.num_recycle = args.max_recycles
                            cfg.model.embeddings_and_evoformer.initial_guess = (
                                args.initial_guess
                            )  # new for initial guessing
                            # do I also need to turn on template?

                        cfg.model.recycle_tol = args.recycle_tol
                        cfg.model.num_recycle = args.max_recycles

                        if args.initial_guess:
                            initial_guess = af2_all_atom_pymol_object_name(
                                target.pymol_obj_name
                            )
                        else:
                            initial_guess = None

                        params = data.get_model_haiku_params(
                            model_name, data_dir=ALPHAFOLD_DATADIR
                        )

                        # hack to bandaid initial guess w/ multimer
                        if args.initial_guess:
                            model_runner = model.RunModel(
                                cfg,
                                params,
                                is_training=args.enable_dropout,
                                return_representations=args.save_intermediates,
                                initial_guess=initial_guess,
                            )
                        else:
                            model_runner = model.RunModel(
                                cfg,
                                params,
                                is_training=args.enable_dropout,
                                return_representations=args.save_intermediates,
                            )

                        prev_compile_settings = compile_settings
                        recompile = False

                else:
                    # cf.clear_mem(device) #is this ok
                    recompile = True

                # cleanup
                if "outs" in dir():
                    del outs
                outs = {}
                # cf.clear_mem(device)   #is this ok?

                #######################################################################

                def report(key):
                    pbar2.update(n=1)
                    o = outs[key]
                    out_dict = {}
                    out_dict["mean_plddt"] = o["mean_plddt"]

                    out_dict["recycles"] = o["recycles"]
                    out_dict["tol"] = o["tol"]
                    out_dict["model"] = key.split("_")[1]
                    out_dict["type"] = args.type
                    out_dict["seed"] = key.split("_")[-1]

                    output_line = f"{name} {key} recycles:{o['recycles']} tol:{o['tol']:.2f} mean_plddt:{o['mean_plddt']:.2f}"
                    if args.type == "monomer_ptm" or args.type == "multimer":
                        output_line += f" pTMscore:{o['pTMscore']:.2f}"

                    prefix = f"{name}_{key}"
                    fout_name = os.path.join(args.out_dir, f"{prefix}_unrelaxed.pdb")

                    output_pdbstr = protein.to_pdb(o["unrelaxed_protein"])
                    output_pdbstr = convert_pdb_chainbreak_to_new_chain(output_pdbstr)
                    output_pdbstr = renumber(output_pdbstr)

                    import string

                    alphabet = string.ascii_uppercase
                    chain_range_map = get_chain_range_map(output_pdbstr)

                    num_chains = len(chain_range_map)

                    final_chain_order = list(
                        alphabet[:num_chains]
                    )  # initialize with original order, basically, for the default case where there is no refernce or input pdb file

                    if args.reference_pdb is not None:
                        pymol.cmd.read_pdbstr(output_pdbstr, oname="temp_target")
                        rmsd, output_pdbstr, final_chain_order = pymol_multichain_align(
                            "temp_target", reference_pdb_name, "super"
                        )  # use super here b/c sequence is not guaranteed to be very similar

                        out_dict["rmsd_to_reference"] = rmsd
                        pymol.cmd.delete("temp_target")
                        output_line += f" rmsd_to_reference:{rmsd:0.2f}"

                    if target.pymol_obj_name is not None:
                        # pymol.cmd.read_pdbstr("\n".join(bfactored_pdb_lines),oname='temp_target')
                        pymol.cmd.read_pdbstr(output_pdbstr, oname="temp_target")
                        rmsd, output_pdbstr, final_chain_order = pymol_multichain_align(
                            "temp_target", target.pymol_obj_name
                        )

                        out_dict["rmsd_to_input"] = rmsd
                        pymol.cmd.delete("temp_target")
                        output_line += f" rmsd_to_input:{rmsd:0.2f}"

                    with open(fout_name, "w") as f:
                        f.write(output_pdbstr)

                    final_chain_order_mapping = {
                        old_chain: new_chain
                        for old_chain, new_chain in zip(alphabet, final_chain_order)
                    }

                    import itertools

                    if args.type == "monomer_ptm":
                        # calculate mean PAE for interactions between each chain pair, taking into account the changed chain order
                        pae = o["pae"]

                        # first, truncate the matrix to the full length of the sequence (without chainbreak characters "/"). It can sometimes be too long because of padding inputs
                        sequence_length = len(
                            target.seq.replace("/", "").replace("U", "")
                        )
                        print("###################### DEGUB ######################")
                        print(sequence_length)
                        print(pae.shape)
                        pae = pae[:sequence_length, :sequence_length]
                        print(pae.shape)

                        if args.output_pae:
                            out_dict["pae"] = pae

                        interaction_paes = []
                        for chain_1, chain_2 in itertools.permutations(
                            final_chain_order, 2
                        ):
                            chain_1_range_start, chain_1_range_stop = chain_range_map[
                                chain_1
                            ]
                            chain_2_range_start, chain_2_range_stop = chain_range_map[
                                chain_2
                            ]

                            final_chain_1 = final_chain_order_mapping[chain_1]
                            final_chain_2 = final_chain_order_mapping[chain_2]
                            interaction_pae = np.mean(
                                pae[
                                    chain_1_range_start:chain_1_range_stop,
                                    chain_2_range_start:chain_2_range_stop,
                                ]
                            )
                            interaction_paes.append(interaction_pae)
                            out_dict[
                                f"mean_pae_interaction_{final_chain_1}{final_chain_2}"
                            ] = interaction_pae

                        # average all the interaction PAEs
                        out_dict["mean_pae_interaction"] = np.mean(interaction_paes)

                        # calculate mean intra-chain PAE per chain
                        intra_chain_paes = []
                        for chain in alphabet[:num_chains]:
                            chain_range_start, chain_range_stop = chain_range_map[chain]
                            intra_chain_pae = np.mean(
                                pae[
                                    chain_range_start:chain_range_stop,
                                    chain_range_start:chain_range_stop,
                                ]
                            )
                            intra_chain_paes.append(intra_chain_pae)
                            out_dict[f"mean_pae_intra_chain_{chain}"] = intra_chain_pae

                        # average all the intrachain PAEs
                        out_dict["mean_pae_intra_chain"] = np.mean(intra_chain_paes)

                        # average all the PAEs
                        out_dict["mean_pae"] = np.mean(pae)

                        out_dict["pTMscore"] = o["pTMscore"]
                    elif args.type == "multimer":
                        out_dict["ptm"] = o["pTMscore"]
                        out_dict["iptm"] = o["iptm"]

                    if args.show_images:
                        fig = cf.plot_protein(
                            o["unrelaxed_protein"], Ls=Ls_plot, dpi=200
                        )
                        plt.savefig(
                            os.path.join(args.out_dir, f"{prefix}.png"),
                            bbox_inches="tight",
                        )
                        plt.close(fig)

                    if args.amber_relax:
                        # Relax the prediction.
                        relaxed_pdb_str, _, _ = amber_relaxer.process(
                            prot=o["unrelaxed_protein"]
                        )

                        # Save the relaxed PDB.
                        relaxed_output_path = os.path.join(
                            args.out_dir, f"relaxed_{model_name}.pdb"
                        )
                        with open(relaxed_output_path, "w") as f:
                            f.write(relaxed_pdb_str)

                    # np.savez_compressed(os.path.join(args.out_dir,f'{prefix}_prediction_results.npz'),**out_dict)

                    # cast devicearray to serializable type
                    for key in out_dict:
                        out_dict[key] = np.array(out_dict[key]).tolist()

                    import json

                    # output as nicely formatted json
                    global time_checkpoint
                    elapsed_time = time.time() - time_checkpoint
                    output_line += f" elapsed time (s): {elapsed_time}"
                    with open("reports.txt", "a") as f:
                        f.write(output_line + "\n")
                    print(output_line)

                    out_dict["elapsed_time"] = elapsed_time

                    with open(
                        os.path.join(args.out_dir, f"{prefix}_prediction_results.json"),
                        "w",
                    ) as f:
                        json.dump(out_dict, f, indent=2)

                    time_checkpoint = time.time()

                #######################################################################

                if args.turbo:
                    # go through each random_seed
                    for seed in seed_range:

                        # prep input features
                        processed_feature_dict = model_runner.process_features(
                            feature_dict, random_seed=seed
                        )

                        # go through each model
                        for num, model_name in enumerate(model_names):
                            model_mod = ""
                            if args.type == "monomer_ptm":
                                model_mod = "_ptm"
                            elif args.type == "multimer":
                                model_mod = "_multimer"
                            model_name = model_name + model_mod
                            key = f"{model_name}_seed_{seed}"
                            pbar2.set_description(f"Running {key}")

                            # check if this prediction/seed has already been done
                            prefix = f"{name}_{key}"
                            if not args.overwrite and os.path.exists(
                                os.path.join(
                                    args.out_dir, f"{prefix}_prediction_results.json"
                                )
                            ):
                                print(
                                    f"{prefix}_prediction_results.json already exists"
                                )
                                continue

                            # replace model parameters
                            params = data.get_model_haiku_params(
                                model_name, data_dir=ALPHAFOLD_DATADIR
                            )
                            for k in model_runner.params.keys():
                                model_runner.params[k] = params[k]

                            # predict
                            if args.initial_guess:
                                prediction_result, (r, t) = cf.to(
                                    model_runner.predict(
                                        processed_feature_dict,
                                        random_seed=seed,
                                        initial_guess=initial_guess,
                                    ),
                                    device,
                                )  # is this ok?
                            else:
                                # a quick hack because the multimer version of the model_runner doesn't have initial_guess in its signature (is that the term?).
                                # the fix will be to update Multimer code to accept initial_guess deep down in the actual code
                                prediction_result, (r, t) = cf.to(
                                    model_runner.predict(
                                        processed_feature_dict, random_seed=seed
                                    ),
                                    device,
                                )  # is this ok?

                            # save results
                            outs[key] = parse_results(
                                prediction_result, processed_feature_dict
                            )
                            outs[key].update({"recycles": r, "tol": t})
                            report(key)

                            del prediction_result, params
                        del processed_feature_dict

                else:
                    # go through each model
                    for num, model_name in enumerate(model_names):
                        model_mod = ""
                        if args.type == "monomer_ptm":
                            model_mod = "_ptm"
                        elif args.type == "multimer":
                            model_mod = "_multimer"
                        model_name = model_name + model_mod
                        params = data.get_model_haiku_params(
                            model_name, data_dir=ALPHAFOLD_DATADIR
                        )
                        cfg = config.model_config(model_name)

                        if args.type == "multimer":
                            cfg.data.num_ensemble_eval = args.num_ensemble
                        else:
                            cfg.data.common.num_recycle = args.max_recycles
                            cfg.data.eval.num_ensemble = args.num_ensemble
                            cfg.model.embeddings_and_evoformer.initial_guess = (
                                args.initial_guess
                            )  # new for initial guessing

                        cfg.model.recycle_tol = args.recycle_tol
                        cfg.model.num_recycle = args.max_recycles

                        if args.initial_guess:
                            initial_guess = af2_all_atom_pymol_object_name(
                                target.pymol_obj_name
                            )
                        else:
                            initial_guess = None

                        if args.initial_guess:
                            model_runner = model.RunModel(
                                cfg,
                                params,
                                is_training=args.enable_dropout,
                                return_representations=args.save_intermediates,
                                initial_guess=initial_guess,
                            )
                        else:
                            model_runner = model.RunModel(
                                cfg,
                                params,
                                is_training=args.enable_dropout,
                                return_representations=args.save_intermediates,
                            )

                        # go through each random_seed
                        for seed in seed_range:
                            key = f"{model_name}_seed_{seed}"
                            pbar2.set_description(f"Running {key}")

                            # check if this prediction/seed has already been done
                            prefix = f"{name}_{key}"
                            if not args.overwrite and os.path.exists(
                                os.path.join(
                                    args.out_dir, f"{prefix}_prediction_results.json"
                                )
                            ):
                                print(
                                    f"{prefix}_prediction_results.json already exists"
                                )
                                continue

                            # print(feature_dict)
                            processed_feature_dict = model_runner.process_features(
                                feature_dict, random_seed=seed
                            )
                            if args.initial_guess:
                                prediction_result, (r, t) = cf.to(
                                    model_runner.predict(
                                        processed_feature_dict,
                                        random_seed=seed,
                                        initial_guess=initial_guess,
                                    ),
                                    device,
                                )  # is this ok?
                            else:
                                prediction_result, (r, t) = cf.to(
                                    model_runner.predict(
                                        processed_feature_dict, random_seed=seed
                                    ),
                                    device,
                                )  # is this ok?

                            # save results
                            outs[key] = parse_results(
                                prediction_result, processed_feature_dict
                            )
                            outs[key].update({"recycles": r, "tol": t})
                            report(key)

                            # cleanup
                            del processed_feature_dict, prediction_result

                        del params, model_runner, cfg

            pbar1.update(1)
