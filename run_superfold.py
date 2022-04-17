__author__ = "Ryan Kibler, Sergey Ovchinnikov, Nate Bennet, Philip Leung, Adam Broerman"
# most of the code is copied from krypton's colabfold https://colab.research.google.com/drive/1teIx4yDyrSm0meWvM9X9yYNLHGO7f_Zy#scrollTo=vJxiCSLc7IWD
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
    help="Paths to PDB files or FASTA files to run AlphaFold2 predictions on. All chains in a PDB file will be predicted as a multichain prediction. To specify chainbreaks in FASTA format, separate sequences with '/' or ':'",
)

# could try using a type here (like input files) to assert that the value is greater than 1. Instead right now we assert below.
parser.add_argument(
    "--mock_msa_depth",
    default=1,
    help="fake the msa. Lower is faster, but potentially less accurate. Range [1,inf). AF2 default is 512. Our Default = 1.",
    type=int,
)

parser.add_argument(
    "--models",
    choices=["1", "2", "3", "4", "5", "all"],
    default="all",
    nargs="+",
    help="Deepmind provided five sets of weights/models. You can choose any combination of models to run.",
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

# #An idea in current colab fold.
# parser.add_argument(
#     "--prediction_threshold",
#     nargs=2,
#     metavar=('value','type'),
#     help="Continue recycling until the prediction is above the threshold or the num_recycles == max_recycles. Type choices are ['mean_plddt','mean_pae','rmsd_prev']",
# )

# unknown if this currently works
parser.add_argument("--show_images", action="store_true")

parser.add_argument(
    "--output_pae",
    action="store_true",
    help="dump the PAE matrix to disk. This is useful for investigating interresidue relationships.",
)

parser.add_argument(
    "--output_summary",
    action="store_true",
    help="write a 1-line summary of each prediction to disk under output_dir named 'reports.txt'.",
)

# # unknown if this currently works
# parser.add_argument(
#     "--save_intermediates",
#     action="store_true",
#     help="save intermediate structures between recycles. This is useful for making folding movies/trajectories",
# )

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
    "--reference_pdb",
    type=str,
    help="reference PDB to use for RMSD calculations. Coordinates (after alignment) and chain order will be updated to that of this reference, unless the input_files are PDB files",
)

parser.add_argument(
    "--simple_rmsd",
    action="store_true",
    help="compute RMSD directly with the alphafold prediction and without trying to rearrange chain orders. USE THIS FLAG IF YOU HAVE DIFFERENT CHAINS IN YOUR INPUT PDB FILE. RMSD calculations are unreliable otherwise",
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

parser.add_argument(
    "--out_dir",
    type=str,
    default="output/",
    help="Directory to output models and data.",
)

args = parser.parse_args()

#adding this to keep code working later on while I figure out how to make it work
args.save_intermediates = False

assert args.mock_msa_depth > 0

from pathlib import Path
import pymol
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "silent_tools"))
import silent_tools  # installed as a submodule
from dataclasses import dataclass
from typing import Union, Tuple, Dict
import numpy as np
from matplotlib import pyplot as plt

plt.switch_backend("agg")

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

def pymol_align(pymol_object_name: str, reference_pdb: str, alignmnet_mode: str = "align") -> float:
    """
    Naively align the output of alphafold to the reference pdb
    """
    import pymol

    align_func = getattr(pymol.cmd, alignmnet_mode)
    rmsd = align_func(
        f"{pymol_object_name} and n. CA",
        f"{reference_pdb} and n. CA",
        cycles=0
    )[0]

    return rmsd

# TODO refactor this code to only rearrange identical sequences to find better fits.
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

    alphabet = string.ascii_uppercase + string.digits + string.ascii_lowercase
    for line in pdbstring.split("\n"):
        if line[:4] == "ATOM":
            resid = int(line[22:26])
            if resid - previous_resid > 1:
                chain_num += 1
                if chain_num >= len(alphabet):
                    raise Exception(
                        "Too many chains to convert to new chain format. "
                        "Decrease the number of chains or increase the alphabet size."
                    )
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
    seq = seq.replace(":","/")
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
        "=======================================================================================\n"
        + f"WARNING: Your query{plural} shorter than 400 residues. This is a very small protein.\n"
        + "You may want to use the CPU to conserve GPU resources for those who need them.\n"
        + "Remember that you can launch far more jobs in parallel on CPUs than you can on GPUs...\n"
        + "See this example of how prediction time scales on CPU vs GPU: \n"
        + "https://docs.google.com/spreadsheets/d/1jTGITpIx6fJehAplUkXtePOp7me3Dpq_pPKHn68F7XY\n"
        + "======================================================================================="
    )

seed_range = list(range(args.seed_start, args.seed_start + args.nstruct))

# blatently stolen from https://github.com/sokrypton/ColabFold/blob/8e6b6bb582f40a4fea06b19fc001d3d9ca208197/colabfold/alphafold/msa.py#L15
# by konstin i think
# no worries, I plan on going and actually forking colabfold eventually.
from alphafold.model.features import FeatureDict
from alphafold.model.tf import shape_placeholders
import tensorflow as tf

NUM_RES = shape_placeholders.NUM_RES
NUM_MSA_SEQ = shape_placeholders.NUM_MSA_SEQ
NUM_EXTRA_SEQ = shape_placeholders.NUM_EXTRA_SEQ
NUM_TEMPLATES = shape_placeholders.NUM_TEMPLATES


def make_fixed_size(feat, runner, max_length):
    """pad input features"""
    cfg = runner.config

    if cfg.model.global_config.multimer_mode:
        # shape_schema = ?
        # pad_size_map = {
        #     shape_placeholders.NUM_RES: max_length,
        #     shape_placeholders.NUM_MSA_SEQ: cfg.model.embeddings_and_evoformer.num_msa,
        #     shape_placeholders.NUM_EXTRA_SEQ: cfg.model.embeddings_and_evoformer.num_extra_msa,
        #     shape_placeholders.NUM_TEMPLATES: 0,
        # }
        print("Warning: padding sequences in multimer mode is not implemented yet")
        return feat
    else:
        shape_schema = {k: [None] + v for k, v in dict(cfg.data.eval.feat).items()}
        pad_size_map = {
            shape_placeholders.NUM_RES: max_length,
            shape_placeholders.NUM_MSA_SEQ: cfg.data.eval.max_msa_clusters,
            shape_placeholders.NUM_EXTRA_SEQ: cfg.data.common.max_extra_msa,
            shape_placeholders.NUM_TEMPLATES: 0,
        }


    for k, v in feat.items():
        # Don't transfer this to the accelerator.
        if k == "extra_cluster_assignment":
            continue
        shape = list(v.shape)

        schema = shape_schema[k]
        assert len(shape) == len(schema), (
            f"Rank mismatch between shape and shape schema for {k}: "
            f"{shape} vs {schema}"
        )
        pad_size = [pad_size_map.get(s2, None) or s1 for (s1, s2) in zip(shape, schema)]
        padding = [(0, p - tf.shape(v)[i]) for i, p in enumerate(pad_size)]
        if padding:
            feat[k] = tf.pad(v, padding, name=f"pad_to_fixed_{k}")
            feat[k].set_shape(pad_size)
    return {k: np.asarray(v) for k, v in feat.items()}



#### load up reference pdb, if present
if args.reference_pdb is not None:
    reference_pdb_name = "REFERENCE"
    pymol.cmd.load(args.reference_pdb, reference_pdb_name)

######################################

max_length = max([len(tgt) for tgt in query_targets])

if args.type == "multimer":
    model_name = "model_5_multimer"
else:
    model_name = "model_5_ptm" if args.type == "monomer_ptm" else "model_5"

if "all" in args.models:
    model_names = ["model_1", "model_2", "model_3", "model_4", "model_5"]
else:
    model_names = [f"model_{model_num}" for model_num in args.models]


cfg = config.model_config(model_name)
params = data.get_model_haiku_params(model_name, data_dir=ALPHAFOLD_DATADIR)

if args.version == "multimer":
    cfg.model.num_ensemble_eval = args.num_ensemble
    # cfg.model.embeddings_and_evoformer.num_extra_msa = args.mock_msa_depth
    cfg.model.embeddings_and_evoformer.masked_msa.replace_fraction = args.pct_seq_mask

    # templates are enabled by default, but I'm not supplying them, so disable
    cfg.model.embeddings_and_evoformer.template.enabled = False

else:
    cfg.data.eval.num_ensemble = args.num_ensemble
    cfg.data.eval.max_msa_clusters = args.mock_msa_depth
    cfg.data.common.max_extra_msa = args.mock_msa_depth
    cfg.data.eval.masked_msa_replace_fraction = args.pct_seq_mask
    cfg.data.common.num_recycle = args.max_recycles
    # do I also need to turn on template?

cfg.model.recycle_tol = args.recycle_tol
cfg.model.num_recycle = args.max_recycles


model_runner = model.RunModel(
    cfg,
    params,
    is_training=args.enable_dropout,
    return_representations=args.save_intermediates,
)

with tqdm.tqdm(total=len(query_targets)) as pbar1:
    for target in query_targets:
        pbar1.set_description(f"Input: {target.name}")

        # this is a lazy hack to avoid replacing var names.
        full_sequence = target.seq.replace("/", "")
        query_sequences = target.seq.split("/")
        name = target.name

        #############################
        # define input features
        #############################

        num_res = len(full_sequence)
        feature_dict = {}
        msas = [parsers.Msa([full_sequence], [[0] * len(full_sequence)], [name])]

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

        ###########################
        # run alphafold
        ###########################
        def parse_results(prediction_result, processed_feature_dict):

            # figure note... it would be nice to use prediction_result["structure_module"]["final_atom_mask"] to mask out everything in prediction_result that shouldn't be there due to padding.
            b_factors = (
                prediction_result["plddt"][:, None]
                * prediction_result["structure_module"][
                    "final_atom_mask"
                ]  # I think not needed b/c I truncated the vector earlier
            )

            # but for now let's focus on truncating the results we most care about to the length of the target sequence
            prediction_result["plddt"] = prediction_result["plddt"][: len(target.seq)]
            if "predicted_aligned_error" in prediction_result:
                prediction_result["predicted_aligned_error"] = prediction_result[
                    "predicted_aligned_error"
                ][: len(target.seq), : len(target.seq)]

            dist_bins = jax.numpy.append(0, prediction_result["distogram"]["bin_edges"])
            dist_mtx = dist_bins[prediction_result["distogram"]["logits"].argmax(-1)]
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

        total = len(model_names) * len(seed_range)

        with tqdm.tqdm(total=total) as pbar2:
            outs = {}

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

                alphabet = string.ascii_uppercase + string.digits + string.ascii_lowercase
                chain_range_map = get_chain_range_map(output_pdbstr)

                num_chains = len(chain_range_map)

                final_chain_order = list(
                    alphabet[:num_chains]
                )  # initialize with original order, basically, for the default case where there is no refernce or input pdb file

                pymol.cmd.read_pdbstr(output_pdbstr, oname="temp_target")
                if args.reference_pdb is not None:
                    if args.simple_rmsd:
                        rmsd = pymol_align(
                            "temp_target",
                            reference_pdb_name, "super"
                        )
                    else:
                        rmsd, output_pdbstr, final_chain_order = pymol_multichain_align(
                            "temp_target", reference_pdb_name, "super"
                        )  # use super here b/c sequence is not guaranteed to be very similar

                    out_dict["rmsd_to_reference"] = rmsd
                    pymol.cmd.delete("temp_target")
                    output_line += f" rmsd_to_reference:{rmsd:0.2f}"

                if target.pymol_obj_name is not None:
                    if args.simple_rmsd:
                        rmsd = pymol_align(
                            "temp_target", target.pymol_obj_name
                        )
                    else:

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
                    sequence_length = len(target.seq.replace("/", "").replace("U", ""))
                    pae = pae[:sequence_length, :sequence_length]

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
                    fig = cf.plot_protein(o["unrelaxed_protein"], Ls=Ls_plot, dpi=200)
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
                        args.out_dir, f"{prefix}_relaxed.pdb"
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

                if args.output_summary:
                    with open(os.path.join(args.out_dir, "reports.txt"), "a") as f:
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

            # go through each random_seed
            for seed in seed_range:

                # prep input features
                processed_feature_dict = model_runner.process_features(
                    feature_dict, random_seed=seed
                )

                # pad input features
                # Pad sequences to the same length
                ##sequence padding
                # I'm not sure if this is compatible with multimer version or not, but I'll stick it here for now
                # model_config = model_runner.config
                # eval_cfg = model_config.data.eval
                # crop_feats = {k: [None] + v for k, v in dict(eval_cfg.feat).items()}
                # print(crop_feats)
                # feature_dict = make_fixed_size(
                #     feature_dict,
                #     crop_feats,
                #     args.mock_msa_depth,
                #     args.mock_msa_depth,
                #     max_length,
                # )
                processed_feature_dict = make_fixed_size(
                    processed_feature_dict, model_runner, max_length
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
                        os.path.join(args.out_dir, f"{prefix}_prediction_results.json")
                    ):
                        print(f"{prefix}_prediction_results.json already exists")
                        continue

                    # replace model parameters
                    params = data.get_model_haiku_params(
                        model_name, data_dir=ALPHAFOLD_DATADIR
                    )
                    for k in model_runner.params.keys():
                        model_runner.params[k] = params[k]

                    # predict
                    prediction_result, (r, t) = cf.to(
                        model_runner.predict(
                            processed_feature_dict, random_seed=seed
                        ),
                        device,
                    )  # is this ok?

                    # save results
                    outs[key] = parse_results(prediction_result, processed_feature_dict)
                    outs[key].update({"recycles": r, "tol": t})
                    report(key)

                    del prediction_result, params
                del processed_feature_dict

        pbar1.update(1)
