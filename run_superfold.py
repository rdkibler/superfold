#!/home/rdkibler/.conda/envs/pyroml/bin/python3.8

#this is almost entirely based on krypton's stuff https://colab.research.google.com/drive/1teIx4yDyrSm0meWvM9X9yYNLHGO7f_Zy#scrollTo=vJxiCSLc7IWD

import argparse
import os
#from Bio import SeqIO
parser = argparse.ArgumentParser()


#This hack is probably unnecessary with AF2-multimer since they've switched to jax for feature processing
#tell Jax and Tensorflow to use the same memory. This allows us to run larger structures
os.environ['TF_FORCE_UNIFIED_MEMORY'] = '1'  
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '2.0' 


#let's use a linkfile-like strategy for telling the script where to find stuff like data
SCRIPTDIR = os.path.dirname(os.path.realpath(__file__))
with open(f"{SCRIPTDIR}/alphafold_weights.pth",'r') as f:
  ALPHAFOLD_DATADIR=f.read().strip()
assert(os.path.exists(ALPHAFOLD_DATADIR))

def validate_file(parser, path):
        """
        Check for file existance and read files first so that we can fail early before loading alphafold, etc

        https://stackoverflow.com/a/11541450
        """
        if not os.path.exists(path):
                parser.error("The file %s does not exist!" % path)
        else:
                if ( path.endswith(".pdb") or path.endswith(".pdb.gz") 
                     or path.lower().endswith(".fasta") or path.lower().endswith(".fa") 
                     or path.lower().endswith(".silent")):
                    return path
                else:
                  parser.error("Only PDB files, silent files, and FASTA files are allowed. You supplied: %s" % path)

parser.add_argument("input_files",metavar="PATH",nargs="+",type=lambda x: validate_file(parser, x),help="Paths to PDB files or FASTA files to run AlphaFold2 predictions on.")

parser.add_argument("--pad_lengths",action="store_true",help="compile the model once to the longest input PDB and pad the remaining sequences. Ryan is unsure how this affects prediction accuracy, but it will speed up multiple prediction.")
parser.add_argument("--mock_msa_depth", default=512,help="fake the msa. Default = 512. to go fast, use 1",type=int)
parser.add_argument("--models",choices=["1","2","3","4","5","all"],default="4",nargs="+",help="Deepmind provided five sets of weights/models. You can choose any combination of models to run. The model number 5 has been found (by aivan) to perform the best on single sequences so this is the default, but using multiple models might provide you with a relevent ensemble of structures.")
parser.add_argument("--type", choices=['monomer','monomer_ptm','multimer'] ,default="monomer",help="The flavor of alphafold weights to use. 'monomer' is the original AF2. 'ptm' is the original AF2 with an extra head that predicts pTMscore. 'multimer' is AF2-Multimer. The use of multimer weights with standard AF2 probably won't work")
parser.add_argument("--version",choices=["monomer","multimer"],default="monomer",help="The version of AF2 Module to use. Both versions can predict both mulimers. When used to predict multimers, the 'monomer' version is equivalent to AF2-Gap. The 'multimer' version is equivalent to AF2-Multimer and should not be used with the monomer weight types.")


parser.add_argument("--nstruct",type=int,help="Number of independent outputs to generate PER MODEL. Each starts with a different predetermined seed. Default=1", default=1)
parser.add_argument("--num_ensemble",type=int,default=1,help="number of times to process the input features and combine. default = 1. Deepmind used 8 for casp. Expert Option.")
parser.add_argument("--turbo",action="store_true",help="use the latest and greatest hacks to make it run fast fast fast.")
parser.add_argument("--max_recycles",type=int,default=3,help="max number of times to run evoformer. Default is 3. Single domain proteins need fewer runs. Multidomain or PPI may need more")
parser.add_argument("--recycle_tol",type=float,default=0.0,help="Stop recycling early if CA-RMSD difference between current output and previous is < recycle_tol. Default = 0.0 (no early stopping)")
parser.add_argument("--show_images",action="store_true")
# parser.add_argument("--ptm",action="store_true",help="use the version of the models that output predicted TMalign score. Incompatible with 'multimer' mode.")
# parser.add_argument("--multimer",action="store_true",help="Use the Alphafold-Multimer module and weights. Incompatible with 'ptm' mode.")
parser.add_argument("--save_intermediates",action="store_true",help="save intermediate structures between recycles. This is useful for making folding movies/trajectories")

parser.add_argument("--amber_relax",action="store_true",help="use AMBER to relax the structure after prediction")
# sidechain_relax_parser = parser.add_mutually_exclusive_group(required=False)
# sidechain_relax_parser.add_argument("--amber_relax",help="run Amber relax on each output prediction")
# sidechain_relax_parser.add_argument("--rosetta_relax",help="run Rosetta relax (sidechain only) on each output prediction")

parser.add_argument("--enable_dropout",action="store_true",help="Introduce structural diversity by enabling dropout")
parser.add_argument("--pct_seq_mask", type=float,default=0.15,help="percent of sequence to make during inference. Default = 15% (0.15). Setting to 0 might reduce prediction stocasticity.")
#parser.add_argument("--deepaccnet",action="store_true",help="Run DeepAccNet on the AlphaFold2 outputs.")

parser.add_argument("--out_dir",type=str,default="output/",help="Directory to output models and data.")
possible_prediction_results = ['unrelaxed_protein', 'plddt', 'mean_plddt', 'dists', 'adj', 'pae', 'pTMscore', 'recycles', 'tol']

parser.add_argument("--save_prediction_results",choices=possible_prediction_results + ['all'], nargs="+",default=['mean_plddt'], help="save the data returned by AF2. Warning, this could be big! Default: [mean_plddt]")

args = parser.parse_args()

#this could be improved
if "all" in args.save_prediction_results:
  args.save_prediction_results = possible_prediction_results

import os
import pymol
import silent_tools
from dataclasses import dataclass
from typing import Union, Tuple, Dict
import numpy as np
import pickle
from matplotlib import pyplot as plt
plt.switch_backend('agg')



os.makedirs(args.out_dir,exist_ok=True)

def pymol_renumber(sele="all"):
  pdbstr = pymol.cmd.get_pdbstr(sele)

  previous_resid = None


  new_resid = 0
  new_atomid = 1

  lines = [line for line in pdbstr.split("\n") if line[:4] == "ATOM"]
  fixed_pdbstr = ""

  for line in lines:
    resid = int(line[22:26])
    if (resid != previous_resid):
      new_resid += 1
    new_line = line[:6] + f"{new_atomid: >5}" + line[11:22] + f"{str(new_resid): >4}" + line[26:]
    previous_resid = resid
    fixed_pdbstr += new_line + "\n"
    new_atomid += 1

  pymol.cmd.delete(sele)
  pymol.cmd.read_pdbstr(fixed_pdbstr,sele)

def pymol_apply_new_chains(pymol_object_name:str, new_chains:list) -> None:
    """
    Applies the new chains to a pymol object in the current order of chains

    i.e. if the pymol object has chains A, B, C, D, and the new_chains list is D, B, A, C
    it will turn current chain A into chain D, current chain B into chain B, 
    ccurrent chain C into chain A, and current chain D into chain C.
    """
    import pymol

    #first, make selections for each current chain so we don't lose track of them
    chain_selections = []
    for i, chain in enumerate(pymol.cmd.get_chains(pymol_object_name)):
        chain_name = f'{pymol_object_name}_chain_{i}'
        pymol.cmd.select(chain_name, f'{pymol_object_name} and chain {chain}')
        chain_selections.append(chain_name)

    #print(pymol.cmd.get_names('all', 0))

    #now, apply the new chains
    for chain_selection, new_chain in zip(chain_selections, new_chains):
        pymol.cmd.alter(chain_selection, f'chain = "{new_chain}"')
    
    pymol.cmd.sort(pymol_object_name)

def get_chain_permutations(chains:list) -> list:
    """
    Gets all permutations of the chains.
    """
    import itertools

    return list(itertools.permutations(chains))
    
def pymol_multichain_align(model_pymol_name:str, reference_pymol_name:str) -> Tuple[float, str]:
    """
    Aligns two multichain models using pymol.
    Returns the RMSD and the aligned model.
    """
    import pymol
    import random
    import string

    #generate a random prefix so we don't overwrite anything else in the pymol session
    prefix = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(5))

    temp_pymol_name = f'{prefix}_temp'
    best_pymol_name = f'{prefix}_best'

    chains = pymol.cmd.get_chains(model_pymol_name)

    best_rmsd = float('inf')
    for new_order in get_chain_permutations(chains):
        #make a temporary object with the new order of chains
        pymol.cmd.delete(temp_pymol_name)
        pymol.cmd.create(temp_pymol_name, model_pymol_name)
        pymol_apply_new_chains(temp_pymol_name, new_order)
        rmsd = pymol.cmd.align(f'{temp_pymol_name} and n. CA', f'{reference_pymol_name} and n. CA',cycles=0)[0]
        
        #debug: useful to see if alignment is working
        #print(f'{rmsd} {new_order}')

        if rmsd < best_rmsd:
            best_rmsd = rmsd
            pymol.cmd.create(best_pymol_name, temp_pymol_name)

    best_pdbstr = pymol.cmd.get_pdbstr(best_pymol_name)

    #clean up
    pymol.cmd.delete(temp_pymol_name)
    pymol.cmd.delete(best_pymol_name)

    return best_rmsd, best_pdbstr

@dataclass(frozen=True)
class PredictionTarget:
  name: str
  seq: str
  pymol_obj_name: str = None

  def __lt__(self,other):
    return len(self) < len(other)
  
  def __len__(self):
    return len(self.seq)

  def padseq(self,pad_amt):
    return PredictionTarget(self.name,self.seq + "U"*pad_amt,self.pymol_obj_name)

def parse_fasta(path):
  if path.endswith(".gz"):
    import gzip
    filehandle = gzip.open(path,'rt')
  else:
    filehandle = open(path,'rt')

  outputs = []

  seq = ""
  name = ""
  for line in filehandle:
    if line.startswith(">"):
      if len(seq) > 0:
        outputs.append(PredictionTarget(name,seq))
      name = line[1:].strip()
      seq = ""
    else:
      seq += line.strip()
  if len(seq) > 0:
    #This should always be true for a well formatted fasta file
    outputs.append(PredictionTarget(name,seq))

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
  pymol.cmd.load(path, object = pymol_obj_name)
  fastastring = pymol.cmd.get_fastastr(pymol_obj_name)
  #kinda obtuse, sorry. Basically, split the string on newlines and throw out the leading ">blahblahblah" line b/c we don't need it
  #then step through and (eventuallY) concat the normal seq lines. When it encounters a new ">blahbalbha" line, this signifies a chain
  #break and put in the chainbreak char instead
  seq = "".join([line if not line.startswith(">") else "/" for line in fastastring.split()[1:]])


  return [PredictionTarget(name,seq,pymol_obj_name)]

def parse_silent(path):
  outputs = []
  index = silent_tools.get_silent_index(path)

  tags = index['tags']

  structures = silent_tools.get_silent_structures(path,index,tags)

  for name,structure in zip(tags,structures):

    chain_per_res = silent_tools.get_chain_ids(structure)

    #only gonna grab C-alphas
    seq = "".join(silent_tools.get_sequence_chunks(structure))
    #atoms = silent_tools.sketch_get_cas_protein_struct(structure)
    atoms = silent_tools.sketch_get_atoms(structure,1)
    pdbstring = silent_tools.write_pdb_atoms(atoms, seq, ["CA"], chain_ids=chain_per_res)
    pymol_obj_name = get_unique_name()
    pymol.cmd.read_pdbstr("".join(pdbstring), oname=pymol_obj_name)

    #but you ask "why? you already have 'seq'!" Well, this one has chain breaks and I already wrote the code above for the pdb parsing
    fastastring = pymol.cmd.get_fastastr(pymol_obj_name)
    seq = "".join([line if not line.startswith(">") else "/" for line in fastastring.split()[1:]])

    outputs.append(PredictionTarget(name,seq,pymol_obj_name))

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

#I don't know if this is a good idea.
if args.version == "multimer":
  from alphafold.data import pipeline_multimer
from alphafold.data import pipeline

import colabfold as cf
from collections import defaultdict
import tqdm
import jax
from jax.lib import xla_bridge
device = xla_bridge.get_backend().platform
print(device)


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
      max_outer_iterations=RELAX_MAX_OUTER_ITERATIONS)



if args.pad_lengths:
  #I don't htink this is the best implememntation wrt the use of "U" to pad
  longest = max([len(tgt) for tgt in query_targets])
  query_targets = [tgt.padseq(longest - len(tgt)) for tgt in query_targets]


length_dict = defaultdict(lambda: [])
for tgt in query_targets:
  length_dict[len(tgt)].append(tgt)

#sort so longer first so it fails early
lengths = sorted(length_dict.keys(),reverse=True)

if args.pad_lengths:
  assert(len(lengths) == 1)

prev_compile_settings = tuple()

with tqdm.tqdm(total=len(query_targets)) as pbar1:
  for length in lengths:
    cohort = length_dict[length]
    for target in cohort:
      pbar1.set_description(f"Input: {target.name}")

      #this is a lazy hack to avoid replacing var names. 
      full_sequence = target.seq.replace("/","")
      query_sequences = target.seq.split("/")
      name = target.name

      pad_mask = np.array([c != "U" for c in full_sequence])
      
      #############################
      # define input features
      #############################

      num_res = len(full_sequence)
      feature_dict = {}
      msas = [parsers.Msa([full_sequence],[[0]*len(full_sequence)],[name])]
      # deletion_matrices = [[[0]*len(full_sequence)]]

    

      if args.version == "multimer":
        
        feature_dict = pipeline_multimer.DataPipelineFaker().process(query_sequences)
      else:
        feature_dict.update(pipeline.make_sequence_features(full_sequence, name, num_res))
        feature_dict.update(pipeline.make_msa_features(msas))

        Ls = [len(chain_seq) for chain_seq in query_sequences]
        Ls_plot = sum([[len(seq)] for seq in query_sequences],[])
        #this introduces a bug where the plot just doesn't work for multimer version

        feature_dict['residue_index'] = cf.chain_break(feature_dict['residue_index'], Ls)





      ###########################
      # run alphafold
      ###########################
      def parse_results(prediction_result, processed_feature_dict):
        
        b_factors = prediction_result['plddt'][:,None] * prediction_result['structure_module']['final_atom_mask']  
        dist_bins = jax.numpy.append(0,prediction_result["distogram"]["bin_edges"])
        dist_mtx = dist_bins[prediction_result["distogram"]["logits"].argmax(-1)]
        contact_mtx = jax.nn.softmax(prediction_result["distogram"]["logits"])[:,:,dist_bins < 8].sum(-1)

        out = {"unrelaxed_protein": protein.from_prediction(processed_feature_dict, prediction_result, b_factors=b_factors,remove_leading_feature_dimension = args.type != "multimer"),
               "plddt": prediction_result['plddt'],
               "mean_plddt": prediction_result['plddt'].mean(),
               "dists": dist_mtx,
               "adj": contact_mtx}

        if "ptm" in prediction_result:
          out.update({"pae": prediction_result['predicted_aligned_error'],
                      "pTMscore": prediction_result['ptm']})
        return out


      if args.models[0] == "all":
        model_names = ["model_1","model_2","model_3","model_4","model_5"]
      else:
        model_names = [f"model_{model_num}" for model_num in args.models]

      total = len(model_names) * args.nstruct


      with tqdm.tqdm(total=total) as pbar2:
        #######################################################################
        # precompile model and recompile only if length changes
        #######################################################################
        if args.turbo:
          if args.type == "multimer":
            model_name = "model_5_multimer"
          else:
            model_name = "model_5_ptm" if args.type == "ptm" else "model_5"

          N = len(feature_dict["msa"])
          L = len(feature_dict["residue_index"])
          compile_settings = (N, L, args.type, args.max_recycles, args.recycle_tol, args.num_ensemble, args.mock_msa_depth, args.enable_dropout, args.pct_seq_mask)

          recompile = prev_compile_settings != compile_settings

          if recompile:
            cf.clear_mem(device) ##is this ok?
            cfg = config.model_config(model_name)


            if args.version == "multimer":
              cfg.model.num_ensemble_eval = args.num_ensemble
              #cfg.model.embeddings_and_evoformer.num_extra_msa = args.mock_msa_depth
              cfg.model.embeddings_and_evoformer.masked_msa.replace_fraction = args.pct_seq_mask

              #templates are enabled by default, but I'm not supplying them, so disable
              cfg.model.embeddings_and_evoformer.template.enabled = False
              
            else:
              cfg.data.eval.num_ensemble = args.num_ensemble
              cfg.data.eval.max_msa_clusters = args.mock_msa_depth
              cfg.data.common.max_extra_msa = args.mock_msa_depth
              cfg.data.eval.masked_msa_replace_fraction = args.pct_seq_mask
              cfg.data.common.num_recycle = args.max_recycles

            # set size of msa (to reduce memory requirements)
            
            
            cfg.model.recycle_tol = args.recycle_tol
            cfg.model.num_recycle = args.max_recycles
            
            


            params = data.get_model_haiku_params(model_name,data_dir=ALPHAFOLD_DATADIR)
            model_runner = model.RunModel(cfg, params, is_training=args.enable_dropout, return_representations=args.save_intermediates)
            prev_compile_settings = compile_settings
            recompile = False

        else:
          #cf.clear_mem(device) #is this ok
          recompile = True

        # cleanup
        if "outs" in dir(): del outs
        outs = {}
        #cf.clear_mem(device)   #is this ok?

        #######################################################################
        def report(key):
          pbar2.update(n=1)
          o = outs[key]

          output_line = f"{name} {key} recycles:{o['recycles']} tol:{o['tol']:.2f} mean_plddt:{o['mean_plddt']:.2f}"
          if args.type == "ptm" or args.type == 'multimer': output_line += f" pTMscore:{o['pTMscore']:.2f}"

          prefix = f"{name}_{key}_recycle_{o['recycles']}" 
          fout_name = os.path.join(args.out_dir,f'{prefix}_unrelaxed.pdb')


          #add bfactor. suggested by drhicks
          seq_id = -1
          seq_id_then = ""
          bfactored_pdb_lines = []
          bfac = o['plddt']/100

          for line in protein.to_pdb(o["unrelaxed_protein"]).split("\n"):
            if line[0:6] == "ATOM  ":
              seq_id_now = int(line[23:26].strip()) - 1
              if seq_id_now != seq_id_then:
                seq_id += 1
              bfactored_pdb_lines.append("{before_section}{bfac:6.2f}{after_section}".format(before_section=line[:60], bfac=bfac[seq_id], after_section=line[66:]))
              seq_id_then = int(line[23:26].strip()) - 1
            else:
              bfactored_pdb_lines.append(line)


          if target.pymol_obj_name is not None:
            pymol.cmd.read_pdbstr("\n".join(bfactored_pdb_lines),oname='temp_target')
            rmsd,bfactored_pdb_lines = pymol_multichain_align('temp_target',target.pymol_obj_name)
            bfactored_pdb_lines = bfactored_pdb_lines.split("\n")

            o['rmsd_to_design'] = rmsd
            pymol.cmd.delete('temp_target')
            output_line += f" rmsd_to_input:{rmsd:0.2f}"

          with open(fout_name, 'w') as f:
            f.write("\n".join(bfactored_pdb_lines))

          with open('reports.txt','a') as f:
            f.write(output_line+"\n")
          print(output_line)

          if args.show_images:
            fig = cf.plot_protein(o["unrelaxed_protein"], Ls=Ls_plot, dpi=200)
            plt.savefig(os.path.join(args.out_dir,f'{prefix}.png'), bbox_inches = 'tight')
            plt.close(fig)


          if args.amber_relax:
            # Relax the prediction.
            relaxed_pdb_str, _, _ = amber_relaxer.process(prot=o["unrelaxed_protein"])

            #inject bfactor
            seq_id = -1
            seq_id_then = ""
            bfactored_relaxed_pdb_lines = []
            bfac = o['plddt']/100

            for line in protein.to_pdb(relaxed_pdb_str).split("\n"):
              if line[0:6] == "ATOM  ":
                seq_id_now = int(line[23:26].strip()) - 1
                if seq_id_now != seq_id_then:
                  seq_id += 1
                bfactored_relaxed_pdb_lines.append("{before_section}{bfac:6.2f}{after_section}".format(before_section=line[:60], bfac=bfac[seq_id], after_section=line[66:]))
                seq_id_then = int(line[23:26].strip()) - 1
              else:
                bfactored_relaxed_pdb_lines.append(line)

            # Save the relaxed PDB.
            relaxed_output_path = os.path.join(
                args.out_dir, f'relaxed_{model_name}.pdb')
            with open(relaxed_output_path, 'w') as f:
              f.write("\n".join(bfactored_relaxed_pdb_lines))


          out_dict = {k:o[k] for k in args.save_prediction_results}

          np.savez_compressed(os.path.join(args.out_dir,f'{prefix}_prediction_results.npz'),**out_dict)


        #######################################################################


        if args.turbo:
          # go through each random_seed
          for seed in range(args.nstruct):
            
            # prep input features
            processed_feature_dict = model_runner.process_features(feature_dict, random_seed=seed)

            # go through each model
            for num, model_name in enumerate(model_names):
              model_mod = ""
              if args.type == "ptm":
                model_mod = "_ptm"
              elif args.type == "multimer":
                model_mod = "_multimer"
              model_name = model_name+model_mod
              key = f"{model_name}_seed_{seed}"
              pbar2.set_description(f'Running {key}')

              # replace model parameters
              params = data.get_model_haiku_params(model_name, data_dir=ALPHAFOLD_DATADIR)
              for k in model_runner.params.keys():
                model_runner.params[k] = params[k]

              # predict
              prediction_result, (r, t) = cf.to(model_runner.predict(processed_feature_dict, random_seed=seed),device) #is this ok?

              # save results
              outs[key] = parse_results(prediction_result, processed_feature_dict)
              outs[key].update({"recycles":r, "tol":t})
              report(key)

              del prediction_result, params
            del processed_feature_dict

        else:  
          # go through each model
          for num, model_name in enumerate(model_names):
            model_mod = ""
            if args.type == "ptm":
              model_mod = "_ptm"
            elif args.type == "multimer":
              model_mod = "_multimer"
            model_name = model_name+model_mod
            params = data.get_model_haiku_params(model_name, data_dir=ALPHAFOLD_DATADIR)  
            cfg = config.model_config(model_name)

            if args.type == "multimer":
              cfg.data.num_ensemble_eval = args.num_ensemble
            else:
              cfg.data.common.num_recycle =args.max_recycles
              cfg.data.eval.num_ensemble = args.num_ensemble

            cfg.model.recycle_tol = args.recycle_tol
            cfg.model.num_recycle = args.max_recycles


            model_runner = model.RunModel(cfg, params, is_training=args.enable_dropout, return_representations=args.save_intermediates)

            # go through each random_seed
            for seed in range(args.nstruct):
              key = f"{model_name}_seed_{seed}"
              pbar2.set_description(f'Running {key}')
              #print(feature_dict)
              processed_feature_dict = model_runner.process_features(feature_dict, random_seed=seed)
              prediction_result, (r, t) = cf.to(model_runner.predict(processed_feature_dict, random_seed=seed),device) #is this ok?

              # save results
              outs[key] = parse_results(prediction_result, processed_feature_dict)
              outs[key].update({"recycles":r, "tol":t})
              report(key)

              # cleanup
              del processed_feature_dict, prediction_result

            del params, model_runner, cfg


      pbar1.update(1)

