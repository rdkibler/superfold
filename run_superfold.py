#!/home/rdkibler/.conda/envs/pyroml/bin/python3.8

#!/software/conda/envs/SE3/bin/python


#this is almost entirely based on krypton's stuff https://colab.research.google.com/drive/1teIx4yDyrSm0meWvM9X9yYNLHGO7f_Zy#scrollTo=vJxiCSLc7IWD




import argparse
import os,sys
#from Bio import SeqIO
parser = argparse.ArgumentParser()


#tell Jax and Tensorflow to use the same memory. This allows us to run larger structures
os.environ['TF_FORCE_UNIFIED_MEMORY'] = '1'  
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '2.0' 


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
parser.add_argument("--nstruct",type=int,help="Number of independent outputs to generate PER MODEL. Each starts with a different predetermined seed. Default=1", default=1)
parser.add_argument("--num_ensemble",type=int,default=1,help="number of times to process the input features and combine. default = 1. Deepmind used 8 for casp. Expert Option.")
parser.add_argument("--turbo",action="store_true",help="use the latest and greatest hacks to make it run fast fast fast.")
parser.add_argument("--max_recycles",type=int,default=3,help="max number of times to run evoformer. Default is 3. Single domain proteins need fewer runs. Multidomain or PPI may need more")
parser.add_argument("--recycle_tol",type=float,default=0.0,help="Stop recycling early if CA-RMSD difference between current output and previous is < recycle_tol. Default = 0.0 (no early stopping)")
parser.add_argument("--show_images",action="store_true")
parser.add_argument("--ptm",action="store_true",help="use the version of the models that output predicted TMalign score. Might be good for complexes")

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
os.makedirs(args.out_dir,exist_ok=True)

import pymol

#created syslink to silent_tools git repo
#sys.path.append("/home/rdkibler/scripts/silent_tools_git/")
import silent_tools

from dataclasses import dataclass

import itertools
import subprocess

def renumber(sele="all"):
        pdbstr = pymol.cmd.get_pdbstr(sele)
        process = subprocess.Popen(['perl',"/home/rdkibler/scripts/renum.pl"], stdout=subprocess.PIPE, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        fixed_pdbstr = process.communicate(input=pdbstr.encode())[0].decode('utf-8')
        pymol.cmd.delete(sele)
        pymol.cmd.read_pdbstr(fixed_pdbstr,sele)


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








# #parse files before loading heavy ML stuff
# query_sequences = []
# query_names = []
# query_have_structure
# for path in args.input_files:
#   if path.endswith(".gz"):
#     import gzip
#     f = gzip.open(path,'rt')
#   else:
#     f = open(path,'rt')

#   if path.replace(".gz","").lower().endswith(".fasta") or path.replace(".gz","").lower().endswith(".fa"):
#     seq = ""
#     name = ""
#     for line in f:
#       if line.startswith(">"):
#         if len(seq) > 0:
#           query_names.append(name)
#           query_sequences.append(seq.split("/"))
#         name = line[1:].strip()
#         seq = ""
#       else:
#         seq += line.strip()
#     if len(seq) > 0:
#       query_names.append(name)
#       query_sequences.append(seq.split("/"))
#   else:
#     query_names.append(path.split(".pdb")[0].split("/")[-1])
#     query_sequences.append([str(record.seq) for record in SeqIO.parse(f,'pdb-atom')])
#   f.close()


query_targets = []
for file in args.input_files:
  query_targets.extend(parse_file(file))


import numpy as np
import pickle
from typing import Dict

from matplotlib import pyplot as plt

plt.switch_backend('agg')

import sys
sys.path.append("/home/rdkibler/software/alphafold/")

from alphafold.model import model
from alphafold.model import config
from alphafold.model import data
from alphafold.common import protein

from alphafold.data import parsers
from alphafold.data import pipeline


import colabfold as cf
from collections import defaultdict
import tqdm
import jax
from jax.lib import xla_bridge
device = xla_bridge.get_backend().platform
print(device)


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
      query_sequence = target.seq.split("/")
      name = target.name

      pad_mask = np.array([c != "U" for c in full_sequence])
      
      #############################
      # define input features
      #############################

      num_res = len(full_sequence)
      feature_dict = {}
      msas = [[full_sequence]]
      deletion_matrices = [[[0]*len(full_sequence)]]
      feature_dict.update(pipeline.make_sequence_features(full_sequence, name, num_res))
      feature_dict.update(pipeline.make_msa_features(msas, deletion_matrices=deletion_matrices))



      Ls = [len(chain_seq) for chain_seq in query_sequence]
      Ls_plot = sum([[len(seq)] for seq in query_sequence],[])

      feature_dict['residue_index'] = cf.chain_break(feature_dict['residue_index'], Ls)




      #########################################




      ###########################
      # run alphafold
      ###########################
      def parse_results(prediction_result, processed_feature_dict):
        
        b_factors = prediction_result['plddt'][:,None] * prediction_result['structure_module']['final_atom_mask']  
        dist_bins = jax.numpy.append(0,prediction_result["distogram"]["bin_edges"])
        dist_mtx = dist_bins[prediction_result["distogram"]["logits"].argmax(-1)]
        contact_mtx = jax.nn.softmax(prediction_result["distogram"]["logits"])[:,:,dist_bins < 8].sum(-1)

        out = {"unrelaxed_protein": protein.from_prediction(processed_feature_dict, prediction_result, b_factors=b_factors),
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
          model_name = "model_5_ptm" if args.ptm else "model_5"
          N = len(feature_dict["msa"])
          L = len(feature_dict["residue_index"])
          compile_settings = (N, L, args.ptm, args.max_recycles, args.recycle_tol, args.num_ensemble, args.mock_msa_depth, args.enable_dropout, args.pct_seq_mask)

          recompile = prev_compile_settings != compile_settings

          if recompile:
            cf.clear_mem(device) ##is this ok?
            cfg = config.model_config(model_name)

            # set size of msa (to reduce memory requirements)

            cfg.data.eval.max_msa_clusters = args.mock_msa_depth
            cfg.data.common.max_extra_msa = args.mock_msa_depth
            cfg.data.eval.masked_msa_replace_fraction = args.pct_seq_mask
            cfg.model.recycle_tol = args.recycle_tol

            cfg.data.common.num_recycle = cfg.model.num_recycle = args.max_recycles
            
            cfg.data.eval.num_ensemble = args.num_ensemble

            params = data.get_model_haiku_params(model_name,data_dir="/projects/ml/alphafold")
            model_runner = model.RunModel(cfg, params, is_training=args.enable_dropout)
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

          print("############")
          print(list(o.keys()))
          print("############")
          


          output_line = f"{name} {key} recycles:{o['recycles']} tol:{o['tol']:.2f} mean_plddt:{o['mean_plddt']:.2f}"
          if args.ptm: output_line += f" pTMscore:{o['pTMscore']:.2f}"

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

          with open(fout_name, 'w') as f:
            f.write("\n".join(bfactored_pdb_lines))

          if target.pymol_obj_name is not None:
            pymol.cmd.read_pdbstr("\n".join(bfactored_pdb_lines),oname='temp_target')
            best_rmsd = np.inf
            #need to try all permutations of chain orderings
            chains = pymol.cmd.get_chains('temp_target')
            

            for new_chain_order in itertools.permutations(chains,len(chains)):
              print(new_chain_order)
              #make a copy
              pymol.cmd.create("rechained_temp_target","temp_target")

              #set new chain order
              for chain in chains:
                pymol.cmd.select(f"chain{chain}",f"rechained_temp_target and chain {chain}")

              for old_chain,new_chain in zip(chains, new_chain_order):
                pymol.cmd.alter(f"chain{old_chain}",f"chain='{new_chain}'")
              pymol.cmd.alter("rechained_temp_target","segi=''")

              renumber("rechained_temp_target")

              #measure rmsd
              rmsd = pymol.cmd.align(target.pymol_obj_name + " and n. CA",'temp_target and n. CA',cycles=0)[0]
              print(rmsd)
              best_rmsd = min(best_rmsd,rmsd)

              pymol.cmd.delete("rechained_temp_target")


            o['rmsd_to_design'] = best_rmsd
            pymol.cmd.delete('temp_target')
            output_line += f" rmsd_to_input:{best_rmsd:0.2f}"

          with open('reports.txt','a') as f:
            f.write(output_line+"\n")
          print(output_line)

          if args.show_images:
            fig = cf.plot_protein(o["unrelaxed_protein"], Ls=Ls_plot, dpi=200)
            plt.savefig(os.path.join(args.out_dir,f'{prefix}.png'), bbox_inches = 'tight')
            plt.close(fig)


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
              model_name = model_name+"_ptm" if args.ptm else model_name
              key = f"{model_name}_seed_{seed}"
              pbar2.set_description(f'Running {key}')

              # replace model parameters
              params = data.get_model_haiku_params(model_name, data_dir="/projects/ml/alphafold")
              for k in model_runner.params.keys():
                model_runner.params[k] = params[k]

              # predict
              prediction_result, (r, t) = cf.to(model_runner.predict(processed_feature_dict, random_seed=seed),device) #is this ok?

              

              # save results
              outs[key] = parse_results(prediction_result, processed_feature_dict)
              outs[key].update({"recycles":r, "tol":t})
              report(key)

              #hack! TODO: remove
              # with open('result.csv','a') as f:
              #   f.write(f"{name},{np.mean(outs[key]['plddt'][pad_mask])},{np.mean(outs[key]['pTMscore'])}\n")


              del prediction_result, params
            del processed_feature_dict

        else:  
          # go through each model
          for num, model_name in enumerate(model_names):
            model_name = model_name+"_ptm" if args.ptm else model_name
            params = data.get_model_haiku_params(model_name, data_dir="/projects/ml/alphafold")  
            cfg = config.model_config(model_name)
            cfg.data.common.num_recycle = cfg.model.num_recycle = args.max_recycles
            cfg.model.recycle_tol = args.recycle_tol
            cfg.data.eval.num_ensemble = args.num_ensemble
            model_runner = model.RunModel(cfg, params, is_training=args.enable_dropout)

            # go through each random_seed
            for seed in range(args.nstruct):
              key = f"{model_name}_seed_{seed}"
              pbar2.set_description(f'Running {key}')
              #print(feature_dict)
              processed_feature_dict = model_runner.process_features(feature_dict, random_seed=seed)
              prediction_result, (r, t) = cf.to(model_runner.predict(processed_feature_dict, random_seed=seed),device) #is this ok?

              outs[key] = parse_results(prediction_result, processed_feature_dict)
              outs[key].update({"recycles":r, "tol":t})
              report(key)

              # cleanup
              del processed_feature_dict, prediction_result

            del params, model_runner, cfg

            #turns out, no
            #cf.clear_mem(device) #is this ok?



      #   # Find the best model according to the mean pLDDT.
      #   model_rank = list(outs.keys())
      #   model_rank = [model_rank[i] for i in np.argsort([outs[x][rank_by] for x in model_rank])[::-1]]

      #   # Write out the prediction
      #   for n,key in enumerate(model_rank):
      #     prefix = f"{name}_rank_{n+1}_{key}" 
      #     pred_output_path = os.path.join(args.out_dir,f'{prefix}_unrelaxed.pdb')
      #     fig = cf.plot_protein(outs[key]["unrelaxed_protein"], Ls=Ls_plot, dpi=200)
      #     plt.savefig(os.path.join(args.out_dir,f'{prefix}.png'), bbox_inches = 'tight')
      #     plt.close(fig)

      #     pdb_lines = protein.to_pdb(outs[key]["unrelaxed_protein"])
      #     with open(pred_output_path, 'w') as f:
      #       f.write(pdb_lines)
            
      # ############################################################
      # print(f"model rank based on {rank_by}")
      # for n,key in enumerate(model_rank):
      #   print(f"rank_{n+1}_{key} {rank_by}:{outs[key][rank_by]:.2f}")

      pbar1.update(1)


exit()
##################################################################################################################
#old stuff below
##################################################################################################################










# # setup which models to use
# model_runners = {}
# if args.models[0] == "all":
#   models = ["model_1","model_2","model_3","model_4","model_5"]
# else:
#   models = [f"model_{model_num}" for model_num in args.models]

# if args.ptm:
#   models = [model + "_ptm" for model in models]

# for model_name in models:
#   model_name_for_config = model_name
#   if model_name in ["model_1","model_2"]:
#     model_name_for_config = "model_5"
#   model_config = config.model_config(model_name_for_config)

#   #sergey found recycling may only be necessary for "docking" multidomain proteins. We both hypothesized independently that this could also improve PPI predictions
#   model_config.model.num_recycle = args.num_recycle
#   model_config.data.common.num_recycle = args.num_recycle
#   model_config.data.common.max_extra_msa = args.mock_msa_depth # af2 crashes if you make this = 1 with a model that uses templates (1 or 2). If you config the models for ones built WITHOUT templates, it works. 
#   model_config.data.eval.max_msa_clusters = args.mock_msa_depth

#   model_config.data.eval.num_ensemble = 1
#   model_params = data.get_model_haiku_params(model_name=model_name, data_dir="/projects/ml/alphafold")
#   model_runner = model.RunModel(model_config, model_params)
#   model_runners[model_name] = model_runner


# def mk_mock_template(query_sequence):
#   # mock template features
#   output_templates_sequence = []
#   output_confidence_scores = []
#   templates_all_atom_positions = []
#   templates_all_atom_masks = []

#   for _ in query_sequence:
#     templates_all_atom_positions.append(np.zeros((templates.residue_constants.atom_type_num, 3)))
#     templates_all_atom_masks.append(np.zeros(templates.residue_constants.atom_type_num))
#     output_templates_sequence.append('-')
#     output_confidence_scores.append(-1)
#   output_templates_sequence = ''.join(output_templates_sequence)
#   templates_aatype = templates.residue_constants.sequence_to_onehot(output_templates_sequence,
#                                                                     templates.residue_constants.HHBLITS_AA_TO_ID)

#   template_features = {'template_all_atom_positions': np.array(templates_all_atom_positions)[None],
#         'template_all_atom_masks': np.array(templates_all_atom_masks)[None],
#         'template_sequence': [f'none'.encode()],
#         'template_aatype': np.array(templates_aatype)[None],
#         'template_confidence_scores': np.array(output_confidence_scores)[None],
#         'template_domain_names': [f'none'.encode()],
#         'template_release_date': [f'none'.encode()]}
        
#   return template_features

# #suggested by drhicks
# def set_bfactor(pdb_filename, bfac):
#   I = open(pdb_filename,"r").readlines()
#   O = open(pdb_filename,"w")
#   seq_id = -1
#   seq_id_then = ""
#   for line in I:
#     if line[0:6] == "ATOM  ":
#       seq_id_now = int(line[23:26].strip()) - 1
#       if seq_id_now != seq_id_then:
#         seq_id += 1
#       O.write("{prefix}{bfac:6.2f}{suffix}".format(prefix=line[:60], bfac=bfac[seq_id], suffix=line[66:]))
#       seq_id_then = int(line[23:26].strip()) - 1
#   O.close()

# def predict_structure(
#     prefix: str,
#     data_pipeline: pipeline.DataPipeline,
#     model_runners: Dict[str, model.RunModel],
#     Ls: list,
#     random_seed: int):
  
#   """Predicts structure using AlphaFold for the given sequence."""
#   # Get features.
#   feature_dict = data_pipeline.process()

#   # #stolen from krypton
#   # feature_dict = {
#   #   **pipeline.make_sequence_features(sequence=query_sequence,description="none",num_res=len(query_sequence)),
#   #   **pipeline.make_msa_features(msas=[[query_sequence]],deletion_matrices=[[[0]*len(query_sequence)]]),
#   # }

#   # add big enough number to residue index to indicate chain breaks
#   idx_res = feature_dict['residue_index']
#   L_prev = 0
#   # Ls: number of residues in each chain
#   for L_i in Ls[:-1]:
#       idx_res[L_prev+L_i:] += 200
#       L_prev += L_i
#   feature_dict['residue_index'] = idx_res
#   print (feature_dict['residue_index'])

#   # Run the models.
#   for model_name, model_runner in model_runners.items():
#     processed_feature_dict = model_runner.process_features(feature_dict, random_seed=random_seed)
#     prediction_result = model_runner.predict(processed_feature_dict)
#     unrelaxed_protein = protein.from_prediction(processed_feature_dict,prediction_result)
#     output_name = f'{prefix}_unrelaxed_{model_name}_seed{random_seed}_recycle{args.num_recycle}'
#     unrelaxed_pdb_path = f'{output_name}.pdb'

#     with open(unrelaxed_pdb_path, 'w') as f:
#       chain_alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
#       chain_num = 0
#       prev_resid = 1

#       #TODO:Insert some metadata about this run into the pdb file


#       #TODO: change this to make use of Ls for fixing chain
#       for line in protein.to_pdb(unrelaxed_protein).split("\n"):
#         if line[:4] == "ATOM":
#           resid = int(line[22:26])
#           resid_diff = resid - prev_resid
#           if resid_diff > 2:
#             chain_num += 1
#           new_line = line[:21] + alphabet[chain_num] + line[22:]
#           prev_resid = resid
#           f.write(new_line+"\n")
#         else:
#           f.write(line+"\n")

#     out_dict = {}

#     #todo: incorporate this above to modify the pdbstring in memory
#     set_bfactor(unrelaxed_pdb_path,prediction_result['plddt']/100)


#     out_dict['plddt'] = prediction_result['plddt']
#     if args.ptm:
#         out_dict['ptm'] = prediction_result['ptm']
#     if args.pae:
#         out_dict['predicted_aligned_error'] = prediction_result['predicted_aligned_error']

#     if args.save_all_prediction_results:
#       np.savez_compressed(f'{output_name}_all_prediction_results.npz',**prediction_result)
#     else:
#       np.savez_compressed(f'{output_name}_pred_metrics.npz',**out_dict)



# alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890"

# query_names = []
# query_sequences = []

# for (record_id,chains) in zip(names,fasta_sequences):
#   print(record_id,chains)

#   if args.single_only:
#     for chain_num,chain in enumerate(chains):
#       query_names.append(f"{record_id}_{alphabet[chain_num]}")
#       query_sequences.append(chain)

#   elif len(args.only_chains) > 0:
#     for chain_idx in args.only_chains:
#       chain = chains[chain_idx]
#       query_names.append(f"{record_id}_{alphabet[chain_idx]}")
#       query_sequences.append(chain)

#   elif args.complex_only:
#     query_sequences.append("/".join(chains))
#     query_names.append(f"{record_id}_complex")

#   else:
#     #run the complex first so we fail early if we run out of mem
#     query_sequences.append("/".join(chains))
#     query_names.append(f"{record_id}_complex")
#     for chain_num,chain in enumerate(chains):
#       query_names.append(f"{record_id}_{alphabet[chain_num]}")
#       query_sequences.append(chain)



# #sort query_sequences longest-to-shortest (to fail early) and group by length (we can re-use the model for sequences of the same length)
# for seq in query_sequences:
# 	print(len(seq),seq)


# from collections import defaultdict

# length_dict = defaultdict(lambda: [])
# for name,seq in zip(query_names,query_sequences):
# 	length_dict[len(seq)].append((name,seq))

# lengths = sorted(length_dict.keys(),reverse=True)

# for length in lengths:
#   print("now processing all inputs of length",length)
#   cohort = length_dict[length]

#   dummy_seq = "A"*length

#   #batching not quite yet implemented. Need to look at smarter ppl's code
#   for out_prefix,query_sequence in cohort:
#     print(out_prefix,query_sequence)
#     query_sequence = query_sequence.replace("/", " ")
#     query_sequence = query_sequence.split()
#     Ls = list()
#     for seq in query_sequence:
#         Ls.append(len(seq))
#     query_sequence = "".join(query_sequence)


#     # mock pipeline for testing
#     data_pipeline_mock = mock.Mock()
#     data_pipeline_mock.process.return_value = {
#         **pipeline.make_sequence_features(sequence=query_sequence,
#                                           description="none",
#                                           num_res=length),
#         **pipeline.make_msa_features(msas=[[query_sequence]],
#                                      deletion_matrices=[[[0]*length]]),
#         **mk_mock_template(query_sequence)
#     }



#     for random_seed in range(args.nstruct):

#       predict_structure(
#         prefix=out_prefix,
#         data_pipeline=data_pipeline_mock,
#         model_runners=model_runners,
#         Ls=Ls,
#         random_seed=random_seed)

# print('Done')

