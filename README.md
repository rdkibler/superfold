# SuperFold

SuperFold is a derivative of [AlphaFold2](https://github.com/deepmind/alphafold), AlphaFold-Multimer, and [ColabFold](https://github.com/sokrypton/ColabFold) with some novel improvements. 
It is intended only for running single-sequence predictions very quickly and takes advantage of some time and memory saving hacks
found by [krypton](https://github.com/sokrypton) and others in the community. This package is also intended only for use by IPD labs and was 
written with our computing resources (digs, janelia, perlmutter, hyak [todo]) in mind. 

Any publication that discloses findings arising from using this source code should contact the contributors for 
citation or authorhip agreements, as some relevant code (for instance, the `initial_guess` functionality) may 
need to be published beforehand or concurrently since they constitute novel work.



## Usage

TODO

## Installation

1) Download this git repo `$git clone git@github.com:rdkibler/superfold.git`
2) `$cd superfold`
3) [Download the alphafold weights](#model-parameters) or find an existing path to the weights
4) `$realpath /path/to/alphafold_weights/ > alphafold_weights.pth`. "params/" should be a child dir of the alphafold_weights dir
5) (optional, if you don't want to install pyrosetta) `$conda config --add channels https://username:password@conda.graylab.jhu.edu`
use the username and password provided by Comotion when you licensed it. 
This is technically optional because we don't currently use pyrosetta because
of silent_tools, so you could remove the pyrosetta line from the .yml and be fine
6) `$conda create --name pyroml --file pyroml.yml`
7) `$source activate ~/.conda/envs/pyroml`
8) `$pip install absl-py dm-tree tensorflow ml-collections`
9) `$pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_releases.html`
10) `$pip install dm-haiku 

### Model parameters

While the AlphaFold code is licensed under the Apache 2.0 License, the AlphaFold
parameters are made available for non-commercial use only under the terms of the
CC BY-NC 4.0 license. Please see the [Disclaimer](#license-and-disclaimer) below
for more detail.

The AlphaFold parameters are available from
https://storage.googleapis.com/alphafold/alphafold_params_2021-10-27.tar, and
are downloaded as part of the `scripts/download_all_data.sh` script. This script
will download parameters for:

*   5 models which were used during CASP14, and were extensively validated for
    structure prediction quality (see Jumper et al. 2021, Suppl. Methods 1.12
    for details).
*   5 pTM models, which were fine-tuned to produce pTM (predicted TM-score) and
    (PAE) predicted aligned error values alongside their structure predictions
    (see Jumper et al. 2021, Suppl. Methods 1.9.7 for details).
*   5 AlphaFold-Multimer models that produce pTM and PAE values alongside their
    structure predictions.

Place a path to the directory containing the model parameters in a file called
`alphafold_weights.pth`

### SuperFold output

The outputs will be saved in the directory provided via the `--output_dir` flag 
(defaults to `output/`). The outputs include the unrelaxed structure, the relaxed structure
if the `--amber_relax` flag is used, a `reports.txt` metadata summary for all predictions if the `--summarize` flag is used
, and prediction metadata for each prediction in individual .json files.

The contents of each output file are as follows:

*   `*_unrelaxed.pdb` – A PDB format text file containing the predicted
    structure with chain IDs rearranged to best match that of the input
    PDB file, if provided. The b-factor column contains the per-residue
    pLDDT scores ranging from `0` to `100`, where `100` means most 
    confident. Note that, because low is better for real experimental 
    B-factor, care must be taken when running applications that interpret
    the B-factor column, such as molecular replacement.
*   `*_relaxed*.pdb` – A PDB format text file containing the predicted
    structure, after performing an Amber relaxation procedure on the unrelaxed
    structure prediction (see Jumper et al. 2021, Suppl. Methods 1.8.6 for
    details). The chain IDs are rearranged and the b-factor column is filled
    like for the `*_unrelaxed.pdb` files.
*   `*_prediction_results.json` – A JSON format text file containing the times taken to run
    each section of the AlphaFold pipeline.
    *   Mean pLDDT scores in `mean_plddt` serve as an overall per-target monomer 
        confidence score and is the average over all per-residue plddts per target. 
        The range of possible values is from `0` to `100`, where `100`
        means most confident). 
    *   Present only if using pTM models: predicted TMalign-score (`pTMscore` field
        contains a scalar). As a predictor of a global superposition metric,
        this score is designed to also assess whether the model is confident in
        the overall domain packing.
    *   Present only if using the `--output_pae` flag and using pTM models: 
        predicted pairwise aligned errors. `pae` contains a NumPy array of 
        shape [N_res, N_res] with the range of possible values from `0` to
        `max_predicted_aligned_error`, where `0` means most confident). This can
        serve for a visualisation of domain packing confidence within the
        structure.
    *   Present only if using pTM models: predicted pairwise aligned error summaries.
        *   `mean_pae`: the mean of the entire [N_res, N_res] pae matrix
        *   `pae_intrachain_X`: The mean of the pae matrix between residues corresponding to chain X
        *   `pae_interchain_XY`: The mean of the pae matrix between residues corresponding to chain X going to residues corresponding to chain Y and vise versa

## Acknowledgements
### Code contributors

*   Ryan Kibler
TODO finish contributors lol

### 3rd party libraries and packages
SuperFold communicates with and/or references the following separate libraries
and packages:

*   [Abseil](https://github.com/abseil/abseil-py)
*   [Biopython](https://biopython.org)
*   [Chex](https://github.com/deepmind/chex)
*   [Haiku](https://github.com/deepmind/dm-haiku)
*   [Immutabledict](https://github.com/corenting/immutabledict)
*   [JAX](https://github.com/google/jax/)
*   [matplotlib](https://matplotlib.org/)
*   [ML Collections](https://github.com/google/ml_collections)
*   [NumPy](https://numpy.org)
*   [OpenMM](https://github.com/openmm/openmm)
*   [OpenStructure](https://openstructure.org)
*   [pandas](https://pandas.pydata.org/)
*   [pymol3d](https://github.com/avirshup/py3dmol)
*   [SciPy](https://scipy.org)
*   [Sonnet](https://github.com/deepmind/sonnet)
*   [TensorFlow](https://github.com/tensorflow/tensorflow)
*   [Tree](https://github.com/deepmind/tree)
*   [tqdm](https://github.com/tqdm/tqdm)

TODO: are there some that need to be added/removed?

We thank all their contributors and maintainers!

## License and Disclaimer

This is not an officially supported Google product.

Copyright 2021 DeepMind Technologies Limited.

### AlphaFold Code License

Licensed under the Apache License, Version 2.0 (the "License"); you may not use
this file except in compliance with the License. You may obtain a copy of the
License at https://www.apache.org/licenses/LICENSE-2.0.

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

### Model Parameters License

The AlphaFold parameters are made available for non-commercial use only, under
the terms of the Creative Commons Attribution-NonCommercial 4.0 International
(CC BY-NC 4.0) license. You can find details at:
https://creativecommons.org/licenses/by-nc/4.0/legalcode

### Third-party software

Use of the third-party software, libraries or code referred to in the
[Acknowledgements](#acknowledgements) section above may be governed by separate
terms and conditions or license provisions. Your use of the third-party
software, libraries or code is subject to any such terms and you should check
that you can comply with any applicable restrictions or terms and conditions
before use.
