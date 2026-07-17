# Dataset and Derived Artifacts

The code release and data release are separate. Do not commit the local `data/` tree,
ROS bag conversion backups, model checkpoints, or generated experiment results to Git.

## Original interaction data

The experiments use the RBO interaction dataset. Obtain and cite the original release
from Zenodo: <https://doi.org/10.5281/zenodo.1036660> (CC BY 4.0). This repository does
not redistribute the original bags. Converted ROS 2 bags remain derived copies of that
dataset and must retain its attribution and terms.

## Derived artifacts

The filter consumes one optical-flow field, one HAP heatmap, and/or one GraspNet output
per synchronized RGB-D timestamp:

```text
flow/<category>/<instance>/<timestamp>.npy
hap_output/<category>/<instance>/<timestamp>_heatmap.npy
GraspNet/<category>/<instance>/<timestamp>_unprocessed_output.npy
```

Generate permitted artifacts with `cpf-precompute`; record the exact code commit,
experiment/project config, model source commit, checkpoint hash, input dataset version,
and command. A release manifest must additionally list every file's relative path,
byte size, SHA-256, model identifier, sequence, and timestamp.

HAP output redistribution is not yet authorized: the upstream Hands as Probes repository
declares no license. Do not publish HAP source, checkpoints, or heatmaps until the
copyright holders confirm that the intended distribution is permitted. Contact-GraspNet
code/checkpoints and outputs remain subject to NVIDIA's non-commercial research and
evaluation terms; they are not relicensed by this repository's MIT license.

## Release exclusions and known gaps

- Exclude all `*.orig_ros2` conversion backups.
- Never include checkpoints, credentials, absolute machine paths, caches, or results.
- The local interaction index contains entries for `laptop26` and `rubikscube09`, but
  their converted bags were absent during the latest audit. Acquire them or document
  their exclusion before publishing a complete manifest.
- Publish large archives through an archival data service such as Zenodo and assign a
  version/DOI; GitHub source history is not the data distribution mechanism.
