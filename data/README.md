# Data preparation

Raw datasets are intentionally excluded from version control. Put downloaded
files in `data/raw/<dataset>/` and write processed artifacts to
`data/processed/<dataset>/`.

For each benchmark, document here before release:

- source URL, version, and license;
- exact preprocessing and split procedure;
- required access agreement or checksum;
- mapping from the dataset name to a configuration in `configs/`.

Do not commit restricted, personally identifiable, or license-incompatible data.
