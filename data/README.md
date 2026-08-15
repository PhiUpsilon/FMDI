# Data preparation

Raw data are excluded from this repository. The loaders intentionally preserve
the preprocessing and split protocol used by the CSDI/FSDI benchmark line.

## PhysioNet 2012

1. Obtain the PhysioNet/Computing in Cardiology Challenge 2012 `set-a` records
   from the official PhysioNet distribution and comply with its terms.
2. Place the patient text files at:

   ```text
   data/physio/set-a/132539.txt
   data/physio/set-a/132540.txt
   ...
   ```

3. On first use, `fmdi.data_physio` parses the text records and creates a local
   cache named `data/physio_missing<RATIO>_seed0.pk` for the public seed-0 example.

The public loader performs the same fold construction and artificial target
mask generation used for the paper experiments. Do not commit the generated
cache files.

## Air Quality (PM2.5)

Use the Air Quality files distributed with the STMVL/CSDI benchmark setup. The
expected layout is:

```text
data/pm25/pm25_meanstd.pk
data/pm25/Code/STMVL/SampleData/pm25_ground.txt
data/pm25/Code/STMVL/SampleData/pm25_missing.txt
```

`pm25_meanstd.pk` contains the training-set mean and standard deviation used by
the benchmark preprocessing. The loader uses the selected `valid-index` and
does not download or modify the source dataset.

## Data policy

Do not commit raw records, generated caches, credentials, or access-controlled
data. Dataset use remains governed by the original providers' licenses.
