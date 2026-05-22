# Awesome Configs

This repo is a source of truth for all the workflow definitions available at Mapflow API 
Its purpose is to store and track versioning of workflow definitions

It is possible, but strongly discouraged, to write configs elsewhere.
Use this repo to update the configs and then immediately upload it via mapflow admin.

## Repository structure

__innopolis__ folder contains workflow definitions for INNOPOLIS models
__geoalert__ folder contains defaul workflow definitions for geoalert models

You can add folders and configs at will.

### Scripts
In `./scripts` directory are collected scripts that are used for preprocessing and upload of the configs. 
More on scripts usage below.

## Config composition
Every config consists of 3 instances:
- a __Workflow Definition__, which is used by the workflow-engine. It includes data acquisition and storage, 
merging the splits of results, vector and raster tile serving.
- an inference __Pipeline__, which is used by the inference module. It includes thematic image preprocessing,
neural model settings, vector postprocessing - basically, all the problem-specific processing steps.
 There can be multiple `pipeline` files, with corresponsing placeholder in `wd.yml` file
- a __Requirements__ file, which represents data requirements for this workflow

Before upload to the platform, `requirements.yml` and `inference.yml` are packed as base64 string 
and placed inside the dedicated places inside the WD, which afterwards is saved as `wd_base64.yml` (ignored by git)

## Installation

Install the requirements
```bash
pip3 install pyyaml==6.0
```

## Packing and unpacking WDs
This repo has script `pack.py` which can pack/unpack the pipelines.
Usage:
for packing the pipeline inside a WD specify all 4 files (input pipeline, input wd, out wd, requirements).
Alternatively, you can specify a folder that contains files `wd.yml`, `requirements.yml` and `pipeline.yml` which will be used as input, 
and optional output file.
__requirements__ file is not needed if there is no `validate_source` stage, 
but this variant is deprecated (all new configs should contain this stage)
If output file is not specified, the result is stored in the same folder with name `wd_base64.yaml`

```bash 
python3 -m scripts.pack <input_pipeline_name> <input_wd_name> <output_wd_name> <requirements_file_name>
python3 -m scripts.pack <pipeline_wd_folder> [out_file]
```

If you want to pack all the WDs inside a folder recursively, use `-r` switch and root folder name. For example, command
```bash 
python3 -m scripts.pack -r .
```
will pack all workflows (folders with required files inside) found recursively in `./config` folder
By deafult, it will __ignore__ folder named `archive`, use `-a`switch to pack archive also

## Uploading the workflows to the platform

Use Mapflow-Admin (web UI) `Workflows -> Create workflow` or `Edit` menu to upload\edit the workflow definition to AI platform.
Admin account is required.