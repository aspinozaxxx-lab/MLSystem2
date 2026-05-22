import sys
import yaml
import base64
import logging
import argparse
from pathlib import Path
from typing import Mapping, Optional
logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger('wd_updater')


PIPELINE_KEY = 'pipeline'


def get_filenames_from_wd(wd_file: Path):
    """
    find keys (requirements and inference pipeline files) inside the WD file
    """
    with open(wd_file, encoding='utf-8') as src:
        wd = yaml.safe_load(src.read())

    requirements_file = None
    pipelines = {}
    if 'validate-source' in wd['stages'].keys():
        requirements_file = wd['stages']['validate-source']['config']['params']['requirements']
        requirements_file = wd_file.with_name(requirements_file)

    if 'inference' in wd['stages'].keys():
        inference_params = wd['stages']['inference']['config']['params']
        pipelines = {key: wd_file.with_name(name) for key, name in inference_params.items() if key.startswith('pipeline')}

    return requirements_file, pipelines


def encode_file(filename: Path):
    with open(filename, 'rb') as src:
        encoded = base64.b64encode(src.read())
        encoded_str = encoded.decode('utf-8')
    return encoded_str


def make_base64_wd(pipeline_files: Mapping[str, Path],
                   wd_file: Path,
                   out_file: Path,
                   requirements_file: Optional[Path] = None):
    wd = yaml.safe_load(open(wd_file))

    if 'inference' in wd['stages'].keys():
        with open(wd_file, encoding='utf-8') as src:
            wd = yaml.safe_load(src.read())
        encoded_pipelines = {key: encode_file(filepath) for key, filepath in pipeline_files.items()}
        wd['stages']['inference']['config']['params'].update(encoded_pipelines)

    if 'validate-source' in wd['stages'].keys():
        if not requirements_file:
            print('There is requirements section. If you want to fill it, select requirements file!')
        else:
            encoded_requirements = encode_file(requirements_file)
            wd['stages']['validate-source']['config']['params']['requirements'] = encoded_requirements

    with open(out_file, 'w') as dst:
        dst.write(yaml.safe_dump(wd))


def is_valid_config(folder):
    if folder/'wd.yml' in list(folder.iterdir()):
        wd = folder/'wd.yml'
    elif folder/'wd.yaml' in list(folder.iterdir()):
        wd = folder/'wd.yaml'
    else:
        logger.debug(f"Config in {folder} does not contain wd")
        return None

    requirements_file, inference_files = get_filenames_from_wd(wd)

    if requirements_file is not None and not requirements_file.exists():
        logger.info(f"Config in {folder} does not contain requirements")
        return None

    for inference_file in inference_files.values():
        if not inference_file.exists():
            logger.info(f"Config in {folder} does not contain inference {inference_file.name}")
            return None
    logger.info(f"Found valid config in {folder}")
    return wd, inference_files, requirements_file


def find_all_valid_configs(folder, ignore_archive):
    result = []
    current = is_valid_config(folder)
    if current:
        result.append(current)
    for entry in folder.iterdir():
        is_archive = entry.name in 'archive'
        in_ignore_list = entry.name.startswith ('.') or entry.name == 'tmp'
        if entry.is_dir() and not (is_archive and ignore_archive) and not in_ignore_list:
            result += find_all_valid_configs(entry, ignore_archive)
    return result


def main(args):
    logger.setLevel("DEBUG" if args.v else "INFO")
    if args.recursive:
        folders = find_all_valid_configs(Path(args.input), ignore_archive=not args.add_archive)
        logger.info(f'Found {len(folders)} configs')
        for folder in folders:
            wd, inference, requirements = folder
            out_file = wd.with_name('wd_base64.yml')
            if args.v:
                logger.info(f'Packing {str(wd.parent)}')
            try:
                make_base64_wd(inference, wd, out_file, requirements)
            except Exception as e:
                logger.info(f'Error while packing folder {str(wd.parent)}')
        logger.info('Done')
        exit()
    if not args.input:
        logger.info('Input is not specified!')
        exit()
    elif Path(args.input).is_dir():
        files = is_valid_config(Path(args.input))
        print(files)
        if not files:
            logger.info('Input folder does not contain a valid config!')
            exit()
        wd, inference, requirements = files
        if args.v:
            print(f'Packing pipeline in {args.input}')
        make_base64_wd(inference,
                       wd,
                       wd.with_name('wd_base64.yml'),
                       requirements)
        logger.info('Done')
        exit()
    else:
        requirements_file, pipelines = get_filenames_from_wd(Path(args.input))
        if args.output:
            output = Path(args.output)
        else:
            output = Path(args.input).with_name('wd_base64.yml')

        logger.info(f'Packing {args.input} {pipelines} {requirements_file} to {args.output}')
        make_base64_wd(pipeline_files=pipelines,
                       wd_file=Path(args.input),
                       out_file=output,
                       requirements_file=requirements_file)
        logger.info('Done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='pack pipeline',
        description='Insert requirements and inference pipeline into workflow definition'
    )
    parser.add_argument('input', help='WD name or folder name')
    parser.add_argument('-r', '--recursive', action='store_true')
    parser.add_argument('-a', '--add_archive', action='store_true')
    parser.add_argument('-v', action='store_true')
    parser.add_argument('-o', '--output', action='store', default=None)
    args = parser.parse_args()
    main(args)
