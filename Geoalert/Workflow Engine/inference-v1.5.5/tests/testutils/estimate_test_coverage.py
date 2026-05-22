import sys

import yaml
import os
from typing import Callable
from collections import defaultdict
import argparse


def recursive_parse_config(conf: dict) -> list:
    """Parses YAML-like dict, returns all '_class' entities as a list"""
    result = []
    if 'bricks' in conf.keys():
        if not isinstance(conf['bricks'], list):
            print(f"Invalid 'bricks' value in config: {conf}", sys.stderr)
            return []
        for b in conf['bricks']:
            result.extend(recursive_parse_config(b))
    if 'blocks' in conf.keys():
        if not isinstance(conf['blocks'], list):
            print(f"Invalid 'blocks' value in config: {conf}", sys.stderr)
            return []
        for b in conf['blocks']:
            result.extend(recursive_parse_config(b))
    if 'postprocessors' in conf.keys():
        if not isinstance(conf['postprocessors'], list):
            print(f"Invalid 'postprocessors' value in config: {conf}", sys.stderr)
            return []
        for b in conf['postprocessors']:
            result.extend(recursive_parse_config(b))
    if 'adapter' in conf.keys():
        result.append(conf['adapter']['_class'])
    if '_class' in conf.keys():
        result.append(conf['_class'])
    return result


def recursive_get_paths(path: str, name_filter: Callable[[str], bool]) -> list:
    """Returns list of path to files in all subfolders that satisfy name_filter()"""
    res = []
    if not os.path.isdir(path):
        return []
    for name in os.listdir(path):
        full_path = os.path.join(path, name)
        if os.path.isdir(full_path):
            res.extend(recursive_get_paths(full_path, name_filter))
        if not os.path.isdir(full_path) and name_filter(name):
            res.append(full_path)
    return res


def parse_awesome_configs(path: str):
    # paths to directories with up-to-date configs
    configs_paths = (os.path.normpath(os.path.join(path, 'configs/alpha/')),
                     os.path.normpath(os.path.join(path, 'configs/beta/')),
                     os.path.normpath(os.path.join(path, 'configs/customer/')),
                     os.path.normpath(os.path.join(path, 'configs/default/')))
    bricks_per_pipeline = dict()  # Pipeline name: [bricks]
    bricks_set_in_configs = set()  # Full set of actual bricks in configs

    for configs_path in configs_paths:
        for path in recursive_get_paths(configs_path,
                                        name_filter=lambda x: x.startswith('inference') and x.endswith('.yml')):
            with open(path) as f:
                conf = yaml.safe_load(f)
            if 'config' in conf.keys():
                key = '\\'.join(path.split('\\')[max(-len(path.split('\\')), -4):-1])
                parse_res = recursive_parse_config(conf['config'])
                bricks_per_pipeline[key] = set(parse_res)
                for name in parse_res:
                    bricks_set_in_configs.add(name)

    return bricks_set_in_configs


def parse_test_configs():
    configs_path = 'tests/test_pipelines/mock_models_tests'
    bricks_per_pipeline = dict()  # Pipeline name: [bricks]
    bricks_set_in_configs = set()  # Full set of actual bricks in configs

    for path in recursive_get_paths(configs_path,
                                    name_filter=lambda x: x.endswith('.yml')):
        with open(path) as f:
            conf = yaml.safe_load(f)
        if 'config' in conf.keys():
            key = '\\'.join(path.split('\\')[max(-len(path.split('\\')), -4):-1])
            parse_res = recursive_parse_config(conf['config'])
            bricks_per_pipeline[key] = set(parse_res)
            for name in parse_res:
                bricks_set_in_configs.add(name)

    return bricks_set_in_configs


def parse_urbanlib():
    urbanlib_path = 'modules/urban/urban/bricks'
    bricks_set_in_urbanlib = defaultdict(set)  # parent: set of children

    for p in recursive_get_paths(urbanlib_path, lambda x: not (x.startswith('_')) and x.endswith('.py')):
        with open(p, encoding="utf8") as f:
            for line in f.readlines():
                if line.startswith('class ') and line.endswith(':\n'):  # naive, but works good enough in our case
                    if '(' not in line:
                        child = line[6:-2]
                        parent = None
                    else:
                        child = line[6:].split('(')[0]
                        parent = line[6:].split('(')[1].split(')')[0]
                    if parent:
                        bricks_set_in_urbanlib[parent].add(child)
                    if child not in bricks_set_in_urbanlib.keys():
                        bricks_set_in_urbanlib[child] = set()
    return bricks_set_in_urbanlib


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', help='path to awesome configs')
    args = parser.parse_args()
    bricks_set_in_urbanlib = parse_urbanlib()
    print("Total unique bricks in urbanlib: ", len(bricks_set_in_urbanlib.keys()))
    bricks_without_children = set([k for k in bricks_set_in_urbanlib.keys() if not bricks_set_in_urbanlib[k]])
    print("Bricks without inherited bricks: ", len(bricks_without_children))

    bricks_set_in_test_configs = parse_test_configs()
    print("Tested bricks: ", len(bricks_set_in_test_configs))
    print("Test coverage: ",
          len(bricks_set_in_test_configs) / len(bricks_set_in_urbanlib.keys()) * 100,
          '%')
    print('Untested bricks: ', set(bricks_set_in_urbanlib.keys()).difference(bricks_set_in_test_configs))

    if args.path:
        bricks_set_in_awesome_configs = parse_awesome_configs(args.path)
        print("Bricks in awesome-configs: ", len(bricks_set_in_awesome_configs))
        deprecated_bricks = bricks_without_children.difference(bricks_set_in_awesome_configs)
        non_deprecated_bricks = set(bricks_set_in_urbanlib.keys()).difference(deprecated_bricks)
        print('Deprecated bricks: ', deprecated_bricks)
        print('Untested non-deprecated bricks: ',
              set(bricks_set_in_urbanlib.keys()).difference(bricks_set_in_test_configs).difference(deprecated_bricks))

