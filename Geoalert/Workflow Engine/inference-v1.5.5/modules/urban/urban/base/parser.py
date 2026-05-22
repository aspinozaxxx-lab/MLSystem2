import re
import yaml


class ComposeLoader:
    def __init__(self, pipeline_params=None):
        if pipeline_params is None:
            pipeline_params = dict()

        def constructor_variables(loader, node):
            """
            Extracts the variable from the node's value
            :param yaml.Loader loader: the yaml loader
            :param node: the current node in the yaml
            :return: the parsed string that contains the value 
            variable
            """
            # 'box_threshold_define: &box_threshold ${box_threshold:-10.0}'
            value = loader.construct_scalar(node)
            match_ = pattern.findall(value)  # to find all variables in line
            assert len(match_) <= 1, f'Only one variable is allowed in line: {value}'

            if len(match_) == 1:
                value = match_[0]
                var_name, default_value = value.split(':')

                # get value from pipeline_params or default
                value = pipeline_params.get(var_name, default_value)

                # try to convert value to float
                try:
                    value = float(value)
                except ValueError:
                    pass

            return value

        # the tag will be used to mark where to start searching for the pattern
        # e.g.: 'box_threshold_define: &box_threshold ${box_threshold:-10.0}'

        tag = "!variable"
        # pattern for global vars: look for '&variable_name ${variable_name:variable_default_value}'
        pattern = re.compile('.*?\$\{(\w+:.+)\}.*?')

        self.loader = yaml.SafeLoader
        # don't remove "$", it's necessary to prevent slowing down the passing
        self.loader.add_implicit_resolver(tag, pattern, "$")
        self.loader.add_constructor(tag, constructor_variables)


def parse_config(path=None, data=None, pipeline_params=None):
    """
    Load a yaml configuration file and resolve any variables
    The variables must be in this format:
    '&variable_name ${variable_name:variable_default_value}'.
    E.g.:
    database:
        box_threshold_define: &box_threshold ${box_threshold:0.76}
        text_prompt_define: &text_prompt ${text_prompt:red roof}

    :param str path: the path to the yaml file
    :param str data: the yaml data itself as a stream
    :param dict pipeline_params: the new parameters
    :return: the dict configuration
    :rtype: dict[str, T]
    """

    if path:
        with open(path) as conf_data:
            res = yaml.load(conf_data, Loader=ComposeLoader(pipeline_params=pipeline_params).loader)
    elif data:
        res = yaml.load(data, Loader=ComposeLoader(pipeline_params=pipeline_params).loader)
    else:
        raise ValueError('Either a path or data should be defined as input')

    def recursive_substitute(d):
        if isinstance(d, list):
            for idx in range(len(d)):
                recursive_substitute(d[idx])
        elif isinstance(d, dict):
            for k in list(d.keys()):
                recursive_substitute(d[k])
                if k == '_class':
                    d['brick_class'] = d.pop('_class')

    recursive_substitute(res)
    return res

