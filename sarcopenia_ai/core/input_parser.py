import configparser
import os
from argparse import ArgumentParser

"""Example

# Reads configs from a config file using configparser. Use -c or --config to specify path to file. 
# Arguments are overridden based on order of priority:
# defaults < config file < command line arguments

from input_parser import InputParser


def main(args):
    print(args)

if __name__ == '__main__':
    parser = InputParser()

    parser.add_argument('sigma', type=float, default=2.5, help='sigma value')

    args = parser.parse()

    main(args)

"""


class InputParser(ArgumentParser):

    def parse_config_file(self):
        conf_parser = ArgumentParser(
            # Turn off help, so we print all options in response to -h
            add_help=False)
        conf_parser.add_argument("-c", "--config",
                                 help="Specify config file", metavar="FILE")
        cargs, remaining_argv = conf_parser.parse_known_args()
        # read from config file
        if cargs.config:
            config = configparser.ConfigParser()
            if os.path.exists(cargs.config):
                config.read([cargs.config])
            else:
                print('Config file does not exists at %s' % cargs.config)
                exit(1)

            # disregard section info
            str_args = []
            for section_name in config:
                for name, value in config.items(section_name):
                    str_args += ['--' + name] + value.split()

            cargs = self.parse_args(str_args)
            self.set_defaults(**vars(cargs))

        return remaining_argv

    def parse(self, config=True, args=None, namespace=None):
        if config:
            remaining_args = self.parse_config_file()
            parsed_args = self.parse_args(remaining_args)
        else:
            parsed_args = self.parse_args(args, namespace)
        return parsed_args

