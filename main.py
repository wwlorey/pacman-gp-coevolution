#!/usr/bin/env python3

import gp.gp_driver as gp_driver_class
import util.args as args_class
import util.config as config_class

if __name__ == '__main__':

    # Process command line arguments
    args = args_class.Arguments(1, ['config/default.cfg'])
    config_file = args.get_args()[0]


    # Setup configuration
    config = config_class.Config(config_file)


    # Initialize the GP driver and its run variables
    gp_driver = gp_driver_class.GPDriver(config)


    # Run the GP
    while gp_driver.run_count <= int(config.settings['num experiment runs']):
        gp_driver.begin_run()

        gp_driver.evaluate('init')

        while gp_driver.decide_termination():
            if config.settings.getboolean('control bloat'):
                gp_driver.control_bloat()

            gp_driver.select_parents()

            gp_driver.recombine()

            gp_driver.mutate()

            gp_driver.evaluate()

            gp_driver.select_for_survival()

        gp_driver.end_run()
