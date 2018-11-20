class Log:
    def __init__(self, config, seed, overwrite=False):
        """Initializes the Log class.
        
        Where config is a Config object and overwrite determines if the file will be
        appended to or overwritten.
        """

        def write_config_params():
            """Writes all config file parameters to file."""

            self.write(self.config.settings_text[self.config.settings_text.find('#'):])
            self.write('###################################\n# Seed Value\n###################################')
            self.write('seed = ' + str(self.seed.val))
            self.write()            


        self.config = config

        self.file = open(self.config.settings['log file path'], 'w' if overwrite else 'a')

        self.seed = seed

        write_config_params()
        self.write('Result Log')

    
    def write(self, write_string=''):
        """Writes the contents of write_string to file."""
        self.file.write(write_string + '\n')


    def write_run_header(self, run_count):
        """Writes the given run count to file and to the screen."""
        run_header = '\nRun %i' % (run_count)
        self.write(run_header)
        print(run_header)


    def write_run_data(self, eval_count, average_score, highest_score):
        """Writes the given run data to file and to the screen."""
        run_data = str(eval_count) + '\t' + str(average_score) + '\t' + str(highest_score)
        self.write(run_data)
        print(run_data)
