import configparser


class Config:
    def __init__(self, config_file):
        """Initializes the Config class.

        This class assumes CFG format for data in config_file.
        """
        with open(config_file, 'r') as f:
            self.settings_text = f.read()

        self.settings = configparser.ConfigParser()
        self.settings.read(config_file)

        # Remove the reference to the DEFAULT section for ease of use
        # (i.e. direct access of config settings from self.settings)
        self.settings = self.settings['DEFAULT'] 
