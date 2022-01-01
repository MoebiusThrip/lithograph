# reactions.py to represent chemical reactions

# import numpy
import numpy

# import pprint for pretty printing
import pprint

# import h5py to read h5 files
import h5py


# class Reaction to represent data reactions
class Reaction(object):
    """Class reaction to store reaction attributes.

    Inherits from:
        None
    """

    def __init__(self):
        """Initialize a reaction instance.

        Arguments:
            None
        """

        return

    def __repr__(self):
        """Represent the reaction on screen.

        Arguments:
            None

        Returns:
            str
        """

        # make a string of the reaction's route
        name = self.name
        shape = self.shape
        slash = self.slash.replace(self.name, '')[-100:]
        representation = '< Reaction: {} {} {} >'.format(name, shape, slash)

        return representation
