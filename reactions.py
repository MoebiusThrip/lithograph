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

    def __init__(self, yam, forward, backward, energies):
        """Initialize a reaction instance.

        Arguments:
            yam: dict, yaml reaction entru
            forward: float, forward rate constant
            backward: float, backward rate constant
        """

        # name the reaction from the transition for now
        self.name = yam['transition']
        self.transition = yam['transition']

        # populate from yam
        self.electrophile = yam['electrophile']
        self.nucleophile = yam['nucleophile']
        self.reactant = yam['reactant']
        self.product = yam['product']
        self.leaver = yam['leaver']

        # add catalysis
        self.catalysis = yam['catalysis']

        # set energies
        self.energies = energies

        # populate rate constants
        self.forward = forward
        self.backward = backward

        # add plotting color
        self.color = yam['color']

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
        forward = round(self.forward, 4)
        backward = round(self.backward, 4)
        color = self.color
        energies = self.energies
        equation = '{} + {} = {} + {}'.format(self.nucleophile, self.reactant, self.product, self.leaver)
        formats = (equation, name, forward, backward, energies, color)
        representation = '< Reaction: {} [{}] -> {} <- {} {} ({}) >'.format(*formats)

        return representation
