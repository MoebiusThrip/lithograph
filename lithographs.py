# lithographs.py for the Lithograph class to make OMI Rawflux hdf5 files

# import general tools
import sys
import re
import ast

# import yaml for reading configuration file
import yaml

# import time and datetime
import datetime
import calendar

# import math functions
import math
import numpy

# import cores
from cores import Core


# class Lithograph to do OMI data reduction
class Lithograph(Core):
    """Lithograph class to generate chemical reaction sequences.

    Inherits from:
        Core
    """

    def __init__(self, transitions):
        """Initialize a Lithograph instance.

        Arguments:
            transitions: str, file for transition list

        Returns:
            None
        """

        # generate the configuration from transitions file
        self.transitions = transitions

        return

    def __repr__(self):
        """Create string for on screen representation.

        Arguments:
            None

        Returns:
            str
        """

        # display contents
        self._tell(self.paths)

        # create representation
        representation = ' < Lithograph instance at: {} -> {} >'.format(self.source, self.sink)

        return representation

    def _acquire(self, path):
        """Load in a yaml file.

        Arguments:
            path: str, filepath

        Returns:
            dict

        """
        # open yaml file
        with open(path, 'r') as pointer:

            # and read contents
            information = yaml.safe_load(pointer)

        return information

    def _dispense(self, information, destination):
        """Load in a yaml file.

        Arguments:
            information: dict
            destination: str, filepath

        Returns:
            None
        """

        # open yaml file
        with open(destination, 'w') as pointer:

            # try to
            try:

                # dump contents, sorted
                yaml.dump(information, pointer, sort_keys=False)

            # unless not an option (python 2)
            except TypeError:

                # just dump unsorted
                yaml.dump(information, pointer)

        return None

    def _disperse(self, name):
        """Add spacing to yaml files.

        Arguments:
            name: file name

        Returns:
            None
        """

        # improve spacing
        transcription = self._know(name)
        lines = []
        for line in transcription:

            # if it begins with a -
            tab = '    '
            if line.strip().startswith('-'):

                # add extra line
                lines.append('')

            # if it doesn't start with a space
            if len(line) > 0 and line[0].isalpha():

                # add extra line
                lines.append('')
                tab = ''

            # add line, swapping double quotes for sinlges
            lines.append(tab + line.replace("'", '').replace('"', "'"))

        # record as text file
        self._jot(lines, name)

        return None

    def _orient(self, fragment, fragmentii=' '):
        """Orient fragments and combine into standard species.

        Arguments:
            fragment: str
            fragmentii: str

        Returns:
            None
        """

        # reverse second fragment if upper is first
        if fragmentii[0].isupper():

            # reverse
            tokens = list(fragmentii)
            tokens.reverse()
            fragmentii = ''.join(tokens).strip()

        return None

    def _parse(self, transition):
        """Parse a transition state into reactants and products.

        Arguments:
            transition: str

        Returns:
            None
        """

        # get fragments
        fragments = transition.split('-')

        # combine fragments
        nucleophile = self._orient(fragments[0])
        electrophile = self._orient(fragments[1], fragments[2])
        product = self._orient(fragments[0], fragments[1])
        leaver = self._orient(fragments[2])

        return nucleophile, electrophile, product, leaver

    def _generate(self):
        """Generate the reaction information from list of transition states.

        Arguments:
            None

        Returns:
            None
        """

        # get information from transitions file
        transitions = self._know(self.transitions)

        # begin configuration
        configuration = {'reactions': [], 'species': [], 'bonds': [], 'repulsions': []}

        # for each transition
        species = []
        bonds = []
        repulsions = []
        for transition in transitions:

            # add all species
            fragments = transition.split('-')
            reactant = fragments[0]
            reactantii = fragments[1] + fragments[2]
            product = fragments[0] + fragments[1]
            productii = fragments[2]

            # add to species
            species += [reactant, reactantii, product, productii]

            # make reaction
            reaction = {'transition': transition, 'reactant': reactant, 'reactantii': reactantii}
            reaction.update({'product': product, 'productii': productii, 'catalysis': 0})
            configuration['reactions'].append(reaction)

        # remove duplicates and sort
        species = list(set(species))

        # add each entry
        for chemical in species:

            # make entry
            configuration['species'].append({'chemical': chemical, 'quantity': 0})

            # get all bonds and repulsions
            pairs = zip(chemical[:-1], chemical[1:])
            bonds += [''.join(pair) for pair in pairs]

            # get all bonds and repulsions
            triplets = zip(chemical[:-2], chemical[1:-1], chemical[2:])
            repulsions += [''.join([triplet[0], triplet[2]]) for triplet in triplets]

        # add each bond
        bonds = list(set(bonds))
        for bond in bonds:

            # make entry
            configuration['bonds'].append({'bond': bond, 'energy': 0})

        # add each repulsion
        repulsions = list(set(repulsions))
        for repulsion in repulsions:

            # make entry
            configuration['repulsions'].append({'repulsion': repulsion, 'energy': 0})

        # dump into yaml and format
        destination = self.transitions.replace('.txt', '.yaml')
        self._dispense(configuration, destination)
        self._disperse(destination)

        return None