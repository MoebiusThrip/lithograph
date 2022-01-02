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

    def _flip(self, chemical):
        """Flip chemical into standard orientation.

        Arguments:
            chemical: str

        Returns:
            str
        """

        # pair with reversal
        pair = [chemical, chemical[::-1]]

        # sort alphabetically
        pair.sort()

        # sort by nucleophile in beginnig
        pair.sort(key=lambda entry: entry[0].isupper(), reverse=True)

        # choose first
        choice = pair[0]

        return choice

    def _parse(self, transition):
        """Parse a transition state into reactants and products.

        Arguments:
            transition: str

        Returns:
            None
        """

        # split into nucleophile, electrophile, and leaving group
        nucleophile, electrophile, leaver = transition.split('-')

        # combine electrophile and nucleophile into product
        def orienting(fragment): return fragment if fragment[0].islower() else fragment[::-1]
        product = nucleophile + orienting(electrophile)

        # combine electrophile and leaving group into reactant
        def orienting(fragment): return fragment if fragment[-1].islower() else fragment[::-1]
        reactant = orienting(electrophile) + leaver

        # flip all members
        chemicals = [self._flip(chemical) for chemical in (nucleophile, reactant, product, leaver)]
        nucleophile, reactant, product, leaver = chemicals

        return nucleophile, reactant, product, leaver

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

            # parse into species and add
            chemicals = self._parse(transition)
            species += chemicals

            # unpack
            nucleophile, reactant, product, leaver = chemicals

            # make reaction
            reaction = {'transition': transition, 'nucleophile': nucleophile, 'reactant': reactant}
            reaction.update({'product': product, 'leaver': leaver, 'catalysis': 0, 'color': 'black'})
            configuration['reactions'].append(reaction)

        # remove duplicates and sort
        species = list(set(species))
        species.sort()

        # add each entry
        for chemical in species:

            # make entry
            configuration['species'].append({'chemical': chemical, 'quantity': 0})

            # get all bonds
            pairs = zip(chemical[:-1], chemical[1:])
            bonds += [self._flip(''.join(pair)) for pair in pairs]

            # get all repulsions
            triplets = zip(chemical[:-2], chemical[1:-1], chemical[2:])
            repulsions += [self._flip(''.join([triplet[0], triplet[2]])) for triplet in triplets]

        # add each bond
        bonds = list(set(bonds))
        bonds.sort()
        for bond in bonds:

            # make entry
            configuration['bonds'].append({'bond': bond, 'energy': 0})

        # add each repulsion
        repulsions = list(set(repulsions))
        repulsions.sort()
        for repulsion in repulsions:

            # make entry
            configuration['repulsions'].append({'repulsion': repulsion, 'energy': 0})

        # dump into yaml and format
        destination = self.transitions.replace('.txt', '.yaml')
        self._dispense(configuration, destination)
        self._disperse(destination)

        return None