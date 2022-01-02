# lithographs.py for the Lithograph class to make OMI Rawflux hdf5 files

# import cores
from cores import Core
from reactions import Reaction

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

# import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.svm import OneClassSVM, SVC
from sklearn.decomposition import PCA

# import matplotlib for plots
from matplotlib import pyplot
from matplotlib import style as Style
from matplotlib import rcParams
Style.use('fast')
rcParams['axes.formatter.useoffset'] = False


# class Lithograph to do OMI data reduction
class Lithograph(Core):
    """Lithograph class to generate chemical reaction sequences.

    Inherits from:
        Core
    """

    def __init__(self, yam):
        """Initialize a Lithograph instance.

        Arguments:
            transitions: str, file for transition list

        Returns:
            None
        """

        # get the yaml file
        self.yam = yam

        # reserve for bonds, species, and forces
        self.chemicals = []
        self.species = {}
        self.bonds = {}
        self.forces = {}
        self._populate()

        # begin the lattice and the etching
        self.lattice = []
        self.etching = []

        return

    def __repr__(self):
        """Create string for on screen representation.

        Arguments:
            None

        Returns:
            str
        """

        # display contents
        self._tell(self)

        # create representation
        representation = ' < Lithograph instance at: {} >'.format(self.yam)

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

    def _bind(self, chemical):
        """Create list of bonds in a chemical.

        Arguments:
            chemical: str

        Returns:
            list of str
        """

        pairs = zip(chemical[:-1], chemical[1:])
        bonds = [self._flip(''.join(pair)) for pair in pairs]
        bonds.sort()

        return bonds

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

    def _energize(self, point):
        """Calculate the energy of a point.

        Arguments:
            point: list of ints

        Returns:
            float
        """

        # add up all the energies
        chemicals = self.chemicals
        energy = sum([quantity * self.species[chemical]['energy'] for quantity, chemical in zip(point, chemicals)])

        return energy

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
            list of str
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
        chemicals = [electrophile, nucleophile, reactant, product, leaver]
        chemicals = [self._flip(chemical) for chemical in chemicals]

        return chemicals

    def _generate(self, text):
        """Generate the reaction information from list of transition states.

        Arguments:
            text: text file of transitions

        Returns:
            None
        """

        # get information from transitions file
        transitions = self._know(text)

        # begin configuration
        configuration = {'file': self.yam, 'reactions': [], 'species': [], 'bonds': [], 'repulsions': []}

        # for each transition
        species = []
        bonds = []
        repulsions = []
        for transition in transitions:

            # add to species, except for electrophile
            chemicals = self._parse(transition)
            species += chemicals[1:]

            # unpack
            electrophile, nucleophile, reactant, product, leaver = chemicals

            # make reaction
            reaction = {'transition': transition, 'electrophile': electrophile, 'nucleophile': nucleophile}
            reaction.update({'reactant': reactant, 'product': product, 'leaver': leaver})
            reaction.update({'catalysis': 0, 'color': 'black'})
            configuration['reactions'].append(reaction)

        # remove duplicates and sort
        species = list(set(species))
        species.sort()

        # add each entry
        for chemical in species:

            # make entry
            configuration['species'].append({'chemical': chemical, 'quantity': 0})

            # get all bonds and repulsions
            bonds += self._bind(chemical)
            repulsions += self._repulse(chemical)

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
        self._dispense(configuration, self.yam)
        self._disperse(self.yam)

        return None

    def _populate(self):
        """Populate with reactions.

        Arguments:
            None

        Returns:
            None

        Populates:
            self
        """

        # try to
        try:

            # read in the configurationn
            yam = self._acquire(self.yam)

            # unpack yam
            species, reactions, bonds, forces = yam['species'], yam['reactions'], yam['bonds'], yam['repulsions']

            # convert bonds to dictionary
            species = {record['chemical']: record for record in species}
            bonds = {record['bond']: record for record in bonds}
            forces = {record['repulsion']: record for record in forces}

            # update each species with its energy
            for chemical, record in species.items():

                # calculate attractions and repulsions, and update energy
                attractions = sum([bonds[bond]['energy'] for bond in self._bind(chemical)])
                repulsions = sum([forces[force]['energy'] for force in self._repulse(chemical)])
                record.update({'energy': attractions + repulsions})

            # for each reaction
            for reaction in reactions:

                # unpack reaction
                electrophile, nucleophile = reaction['electrophile'], reaction['nucleophile']
                reactant, product, leaver = reaction['reactant'], reaction['product'], reaction['leaver']
                catalysis = reaction['catalysis']

                # get bond energies of reactants
                chemicals = (nucleophile, reactant)
                reactants = sum([bonds[bond]['energy'] for chemical in chemicals for bond in self._bind(chemical)])
                reactants += sum([forces[force]['energy'] for chemical in chemicals for force in self._repulse(chemical)])

                # get bond energies of transition state
                chemicals = (nucleophile, electrophile, leaver)
                transition = sum([bonds[bond]['energy'] for chemical in chemicals for bond in self._bind(chemical)])
                transition += sum([forces[force]['energy'] for chemical in chemicals for force in self._repulse(chemical)])

                # get bond energies of products
                chemicals = (product, leaver)
                products = sum([bonds[bond]['energy'] for chemical in chemicals for bond in self._bind(chemical)])
                products += sum([forces[force]['energy'] for chemical in chemicals for force in self._repulse(chemical)])

                # calculate the forward and backward reection rates
                forward = math.exp(-(catalysis + transition - reactants))
                backward = math.exp(-(catalysis + transition - products))

                # create reaction instance
                member = Reaction(reaction, forward, backward)
                self.append(member)

            # add attributes
            self.species = species
            self.bonds = bonds
            self.forces = forces

            # add list of chemicals
            chemicals = list(species.keys())
            chemicals.sort()
            self.chemicals = chemicals

        # unless not found
        except (KeyError, FileNotFoundError):

            # print warning
            self._print('no yams!')

        return None

    def _repulse(self, chemical):
        """Create list of repulsions in a chemical.

        Arguments:
            chemical: str

        Returns:
            list of str
        """

        # get all repulsions from all elements two apart
        triplets = zip(chemical[:-2], chemical[1:-1], chemical[2:])
        repulsions = [self._flip(''.join([triplet[0], triplet[2]])) for triplet in triplets]
        repulsions.sort()

        return repulsions

    def etch(self, number=20):
        """Etch a glass for a number of steps.

        Arguments:
            number: int, number of time steps

        Returns:
            None

        Populates:
            self.etchingg
            self.lattice
        """

        # specify starting point and energy
        chemicals = self.chemicals
        point = [float(self.species[chemical]['quantity']) for chemical in chemicals]
        energy = self._energize(point)

        # for each time step
        for step in range(number):

            # grab current composition
            composition = {chemical: quantity for chemical, quantity in zip(chemicals, point)}

            # for each reaction
            rates = []
            for reaction in self:

                # calculate the forward and backward rates
                forward = reaction.forward * composition[reaction.nucleophile] * composition[reaction.reactant]
                backward = reaction.backward * composition[reaction.leaver] * composition[reaction.product]

                # add to rates
                rates.append((forward, 1, reaction))
                rates.append((backward, -1, reaction))

            # calculate the logarithm of each rate
            logarithms = [math.log(velocity + 1) for velocity, _, _ in rates]
            weights = [logarithm / sum(logarithms) for logarithm in logarithms]

            # construct lattice points for each reaction
            lattice = []
            for rate, weight in zip(rates, weights):

                # unpack rate
                _, polarity, reaction = rate

                # calculate transition point
                delta = [0.0] * len(chemicals)
                delta[chemicals.index(reaction.nucleophile)] = -0.5 * polarity
                delta[chemicals.index(reaction.reactant)] = -0.5 * polarity
                delta[chemicals.index(reaction.product)] = 0.5 * polarity
                delta[chemicals.index(reaction.leaver)] = 0.5 * polarity

                # calculate transition activation energy
                transition = [entry + change for entry, change in zip(point, delta)]
                activation = energy - math.log(reaction.forward * (polarity > 0) + reaction.backward * (polarity < 0))

                # calculate final point
                delta = [0.0] * len(chemicals)
                delta[chemicals.index(reaction.nucleophile)] = -1.0 * polarity
                delta[chemicals.index(reaction.reactant)] = -1.0 * polarity
                delta[chemicals.index(reaction.product)] = 1.0 * polarity
                delta[chemicals.index(reaction.leaver)] = 1.0 * polarity

                # calculate energy of products
                final = [entry + change for entry, change in zip(point, delta)]
                drop = self._energize(final)

                # add to lattice
                entry = [weight, (energy, activation, drop), point, transition, final]
                lattice.append(entry)

            # pick a random reaction based on weights
            choice = numpy.random.choice(list(range(len(rates))), p=weights)

            # add choice to etching
            etching = lattice[choice].copy()
            etching[0] = rates[choice][2].color

            # update point and energy
            point = etching[-1]
            energy = etching[1][-1]

            # update records
            self.etching.append(etching)
            self.lattice += lattice

        return None

    def peer(self):
        """See a flat representation of the trajectory.

        Arguments:
            None

        Returns:
            None
        """

        # create a matrix from all points in the etching
        matrix = []
        etching = self.etching
        for etch in etching:

            # add all points
            matrix += etch[2:5]

        # create decomposition
        matrix = numpy.array(matrix)
        machine = PCA(n_components=2)
        machine.fit(matrix)

        # begin plot
        pyplot.clf()

        # plot all lattice slats
        for slat in self.lattice:

            # get the point from the machine
            points = machine.transform(slat[2:5])

            # calculate a line width for the weight
            weight = slat[0]
            width = (weight + 0.1) * 5

            # plot the line
            horizontals = [point[0] for point in points]
            verticals = [point[1] for point in points]
            pyplot.plot(horizontals, verticals, color='gray', marker='o', linewidth=width)

        # plot all etchings
        for slat in self.etching:

            # get the point from the machine
            points = machine.transform(slat[2:5])

            # get the color and set the width
            color = slat[0]
            width = 1

            # plot the line
            horizontals = [point[0] for point in points]
            verticals = [point[1] for point in points]
            pyplot.plot(horizontals, verticals, color=color, marker='o', linewidth=width)

        # save the plot and clear
        pyplot.savefig('peer.png')
        pyplot.clf()

        return None