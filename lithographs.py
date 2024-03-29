# lithographs.py for the Lithograph class to make OMI Rawflux hdf5 files

# import cores
from cores import Core
from reactions import Reaction
from scratches import Scratch

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
from mpl_toolkits import mplot3d
Style.use('fast')
rcParams['axes.formatter.useoffset'] = False


# class Lithograph to do OMI data reduction
class Lithograph(Core):
    """Lithograph class to generate chemical reaction sequences.

    Inherits from:
        Core
    """

    def __init__(self, folder):
        """Initialize a Lithograph instance.

        Arguments:
            folder: str, folder

        Returns:
            None
        """

        # make placeholders
        self.folder = folder
        self.text = ''
        self.yam = ''

        # reserve for bonds, species, and forces
        self.chemicals = []
        self.species = {}
        self.bonds = {}
        self.forces = {}
        self._populate()

        # begin the lattice and the etching
        self.lattice = []
        self.etching = []
        self.history = []

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

    def _assemble(self, vertical, horizontal, initial=None, extent=20):
        """Assemble points and rates for quiver plots.

        Arguments:
            vertical: list of str or dict of str, float, species on y axis
            horizontal: list of str or dict of str, float, species on x axis
            initial: list of floats, initial coordinates
            extent: float, extent along axes

        Returns:
            list of vector field coordinates
        """

        # vectorize axes
        vertical = self._vectorize(vertical)
        horizontal = self._vectorize(horizontal)

        # get base vector
        base = self._initialize(initial)

        # begin points
        points = []

        # get half of extent
        half = math.floor(extent / 2)

        # for each node
        for node in range(-half, half + 1):

            # make row
            row = []

            # and each node again
            for nodeii in range(-half, half + 1):

                # create a point
                point = base + node * vertical + nodeii * horizontal
                row.append(point)

            # append to points
            points.append(row)

        # begin rate vectors
        rates = []

        # for each row
        for row in points:

            # make second row
            rowii = []

            # for each point
            for point in row:

                # calculate rate
                rate = self._rate(point)

                # add to rates
                rowii.append(rate)

            # append to rates
            rates.append(rowii)

        # convert points by dot product
        abscissa = numpy.array([[numpy.dot(horizontal, point) for point in row] for row in points])
        ordinate = numpy.array([[numpy.dot(vertical, point) for point in row] for row in points])

        # convert rates by dot product
        vector = numpy.array([[numpy.dot(horizontal, rate) for rate in row] for row in rates])
        vectorii = numpy.array([[numpy.dot(vertical, rate) for rate in row] for row in rates])

        return abscissa, ordinate, vector, vectorii

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

    def _initialize(self, initial=None):
        """Create initial species vector.

        Arguments:
            initial: dict of species, quantities

        Returns:
            numpy array, the center point
        """

        # get default initial
        initial = initial or {}

        # construct base vector
        base = [self.species[chemical]['quantity'] for chemical in self.chemicals]

        # replace with initials
        for chemical, quantity in initial.items():

            # find the index
            index = self.chemicals.index(chemical)
            base[index] = quantity

        # convert to array
        base = numpy.array(base)

        return base

    def _label(self, chemicals):
        """Make a vector label from chemicals and quantities.

        Arguments:
            chemicals: list of str, or dict of str, floats

        Returns:
            numpy array
        """

        # check for list
        if isinstance(chemicals, list):

            # convert to dict
            chemicals = {chemical: 1.0 for chemical in chemicals}

        # sort items
        details = list(chemicals.items())
        details.sort(key=lambda pair: abs(pair[1]), reverse=True)

        # creat text pairs
        texts = ['{}: {}'.format(chemical, quantity) for chemical, quantity in details]

        # join to make a label
        label = ', '.join(texts)

        return label

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

        # if text is not emtpy
        if len(transitions) > 0:

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
                configuration['species'].append({'chemical': chemical, 'quantity': 0, 'color': 'black'})

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

        # otherwise
        else:

            # print error
            self._print('no text file, creating...')
            self._jot([], self.text)

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

        # make folder
        self._make(self.folder)

        # populate text and yam
        self.text = '{}/{}.txt'.format(self.folder, self.folder)
        self.yam = '{}/{}.yaml'.format(self.folder, self.folder)

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
                energies = (reactants, transition + catalysis, products)
                member = Reaction(reaction, forward, backward, energies)
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
            self._print('no yams!, generating...')

            # generate from text
            self._generate(self.text)

        return None

    def _randomize(self, number=1000, initial=None, extent=20):
        """Create a set of ranom points within the designated extent.

        Arguments:
            number: number of points
            initial: dict of species, quantities
            extent: int, extent of sample space around initials

        Returns:
            numpy array of points
        """

        # get base
        base = self._initialize(initial)

        # begin points with random numbers
        random = numpy.random.random((number, len(self.chemicals)))

        # center on extent
        random = (random - 0.5) * extent

        # add to base
        points = random + base

        return points

    def _rate(self, point):
        """Calculate the rate vector for a given point.

        Arguments:
            point: list of floats

        Returns:
            numpy array
        """

        # begin rate vector
        rate = [0] * len(self.chemicals)

        # for each reaction
        for reaction in self:

            # get indices
            chemicals = [reaction.nucleophile, reaction.reactant]
            chemicals += [reaction.product, reaction.leaver]
            indices = [self.chemicals.index(chemical) for chemical in chemicals]

            # compute change for forward and backward reactions
            change = reaction.forward * point[indices[0]] * point[indices[1]]
            changeii = reaction.backward * point[indices[2]] * point[indices[3]]

            # add to rate vector
            rate[indices[0]] += (changeii - change)
            rate[indices[1]] += (changeii - change)
            rate[indices[2]] += (change - changeii)
            rate[indices[3]] += (change - changeii)

        # convert to numpy
        rate = numpy.array(rate)

        return rate

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

    def _vectorize(self, chemicals):
        """Vectorize a list of chemicals.

        Arguments:
            chemicals: list of str, or dict of str, floats

        Returns:
            numpy array
        """

        # check for list
        if isinstance(chemicals, list):

            # convert to dict
            chemicals = {chemical: 1.0 for chemical in chemicals}

        # begin vector
        vector = [chemicals.get(chemical, 0) for chemical in self.chemicals]

        # convert to array
        vector = numpy.array(vector)

        return vector

    def diverge(self, vertical, horizontal, initial=None, extent=20, tag=''):
        """Draw a divergence plot based on a two dimensional projection.

        Arguments:
            vertical: list of floats, vertical axis
            horizontal: list of floats, horizontal axis
            initial: list of floats, initial coordinates
            extent: float, extent along axes
            tag: str, tag for file

        Returns:
            None
        """

        # get vector field coordinates
        abscissa, ordinate, vector, vectorii = self._assemble(vertical, horizontal, initial, extent)

        # get quantities for each species
        quantities = []
        for chemical in self.chemicals:

            # compute quantity
            quantity = abscissa * horizontal.get(chemical, 0) + ordinate * vertical.get(chemical, 0)
            quantities.append(quantity)

        # calculate divergence contributions for each chemical
        divergence = []
        for reaction in self:

            # add contribution from reactant
            index = self.chemicals.index(reaction.reactant)
            contribution = -reaction.forward * quantities[index]
            divergence.append(contribution)

            # add contribution from nucleophile
            index = self.chemicals.index(reaction.nucleophile)
            contribution = -reaction.forward * quantities[index]
            divergence.append(contribution)

            # add contribution from product
            index = self.chemicals.index(reaction.product)
            contribution = -reaction.backward * quantities[index]
            divergence.append(contribution)

            # add contribution from leaving group
            index = self.chemicals.index(reaction.leaver)
            contribution = -reaction.backward * quantities[index]
            divergence.append(contribution)

        # add all contributions
        divergence = numpy.array(divergence)
        divergence = divergence.sum(axis=0)

        # plot divergence
        pyplot.clf()
        pyplot.imshow(divergence)
        pyplot.title('Divergence')
        #
        # # Setting x, y boundary limits
        # pyplot.xlim(abscissa.min(), abscissa.max())
        # pyplot.ylim(ordinate.min(), ordinate.max())
        #
        # set labels
        pyplot.xlabel(self._label(horizontal))
        pyplot.ylabel(self._label(vertical))
        #
        # # plot with grid
        # pyplot.grid()
        #
        # if given tag
        if tag:

            # add underscore
            tag = '_{}'.format(tag)

        # save
        pyplot.savefig('{}/divergence{}.png'.format(self.folder, tag))

        return

    def etch(self, number=20, stochastic=True):
        """Etch a glass for a number of steps.

        Arguments:
            number: int, number of time steps
            stochastic: boolean, roll the dice for reaction?

        Returns:
            None

        Populates:
            self.etchingg
            self.lattice
        """

        # print
        self._print('etching...')

        # specify starting point and energy
        chemicals = self.chemicals
        point = [float(self.species[chemical]['quantity']) for chemical in chemicals]

        # for each time step
        for step in range(number):

            # make lattice
            lattice = self.lace(point)

            # pick next point  based on highest weight
            lattice.sort(key=lambda slat: slat.weight, reverse=True)
            etching = lattice[0]

            # but if rolling
            if stochastic:

                # pick a random reaction based on weights
                weights = [slat.weight for slat in lattice]
                etching = numpy.random.choice(lattice, p=weights)

            # update point and energy
            point = etching.trajectory[-1]

            # update records
            self.etching.append(etching)
            self.lattice += lattice
            self.history.append(etching.reaction)

        return None

    def gaze(self, grid=False):
        """See a #D representation of the trajectory.

        Arguments:
            grid: boolean, plot grid?

        Returns:
            None
        """

        # print
        self._print('gazing...')

        # create a matrix from all points in the etching
        matrix = []
        for scratch in self.etching:

            # add all points
            matrix += scratch.trajectory

        # create decomposition
        matrix = numpy.array(matrix)
        machine = PCA(n_components=2)
        machine.fit(matrix)

        # begin plot
        pyplot.clf()
        figure = pyplot.figure()
        axis = pyplot.axes(projection='3d')

        # if grid
        if grid:

            # plot all lattice slats
            for slat in self.lattice:

                # get the point from the machine
                points = machine.transform(slat.trajectory)
                energies = slat.energies

                # calculate a line width for the weight
                weight = slat.weight
                width = (weight + 0.1) * 5

                # plot the line
                horizontals = [point[0] for point in points]
                verticals = [point[1] for point in points]
                axis.plot(horizontals, verticals, energies, color='gray', marker=',', linewidth=width)

        # plot all etchings
        for scratch in self.etching:

            # set label
            label = scratch.reaction.transition

            # get the point from the machine
            points = machine.transform(scratch.trajectory)
            energies = scratch.energies

            # get the color and set the width
            color = scratch.color
            width = 1

            # plot the line
            horizontals = [point[0] for point in points]
            verticals = [point[1] for point in points]
            axis.plot(horizontals, verticals, energies, color=color, marker=',', linewidth=width)

            # plot a marker
            marker = '2' if energies[2] > energies[0] else '1'
            #axis.plot([horizontals[1]], [verticals[1]], [energies[1]], color=color, marker=marker, markersize=5)

        # # add legend
        # pyplot.legend(loc='best')

        # save the plot and clear
        axis.view_init(30, 125)
        pyplot.savefig('{}/gaze.png'.format(self.folder))
        axis.view_init(30, -125)
        pyplot.savefig('{}/gazeii.png'.format(self.folder))
        pyplot.clf()
        pyplot.close()

        return None

    def lace(self, point):
        """Construct the lattice based on a composition.

        Arguments:
             point: composition vector

        Returns:
            list of Scratch instances
        """

        # grab chemicals
        chemicals = self.chemicals

        # construction composition and energy
        energy = self._energize(point)
        composition = {chemical: quantity for chemical, quantity in zip(chemicals, point)}

        # for each reaction
        rates = []
        for index, reaction in enumerate(self):

            # calculate the forward and backward rates
            forward = reaction.forward * composition[reaction.nucleophile] * composition[reaction.reactant]
            backward = reaction.backward * composition[reaction.leaver] * composition[reaction.product]

            # add to rates
            rates.append((forward, 1, reaction, index))
            rates.append((backward, -1, reaction, index))

        # calculate the logarithm of each rate
        logarithms = [math.log(velocity + 1) for velocity, _, _, _ in rates]
        weights = [logarithm / sum(logarithms) for logarithm in logarithms]
        # velocities = [velocity for velocity, _, _, _ in rates]
        # weights = [velocity / sum(velocities) for velocity in velocities]

        # construct lattice points for each reaction
        lattice = []
        for rate, weight in zip(rates, weights):

            # unpack rate
            _, polarity, reaction, _ = rate

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

            # create scratch
            energies = (energy, activation, drop)
            trajectory = (point, transition, final)
            scratch = Scratch(trajectory, energies, weight, reaction, rate[0], polarity)

            # add to lattice
            lattice.append(scratch)

        return lattice

    def flow(self):
        """See a 3D representation of one point in the trajectory.

        Arguments:
            None

        Returns:
            None
        """

        # print
        self._print('flowing...')

        # specify starting point
        chemicals = self.chemicals
        point = [float(self.species[chemical]['quantity']) for chemical in chemicals]

        # grab the lattice at the point
        lattice = self.lace(point)

        # create a matrix from all points in the etching
        matrix = []
        for slat in lattice:

            # add all points
            matrix += slat.trajectory

        # create decomposition machine
        matrix = numpy.array(matrix)
        machine = PCA(n_components=2)
        machine.fit(matrix)

        # set default input to true
        propagate = True
        while propagate:

            # begin plot
            pyplot.clf()
            axis = pyplot.axes(projection='3d')

            # plot all lattice slats
            for slat in lattice:

                # get the point from the machine
                points = machine.transform(slat.trajectory)
                energies = slat.energies

                # calculate a line width for the weight
                weight = slat.weight
                width = (weight + 0.1) * 5
                color = slat.color

                # plot the line
                horizontals = [point[0] for point in points]
                verticals = [point[1] for point in points]
                axis.plot(horizontals, verticals, energies, color=color, marker=',', linewidth=width)

            # save the plot and clear
            axis.view_init(30, 125)
            pyplot.savefig('{}/flow.png'.format(self.folder))
            axis.view_init(30, -125)
            pyplot.savefig('{}/flowii.png'.format(self.folder))
            pyplot.clf()
            pyplot.close()

            # await input
            propagate = not input('>>>? ')

            # sort lattice by weight
            lattice.sort(key=lambda slat: slat.weight, reverse=True)
            self._print(lattice[0])

            # set new point and lattice
            point = lattice[0].trajectory[-1]
            lattice = self.lace(point)

        return None

    def peer(self, grid=False):
        """See a flat representation of the trajectory.

        Arguments:
            grid: boolean, plot grid?

        Returns:
            None
        """

        # print
        self._print('peering...')

        # create a matrix from all points in the etching
        matrix = []
        etching = self.etching
        for etch in etching:

            # add all points
            matrix += etch.trajectory

        # create decomposition
        matrix = numpy.array(matrix)
        machine = PCA(n_components=2)
        machine.fit(matrix)

        # begin plot
        pyplot.clf()

        # if grid
        if grid:

            # plot all lattice slats
            for slat in self.lattice:

                # get the point from the machine
                points = machine.transform(slat.trajectory)

                # calculate a line width for the weight
                weight = slat.weight
                width = (weight + 0.1) * 5

                # plot the line
                horizontals = [point[0] for point in points]
                verticals = [point[1] for point in points]
                pyplot.plot(horizontals, verticals, color='gray', marker=',', linewidth=width)

        # plot all etchings
        for slat in self.etching:

            # get label
            label = slat.reaction.transition

            # get the point from the machine
            points = machine.transform(slat.trajectory)
            energies = slat.energies

            # get the color and set the width
            color = slat.color
            width = 1

            # plot the line
            horizontals = [point[0] for point in points]
            verticals = [point[1] for point in points]
            pyplot.plot(horizontals, verticals, color=color, marker=',', linewidth=width)

            # plot a marker
            marker = '2' if energies[2] > energies[0] else '1'
            #pyplot.plot([horizontals[1]], [verticals[1]], color=color, marker=marker, markersize=5)

        # # add legend
        # pyplot.legend(loc='best')

        # save the plot and clear
        pyplot.savefig('{}/peer.png'.format(self.folder))
        pyplot.clf()
        pyplot.close()

        return None

    def place(self, point):
        """See a 3D representation of one point in the trajectory.

        Arguments:
            point: index of lattice point

        Returns:
            None
        """

        # print
        self._print('placing...')

        # grab the lattice at the point
        lattice = self.lace(point)

        # create a matrix from all points in the etching
        matrix = []
        for slat in lattice:

            # add all points
            matrix += slat.trajectory

        # create decomposition
        matrix = numpy.array(matrix)
        machine = PCA(n_components=2)
        machine.fit(matrix)

        # begin plot
        pyplot.clf()
        axis = pyplot.axes(projection='3d')

        # plot all lattice slats
        for slat in lattice:

            # get the point from the machine
            points = machine.transform(slat.trajectory)
            energies = slat.energies

            # calculate a line width for the weight
            weight = slat.weight
            width = (weight + 0.1) * 5
            color = slat.color

            # plot the line
            horizontals = [point[0] for point in points]
            verticals = [point[1] for point in points]
            axis.plot(horizontals, verticals, energies, color=color, marker=',', linewidth=width)

        # save the plot and clear
        axis.view_init(30, 125)
        pyplot.savefig('{}/place.png'.format(self.folder))
        axis.view_init(30, -125)
        pyplot.savefig('{}/placeii.png'.format(self.folder))
        pyplot.clf()
        pyplot.close()

        return None

    def plate(self, number=1000, modes=(0, 1), top=3, places=2, initial=None, extent=20, tag=''):
        """Create a plate from the most interesting slice through the phase space.

        Arguments:
            number: number of random points
            modes: tuple of ints, the pca modes to use
            top: int, number of top vector magnitudes to take
            places: int, number of decimal places to round the vector components
            initial: dict of species, quantities
            extent: extent of plate
            tag: str, tag for file name

        Returns:
            None
        """

        # get ranom points
        points = self._randomize(number, initial, extent)

        # convert to rates
        rates = numpy.array([self._rate(point) for point in points])

        # # convert to magnitudes
        # magnitudes = numpy.sqrt((rates ** 2).sum(axis=1))

        # craete pca decomposition
        decomposer = PCA(n_components=len(self.chemicals))
        decomposer.fit(rates)

        # get the componetns
        components = decomposer.components_

        # make horizontal vector
        zipper = list(zip(self.chemicals, components[modes[0]]))
        zipper.sort(key=lambda pair: abs(pair[1]), reverse=True)
        horizontal = {chemical: round(quantity, places) for chemical, quantity in zipper[:top]}

        # make vertical vector
        zipper = list(zip(self.chemicals, components[modes[1]]))
        zipper = [pair for pair in zipper if pair[0] not in horizontal.keys()]
        zipper.sort(key=lambda pair: abs(pair[1]), reverse=True)
        vertical = {chemical: round(quantity, places) for chemical, quantity in zipper[:top]}

        # create plots
        tag = 'pca_{}_{}'.format(*modes)
        self.swirl(vertical, horizontal, initial, extent, tag)

        return None

    def qualify(self):
        """Create a plot of energy with time point.

        Arguments:
            None

        Returns:
            None
        """

        # print
        self._print('qualifying...')

        # grab etching
        etching = self.etching

        # create time series
        series = [etch.energies[0] for etch in etching] + [etching[-1].energies[-1]]

        # begin plot
        pyplot.clf()

        # create vector
        time = [number for number, _ in enumerate(series)]

        # plot
        pyplot.plot(time, series, color='black', marker=',')

        # save plot
        pyplot.savefig('{}/qualify.png'.format(self.folder))
        pyplot.clf()
        pyplot.close()

        return None

    def quantify(self, species=None):
        """Create a plot of all species with time point.

        Arguments:
            species: list of str, species of interest

        Returns:
            None
        """

        # print
        self._print('quantifying...')

        # set default species
        species = species or self.chemicals

        # grab etching
        etching = self.etching

        # create time series
        series = [etch.trajectory[0] for etch in etching] + [etching[-1].trajectory[2]]

        # begin plot
        pyplot.clf()

        # plot each series
        for index, chemical in enumerate(self.chemicals):

            # if in species
            if chemical in species:

                # create vector
                chemistry = [entry[index] for entry in series]
                vector = [math.log10(entry + 1) for entry in chemistry]
                time = [number for number, _ in enumerate(series)]
                color = self.species[chemical]['color']

                # plot
                #pyplot.plot(time, chemistry, color=color, marker=',')
                pyplot.plot(time, vector, color=color, marker=',', label=chemical)

        # add legend
        pyplot.legend(loc='best')

        # save plot
        pyplot.savefig('{}/quantify.png'.format(self.folder))
        pyplot.clf()
        pyplot.close()

        return None

    def quiver(self, vertical, horizontal, initial=None, extent=20, tag=''):
        """Draw a quiver plot based on a two dimensional projection.

        Arguments:
            vertical: list of floats, vertical axis
            horizontal: list of floats, horizontal axis
            initial: list of floats, initial coordinates
            extent: float, extent along axes
            tag: str, tag for file

        Returns:
            None
        """

        # get vector field coordinates
        abscissa, ordinate, vector, vectorii = self._assemble(vertical, horizontal, initial, extent)

        # plot quivwr
        pyplot.clf()
        pyplot.quiver(abscissa, ordinate, vector, vectorii, color='g')
        pyplot.title('Vector Field')

        # Setting x, y boundary limits
        pyplot.xlim(abscissa.min(), abscissa.max())
        pyplot.ylim(ordinate.min(), ordinate.max())

        # set labels
        pyplot.xlabel(self._label(horizontal))
        pyplot.ylabel(self._label(vertical))

        # plot with grid
        pyplot.grid()

        # if given tag
        if tag:

            # add underscore
            tag = '_{}'.format(tag)

        # save
        pyplot.savefig('{}/quiver{}.png'.format(self.folder, tag))

        return

    def recite(self):
        """Recite the reaction counts.

        Arguments:
            None

        Returns:
            None
        """

        # print
        self._print('reciting...')

        # grab history
        history = self.history

        # begin series
        series = [[0 for reaction in self]]
        for entry in history:

            # get index
            indices = [index for index, reaction in enumerate(self) if reaction == entry]

            # add to index
            vector = series[-1][:]
            vector[indices[0]] += 1
            series.append(vector)

        # begin plot
        pyplot.clf()

        # plot each series
        for index, reaction in enumerate(self):

            # create vector
            chemistry = [entry[index] for entry in series]
            time = [number for number, _ in enumerate(series)]
            color = self[index].color

            # plot
            pyplot.plot(time, chemistry, color=color, marker=',', label=reaction.transition)

        # add legend
        pyplot.legend(loc='best')

        # save plot
        pyplot.savefig('{}/recite.png'.format(self.folder))
        pyplot.clf()
        pyplot.close()

        return None

    def stream(self, vertical, horizontal, initial=None, extent=20, tag=''):
        """Draw a stream plot based on a two dimensional projection.

        Arguments:
            vertical: list of floats, vertical axis
            horizontal: list of floats, horizontal axis
            initial: list of floats, initial coordinates
            extent: float, extent along axes
            tag: str, tag for file

        Returns:
            None
        """

        # get vector field coordinates
        abscissa, ordinate, vector, vectorii = self._assemble(vertical, horizontal, initial, extent)

        # plot stream
        pyplot.clf()
        pyplot.streamplot(abscissa, ordinate, vector, vectorii, density=1.4, linewidth=None, color='g')
        pyplot.title('Vector Field')

        # Setting x, y boundary limits
        pyplot.xlim(abscissa.min(), abscissa.max())
        pyplot.ylim(ordinate.min(), ordinate.max())

        # set labels
        pyplot.xlabel(self._label(horizontal))
        pyplot.ylabel(self._label(vertical))

        # plot with grid
        pyplot.grid()

        # if given tag
        if tag:

            # add underscore
            tag = '_{}'.format(tag)

        # save
        pyplot.savefig('{}/stream{}.png'.format(self.folder, tag))

        return

    def study(self, number=100, species=None):
        """Perform an etch and graph results.

        Arguments:
            number: int, number of time steps

        Returns:
            None
        """

        # set default species
        species = species or self.chemicals

        # etch
        self.etch(number)

        # draw graphs
        self.gaze()
        self.peer()
        self.quantify(species=species)
        self.qualify()
        self.recite()

        return None

    def swirl(self, vertical, horizontal, initial=None, extent=20, tag=''):
        """Draw a quiver and stream plot based on a two dimensional projection.

        Arguments:
            vertical: list of floats, vertical axis
            horizontal: list of floats, horizontal axis
            initial: list of floats, initial coordinates
            extent: float, extent along axes
            tag: str, tag for naming files

        Returns:
            None
        """

        # make quiver and stream plots
        self.quiver(vertical, horizontal, initial, extent, tag)
        self.stream(vertical, horizontal, initial, extent, tag)

        # make divergence plots
        self.diverge(vertical, horizontal, initial, extent, tag)

        return

    def tag(self, base, tag):
        """Tag a file with a tag.

        Arguments:
            base: str, base file name
            tag: str, added tag

        Returns:
            None
        """

        # find file
        name = '{}/{}.png'.format(self.folder, base)

        # replace
        self._name(name, '_{}.png'.format(tag), '.png')

        return None

    def trace(self):
        """Trace reaction diagrams.

        Arguments:
            None

        Returns:
            None
        """

        # for each reaction
        for reaction in self:

            # make gaussian for first half
            abscissa = numpy.arange(-3, 0.1, 0.1)
            constant = reaction.energies[1] - reaction.energies[0]
            offset = reaction.energies[0] + (reaction.catalysis - reaction.energies[1])
            ordinate = offset + constant * numpy.exp(-abscissa ** 2)

            # make guassian for second half
            abscissaii = numpy.arange(0, 3.1, 0.1)
            constant = reaction.energies[1] - reaction.energies[2]
            offset = reaction.energies[2] + (reaction.catalysis - reaction.energies[1])
            ordinateii = offset + constant * numpy.exp(-abscissaii ** 2)

            # plot
            pyplot.clf()
            pyplot.plot(abscissa, ordinate, color=reaction.color)
            pyplot.plot(abscissaii, ordinateii, color=reaction.color)

            # make limits
            pyplot.xlim(-3, 3)
            pyplot.ylim(min([ordinate.min(), ordinateii.min()]) - 0.1, 0.1)
            pyplot.ylim(-3 - 0.1, 0.1)

            # add title
            formats = (reaction.nucleophile, reaction.reactant, reaction.transition, reaction.product, reaction.leaver)
            title = '{} + {} -> {} <- {} + {}'.format(*formats)
            pyplot.title(title)

            # save
            destination = '{}/trace_{}.png'.format(self.folder, reaction.transition.replace('-', '_'))
            pyplot.savefig(destination)

        return None
