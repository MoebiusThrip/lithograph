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
            text = self.yam.replace('yaml', 'txt')
            self._generate(text)

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

        # print
        self._print('etching...')

        # specify starting point and energy
        chemicals = self.chemicals
        point = [float(self.species[chemical]['quantity']) for chemical in chemicals]

        # for each time step
        for step in range(number):

            # make lattice
            lattice = self.lace(point)

            # pick a random reaction based on weights
            weights = [slat.weight for slat in lattice]
            etching = numpy.random.choice(lattice, p=weights)

            # update point and energy
            point = etching.trajectory[-1]

            # update records
            self.etching.append(etching)
            self.lattice += lattice

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

        # save the plot and clear
        axis.view_init(30, 135)
        pyplot.savefig('gaze.png')
        axis.view_init(30, -135)
        pyplot.savefig('gazeii.png')
        pyplot.clf()

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
            pyplot.savefig('flow.png')
            axis.view_init(30, -125)
            pyplot.savefig('flowii.png')
            pyplot.clf()

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
            matrix += etch[2:5]

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
                points = machine.transform(slat[2:5])

                # calculate a line width for the weight
                weight = slat[0]
                width = (weight + 0.1) * 5

                # plot the line
                horizontals = [point[0] for point in points]
                verticals = [point[1] for point in points]
                pyplot.plot(horizontals, verticals, color='gray', marker=',', linewidth=width)

        # plot all etchings
        for slat in self.etching:

            # get the point from the machine
            points = machine.transform(slat[2:5])
            energies = slat[1]

            # get the color and set the width
            color = slat[0]
            width = 1

            # plot the line
            horizontals = [point[0] for point in points]
            verticals = [point[1] for point in points]
            pyplot.plot(horizontals, verticals, color=color, marker=',', linewidth=width)

            # plot a marker
            marker = '2' if energies[2] > energies[0] else '1'
            #pyplot.plot([horizontals[1]], [verticals[1]], color=color, marker=marker, markersize=5)

        # save the plot and clear
        pyplot.savefig('peer.png')
        pyplot.clf()

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
        pyplot.savefig('place.png')
        axis.view_init(30, -125)
        pyplot.savefig('placeii.png')
        pyplot.clf()

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
        series = [etch[1][0] for etch in etching] + [etching[-1][1][-1]]

        # begin plot
        pyplot.clf()

        # create vector
        time = [number for number, _ in enumerate(series)]

        # plot
        pyplot.plot(time, series, color='black', marker=',')

        # save plot
        pyplot.savefig('qualify.png')
        pyplot.clf()

        return None

    def quantify(self):
        """Create a plot of all species with time point.

        Arguments:
            None

        Returns:
            None
        """

        # print
        self._print('quantifying...')

        # grab etching
        etching = self.etching

        # create time series
        series = [etch[2] for etch in etching] + [etching[-1][4]]

        # begin plot
        pyplot.clf()

        # plot each series
        for index, chemical in enumerate(self.chemicals):

            # create vector
            chemistry = [entry[index] for entry in series]
            vector = [math.log(entry + 1) for entry in chemistry]
            time = [number for number, _ in enumerate(series)]
            color = self.species[chemical]['color']

            # plot
            #pyplot.plot(time, chemistry, color=color, marker=',')
            pyplot.plot(time, vector, color=color, marker=',')

        # save plot
        pyplot.savefig('quantify.png')
        pyplot.clf()

        return None

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

            # add to index
            vector = series[-1][:]
            vector[(int(entry / 2))] += 1
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
            pyplot.plot(time, chemistry, color=color, marker=',')

        # save plot
        pyplot.savefig('recite.png')
        pyplot.clf()

        return None

    def study(self, number=100):
        """Perform an etch and graph results.

        Arguments:
            number: int, number of time steps

        Returns:
            None
        """

        # etch
        self.etch(number)

        # draw graphs
        self.gaze()
        self.peer()
        self.quantify()
        self.qualify()
        self.recite()

        return None
