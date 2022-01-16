# scratches.py to represent one etching step

# class Scratch to represent etching steps
class Scratch(object):
    """Class Scratch for representing an etching.

    Inherits from:
        None
    """

    def __init__(self, trajectory, energies, weight, reaction, rate, polarity):
        """Initialize a reaction instance.

        Arguments:
            trajectory: list of list of floats
            eneriges; list of floats, the energy contour
            weight: weight of path
            reaction: Reaction instance
            rate: reaction rate
            polarity: signed int, -1, +1
        """

        # name the reaction from the transition for now
        self.trajectory = trajectory
        self.energies = energies
        self.weight = weight

        # link to reaction
        self.reaction = reaction
        self.rate = rate
        self.polarity = polarity

        # link to reaction's color
        self.color = reaction.color

        return

    def __repr__(self):
        """Represent the reaction on screen.

        Arguments:
            None

        Returns:
            str
        """

        # make a string of the sratches attributes
        formats = [round(self.weight, 2), self.color, self.energies, self.reaction.transition]
        formats += ['-' if self.polarity < 1 else '+', round(self.rate, 2), self.trajectory[0], self.trajectory[2]]
        representation = '< Scratch: {} {} {} [{}], {}({}), {} -> {} >'.format(*formats)

        return representation
