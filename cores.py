# cores.py for a core object with basic file directory interaction

# import system tools
import os
import sys
import shutil

# if in python3, use instruction from wiki
if sys.version_info.major == 3:

    # define path
    ACpath = '/tis/releases/ac/python-science/1.0.0/'

    # check for centos distribution
    import distro
    if distro.id() == 'centos':

        # and add paths
        sys.path.append(os.path.join(ACpath,'lib/python3.6/site-packages'))
        sys.path.append(os.path.join(ACpath,'lib64/python3.6/site-packages'))
        print('Setting path for Python 3 on CentOS')

    # otherwise if ubuntu
    elif distro.id() == 'ubuntu':

        # add paths
        sys.path.append(os.path.join(ACpath,'lib/python3.8/site-packages'))
        print('Setting path for Python 3 on Ubuntu')

# otherwise, if python 2
if sys.version_info.major == 2:

    # define missing exceptions
    FileExistsError = OSError
    FileNotFoundError = IOError
    PermissionError = OSError

# print path
print('Path is', sys.path)

# import file manipulations
import json
import csv
import yaml

# import time
import time

# import pretty print
import pprint


# class Core
class Core(list):
    """Class core to define basic file manipulatioin.

    Inherits from:
        list
    """

    def __init__(self):
        """Instantiate a Core instance.

        Arguments:
            None
        """

        # set current time
        self.now = time.time()

        return

    def __repr__(self):
        """Create string for on screen representation.

        Arguments:
            None

        Returns:
            str
        """

        # create representation
        representation = ' < Core instance >'

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

    def _clean(self, directory):
        """Delete all files in a directory and the directory itself.

        Arguments:
            directory: str, directory path

        Returns:
            None
        """

        # get all paths in the directory
        paths = self._see(directory)

        # prompt user and continue on blank
        prompt = input('erase {} !?>>'.format(directory))
        if prompt in ('', ' '):

            # remove all files
            [os.remove(path) for path in paths]

            # print status
            self._print('{} erased.'.format(directory))

        return None

    def _copy(self, path, destination):
        """Copy a file from one path to another.

        Arguments:
            path: str, filepath
            destination: str, filepath

        Returns:
            None
        """

        # try to copy
        try:

            # copy the file
            shutil.copy(path, destination)

        # unless it is a directory
        except IsADirectoryError:

            # in which case, alert and skip
            self._print('{} is a directory'.format(path))

        return None

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

    def _dump(self, contents, destination):
        """Dump a dictionary into a json file.

        Arguments:
            contents: dict
            destination: str, destination file path

        Returns:
            None
        """

        # dump file
        with open(destination, 'w') as pointer:

            # dump contents
            self._print('dumping into {}...'.format(destination))
            json.dump(contents, pointer)

        return None

    def _group(self, members, function):
        """Group a set of members by the result of a function of the member.

        Arguments:
            members: list of dicts
            function: str

        Return dict
        """

        # get all fields
        fields = self._skim([function(member) for member in members])

        # create groups
        groups = {field: [] for field in fields}
        [groups[function(member)].append(member) for member in members]

        return groups

    def _jot(self, lines, destination, mode='w'):
        """Jot down the lines into a text file.

        Arguments:
            lines: list of str
            destination: str, file path
            mode: str, open mode

        Returns:
            None
        """

        # add final endline
        lines = [line + '\n' for line in lines]

        # save lines to file
        with open(destination, mode) as pointer:

            # write lines
            pointer.writelines(lines)

        return None

    def _kick(self, source, sink, *extensions):
        """Move all files in the source directory to the sink.

        Arguments:
            source: str, path to directory
            sink: str, path to directory
            *extensions: unpacked list of extensions

        Returns:
            None
        """

        # make the sink directory
        self._make(sink)

        # get all source paths, only keeping relevant extensions
        paths = self._see(source)
        paths = [path for path in paths if any([extension in path for extension in extensions])]

        # for each path
        for path in paths:

            # create destination
            destination = '{}/{}'.format(sink, path.split('/')[-1])

            # move file
            os.rename(path, destination)

        return None

    def _know(self, path):
        """Transcribe the text file at the path.

        Arguments:
            path: str, file path

        Returns:
            list of str
        """

        # set default lines
        lines = []

        # try to
        try:

            # read in file pointer
            with open(path, 'r') as pointer:

                # read in text and eliminate endlines
                lines = pointer.readlines()
                lines = [line.replace('\n', '') for line in lines]

        # unless it doesn't yet exist
        except FileNotFoundError:

            # in which case, alert and pass empty
            self._print('no such file as {}'.format(path))
            pass

        return lines

    def _load(self, path):
        """Load a json file.

        Arguments:
            path: str, file path

        Returns:
            dict
        """

        # try to
        try:

            # open json file
            with open(path, 'r') as pointer:

                # get contents
                self._print('loading {}...'.format(path))
                contents = json.load(pointer)

        # unless the file does not exit
        except FileNotFoundError:

            # in which case return empty json
            self._print('creating {}...'.format(path))
            contents = {}

        return contents

    def _look(self, contents, level=0):
        """Look at the contents of an h5 file, to a certain level.

        Arguments:
            contents: object to look at
            level=2: the max nesting level to see
            five: the h5 file

        Returns:
            None
        """

        # unpack into a dict
        data = self._unpack(contents, level=level)

        # pretty print it
        pprint.pprint(data)

        return None

    def _make(self, folder):
        """Make a folder in the directory if not yet made.

        Arguments:
            folder: str, directory path

        Returns:
            None
        """

        # try to
        try:

            # create the directory
            os.mkdir(folder)

        # unless directory already exists, or none was given
        except (FileExistsError, FileNotFoundError, PermissionError):

            # in which case, nevermind
            pass

        return None

    def _print(self, *messages):
        """Print the message, localizes print statements.

        Arguments:
            *messagea: unpacked list of str, etc

        Returns:
            None
        """

        # construct  message
        message = ', '.join([str(message) for message in messages])

        # print
        print(message)

        # # go through each meassage
        # for message in messages:
        #
        #     # print
        #     print(message)

        return None

    def _see(self, directory):
        """See all the paths in a directory.

        Arguments:
            directory: str, directory path

        Returns:
            list of str, the file paths.
        """

        # make paths
        paths = ['{}/{}'.format(directory, path) for path in os.listdir(directory)]

        return paths

    def _skim(self, members):
        """Skim off the unique members from a list.

        Arguments:
            members: list

        Returns:
            list
        """

        # trim duplicates and sort
        members = list(set(members))
        members.sort()

        return members

    def _stamp(self, message, initial=False, clock=True):
        """Start timing a block of code, and print results with a message.

        Arguments:
            message: str
            initial: boolean, initial message of block?
            clock: boolean, include timing in output?

        Returns:
            float
        """

        # get final time
        final = time.time()

        # calculate duration and reset time
        duration = round(final - self.now, 7)

        # if rounding to magnitudes:
        if not clock:

            # set duration to approximation
            duration = 0.0

        # reset tme
        self.now = final

        # if initial message
        if initial:

            # add newline
            message = '\n' + message

        # if not an initial message
        if not initial:

            # print duration
            self._print('took {} seconds.'.format(duration))

        # begin new block
        self._print(message)

        return duration

    def _tabulate(self, rows, destination):
        """Create a csv file from a list of records.

        Arguments:
            rows: list of list of strings
            destination: str, file path

        Returns:
            None
        """

        # write rows
        with open(destination, 'w') as pointer:

            # write csv
            csv.writer(pointer).writerows(rows)

        return None

    def _tell(self, queue):
        """Enumerate the contents of a list.

        Arguments:
            queue: list

        Returns:
            None
        """

        # for each item
        for index, member in enumerate(queue):

            # print
            self._print('{}) {}'.format(index, member))

        # print spacer
        self._print(' ')

        return None

    def _unpack(self, contents, level=100, nest=0):
        """Unpack an h5 file into a dict.

        Arguments:
            contents: h5 file or dict
            level=100: int, highest nesting level kept
            nest=0: current nesting level

        Returns:
            dict
        """

        # check for nesting level
        if nest > level:

            # data is truncated to an ellipsis
            data = '...'

        # otherwise
        else:

            # try
            try:

                # to go through keys
                data = {}
                for field, info in contents.items():

                    # add key to data
                    data[field] = self._unpack(info, level, nest + 1)

            # unless there are no keys
            except AttributeError:

                # just return the contents
                data = contents

        return data