# cores.py for a core object with basic file directory interaction

# import system tools
import os
import sys
import shutil

# if in python3, use instruction from wiki
if sys.version_info.major == 3:

    # try to
    try:

        # import distro (not available in python 2)
        import distro

        # define path
        ACpath = '/tis/releases/ac/python-science/1.0.0/'

        # if running on Centos
        if distro.id() == 'centos':

            # define path for python3 distribution
            sys.path.insert(1,os.path.join(ACpath,'lib/python3.6/site-packages'))
            sys.path.insert(1,os.path.join(ACpath,'lib64/python3.6/site-packages'))
            print('Setting path for Python 3 on CentOS')

        # otherwise, for Ubuntu 20
        elif distro.id() == 'ubuntu':

            # define path for python 3 distribution
            sys.path.insert(1,os.path.join(ACpath,'lib/python3.8/site-packages'))
            print('Setting path for Python 3 on Ubuntu')

    # unless on nccs
    except ModuleNotFoundError:

        # skip
        print('skipping distro, on nccs?')

# otherwise, if python 2
if sys.version_info.major == 2:

    # define missing exceptions
    FileExistsError = OSError
    FileNotFoundError = IOError
    PermissionError = OSError

# print path
print('Path is', sys.path[1:])

# import file manipulations
import json
import csv
import yaml

# import time, datetime
import datetime
import time

# import pretty print
import pprint

# import regex
import re

# import math
import math

# import scipy
import scipy

# try to
try:

    # import
    import pidly

# unless not pressent
except ImportError:

    # in which case, pass
    pass

# try to
try:

    # import
    import openpyxl

# unless not pressent
except ImportError:

    # in which case, pass
    pass


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

        # default information to empty
        information = {}

        # try to
        try:

            # open yaml file
            with open(path, 'r') as pointer:

                # and read contents
                information = yaml.safe_load(pointer)

        # unless not found
        except FileNotFoundError:

            # in which case print error
            self._print('yaml {} not found!'.format(path))

        return information

    def _ask(self, path):
        """Retrieve file details.

        Arguments:
            path: str, path name

        Returns:
            dict
        """

        # get file stats
        details = os.stat(path)

        # make into dictionary
        fields = [field for field in dir(details) if field.startswith('st')]
        details = {field: details.__getattribute__(field) for field in fields}

        # trasnlate details
        details['metadata'] = str(datetime.datetime.fromtimestamp(details['st_ctime']))
        details['accessed'] = str(datetime.datetime.fromtimestamp(details['st_atime']))
        details['modified'] = str(datetime.datetime.fromtimestamp(details['st_mtime']))
        details['megabytes'] = details['st_size'] / 1024 ** 2

        return details

    def _chop(self, length, number=0, size=None):
        """Get the sets of indices that chop a list into chunks.

        Arguments:
            length: int, length of list
            number: int, number of chunks
            size: int, size of each chunk

        Returns:
            list of (int, int) tuples, the bounding indices
        """

        # if given a size
        if size:

            # determine number of chunks
            number = math.ceil(length / size)

        # otherwise,
        else:

            # construct size
            size = math.ceil(length / number)

        # construct indices
        pairs = [(index * size, size + index * size) for index in range(number)]

        return pairs

    def _clean(self, directory, force=False):
        """Delete all files in a directory and the directory itself.

        Arguments:
            directory: str, directory path
            force=True: boolean, avoid asking

        Returns:
            None
        """

        # try to
        try:

            # get all paths in the directory
            paths = self._see(directory)

        # unless not a directory
        except NotADirectoryError:

            # set path to just file
            paths = [directory]

        # set default prompt
        prompt = 'X'

        # if not forcing delete
        if not force:

            # prompt user and continue on blank
            prompt = input('erase {} !?>>'.format(directory))

        # if blank reply or forcing
        if prompt in ('', ' ') or force:

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

            # in which case, copy into directory
            pathii = '{}/{}'.format(destination, path.split('/')[-1])
            shutil.copy(path, pathii)

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

    def _excel(self, table, delimiter=','):
        """Create rows from an excel file.

        Arguments:
            table: str, excel file
            delimiter: delimitation charactr

        Returns:
            list of list of str
        """

        # begin rows
        rows = []

        # open the workbook
        book = openpyxl.load_workbook(table)

        # activate
        sheet = book.active

        # for each row
        for row in sheet.iter_rows(values_only=True):

            # add row
            rows.append(list(row))

        return rows

    def _file(self, path, folders=0):
        """Get the filename from a path.

        Arguments:
            path: str, pathname
            folders: int, number of subfolders

        Returns:
            str, file name
        """

        # split on slash
        fragments = path.split('/')

        # join together fragments according to number
        name = '/'.join(fragments[-(1 + folders):])

        return name

    def _fold(self, path):
        """Break apart a path name into directory and file.

        Arguments:
            path: str, pathname

        Returns:
            str, directory
        """

        # get fileame
        words = path.split('/')

        # get folder
        folder = '/'.join(words[:-1])

        return folder

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

    def _idle(self, destination, data):
        """Construct and IDL file from a dict of data.

        Arguments:
            destination: str, the filepath
            data: dict of arrays, the data

        Returns:
            None
        """

        # begin idl session
        idle = pidly.IDL()

        # for each field
        for field, array in data.items():

            # add data
            setattr(idle, field, array)

        # save variable list, with variables at top level
        variables = ' ,'.join(data.keys())

        # save file
        idle('save, {}, filename="{}"'.format(variables, destination))
        self._print('{} saved.'.format(destination))

        return None

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

    def _look(self, contents, level=0, destination=None):
        """Look at the contents of an h5 file, to a certain level.

        Arguments:
            contents: object to look at
            level=2: the max nesting level to see
            destination: str, file path for destination

        Returns:
            None
        """

        # unpack into a dict
        data = self._unpack(contents, level=level)

        # pretty print it
        pprint.pprint(data)

        # check for destination
        if destination:

            # and print to it
            with open(destination, 'w') as pointer:

                # pretty print to it
                pprint.pprint(data, pointer)

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
            self._print('made {}'.format(folder))

        # or if part of the directory is not found
        except FileNotFoundError:

            # in which case, print error
            self._print('error making {}, file not found error!'.format(folder))

        # or if there are permissions issues
        except PermissionError:

            # in which case, print error
            self._print('error making {}, permissions error!'.format(folder))

        # unless directory already exists
        except FileExistsError:

            # in which case, nevermind
            pass

        return None

    def _match(self, pattern, string):
        """Use regex to find a pattern in a string.

        Arguments:
            pattern: str, regex pattern
            string: str, the source string

        Returns:
            str, the found pattern
        """

        # default element to ''
        element = ''

        # try to
        try:

            # find element
            element = re.findall(pattern, string)[0]

        # unless not found
        except IndexError:

            # in which case, pass
            pass

        return element

    def _move(self, path, folder):
        """Move a file from one location or name to another.

        Arguments:
            path: str, filepath
            folder: name of new folder

        Returns:
            None
        """

        # make folder in case it doesn't exist
        self._make(folder)

        # construct new name
        new = '{}/{}'.format(folder, path.split('/')[-1])

        # move file
        destination = shutil.move(path, new)
        self._print('moved {}\nto {}\n'.format(path, destination))

        return None

    def _name(self, path, name, replacement=None):
        """Move a file from one location or name to another.

        Arguments:
            path: str, filepath
            name: str, new name

        Returns:
            None
        """

        # extract old name and set replacement
        old = path.split('/')[-1]
        replacement = replacement or old

        # construct new name
        new = path.replace(replacement, name)

        # move file
        destination = shutil.move(path, new)
        self._print('renamed {}\nto {}\n'.format(path, destination))

        return None

    def _note(self):
        """Convert the time to a string for file names.

        Arguments:
            None

        Returns:
            str
        """

        # get now string
        now = datetime.datetime.fromtimestamp(self.now).strftime('%Ym%m%dt%H%M%S')

        return now

    def _pad(self, number, length=2):
        """Pad a number by converting to string and zfilling.

        Arguments:
            number: int
            length: length of final number

        Returns:
            str
        """

        # convert to str
        pad = str(number)

        # zfill
        pad = pad.zfill(length)

        return pad

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

        return message

    def _read(self, path):
        """Read an IDL sav file.

        Arguments:
            path: str, path to file

        Returns:
            dict of file contents
        """

        # read file
        contents = scipy.io.readsav(path)

        return contents

    def _search(self, collection, word):
        """Search a list for a member with a word.

        Arguments:
            collection: list of str
            word: str

        Returns:
            list of str, member of list with word in it.
        """

        # get members
        members = [member for member in collection if word in member]

        return members

    def _see(self, directory):
        """See all the paths in a directory.

        Arguments:
            directory: str, directory path

        Returns:
            list of str, the file paths.
        """

        # try to
        try:

            # make paths
            paths = ['{}/{}'.format(directory, path) for path in os.listdir(directory)]

        # unless the directory does not exist
        except FileNotFoundError:

            # in which case, alert and return empty list
            self._print('{} does not exist'.format(directory))
            paths = []

        return paths

    def _skim(self, members, maintain=False):
        """Skim off the unique members from a list.

        Arguments:
            members: list
            maintain: boolean: maintain order?

        Returns:
            list
        """

        # if maintaining order
        if maintain:

            # for each member
            uniques = []
            for member in members:

                # if not already in uniques
                if member not in uniques:

                    # add it
                    uniques.append(member)

        # otherwise
        else:

            # trim duplicates and sort
            uniques = list(set(members))
            uniques.sort()

        return uniques

    def _show(self, directory):
        """Show the contents of a directory.

        Arguments:
            directory: str, directory path

        Returns:
            None
        """

        # get paths and sort
        paths = self._see(directory)
        paths.sort()

        # display contents of the directory
        self._tell(paths)

        return None

    def _splay(self, tree, stub='', branches=None):
        """Splay nested tree into branches.

        Arguments:
            tree: dict
            stub: growing member

        Returns:
            list of str
        """

        # set default
        branches = branches or []

        # try to
        try:

            # go through each key
            for field, twig in tree.items():

                # add each branch
                growth = '{}/{}'.format(stub, field)
                branches += self._splay(twig, growth)

        # unless a nod
        except AttributeError:

            # add to branch
            growth = '{}/{}'.format(stub, tree)
            branches += [growth]

        return branches

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

    def _swap(self, originals, replacements):
        """Swap a members in list of originals with those from replacements.

        ArgumentsL
            originals: list
            replacements: list

        Returns:
            list
        """

        # swap in self.labels
        for index, replacement in enumerate(replacements):

            # if the label exists and the index is within
            if replacement and index < len(originals):

                # change label
                originals[index] = replacement

        return originals

    def _tabulate(self, rows, destination, delimiter=','):
        """Create a csv file from a list of records.

        Arguments:
            rows: list of list of strings
            destination: str, file path
            delimiter: str, separation character

        Returns:
            None
        """

        # write rows
        with open(destination, 'w') as pointer:

            # write csv
            csv.writer(pointer, delimiter=delimiter).writerows(rows)

        return None

    def _tape(self, table, delimiter=','):
        """Create rows from a csv fiel.

        Arguments:
            table: str, csv file
            delimiter: delimitation charactr

        Returns:
            list of list of str
        """

        # open the table
        with open(table) as pointer:

            # create rows form reader
            rows = [row for row in csv.reader(pointer, delimiter=delimiter)]

        return rows

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

    def _toss(self, words, source, sink):
        """Copy files matching all keyword criteria from source to sink.

        Arguments:
            words: list of str, the keywords
            source: str, source pathname
            sink: str, sink pathname

        Returns:
            None
        """

        # get paths
        paths = self._see(source)

        # get subset
        subset = [path for path in paths if all([word in self._file(path) for word in words])]

        # for each path
        for path in subset:

            # copy to new directory
            self._copy(path, sink)

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