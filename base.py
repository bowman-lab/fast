import numpy as np

class base:
    def __init__(self):
        pass

    def __repr__(self):
        return str(self)

    def __str__(self):
        # obtain config keys and values
        keys = self.config.keys()
        values = self.config.values()
        # convert config keys and values to a list, where each element
        # is "key=value,"
        s_inputs = np.array(
            [str("%s=%s,") % (k,v) for (k,v) in list(zip(keys, values))])
        # specify the maximum line length for output
        max_l_len = 79
        # initialize string output and length of first line
        s_out = self.class_name + "("
        l_len = len(s_out)
        # for each line, determine if it will make the output line too
        # long and optionally append it as a new line.
        for l in s_inputs:
            if l_len + len(l) > max_l_len:
                s_out += "\n    " + l + " "
                l_len = 4 + len(l)
            else:
                s_out += l + " "
                l_len += len(l)
        # remove the last comma and space (s_out[:-2]) and cap with a
        # parenthesis
        s_out = s_out[:-2] + ")"
        return s_out

