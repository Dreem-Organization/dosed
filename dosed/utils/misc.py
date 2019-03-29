class Compose(object):
    """ From torchvision, with love."""

    def __init__(self, transformations):
        self.transformations = transformations

    def __call__(self, x):
        for transformation in self.transformations:
            x = transformation(x)
        return x

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transformations:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string
