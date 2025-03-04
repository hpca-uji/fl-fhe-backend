class Layer:
    """
    This class is used to represent the weights of a layer of a neural network.

    :param name_layer: the name of the layer
    :param weight: the weights of the layer
    """
    def __init__(self, name_layer, weight):
        self.name = name_layer
        self.weight_array = weight

    def get_name(self):
        """
        This function is used to get the name of the layer.
        """
        return self.name

    def get_weight(self):
        """
        This function is used to get the weights of the layer.
        """
        return self.weight_array

    def __add__(self, other):
        """
        This function is used to add the weights of two layers.

        :param other: the other layer object
        """
        weights = other.get_weight() if type(other) == Layer else other
        return Layer(self.name, self.weight_array + weights)

    def __sub__(self, other):
        """
        This function is used to substract the weights of two layers.

        :param other: the other layer object

        :return: the substraction result in the form of a layer object
        """
        weights = other.get_weight() if type(other) == Layer else other
        return Layer(self.name, self.weight_array - weights)

    def __mul__(self, other):
        """
        This function is used to multiply the weights of two layers.

        :param other: the other layer object

        :return: the multiplication result in the form of a layer object
        """
        weights = other.get_weight() if type(other) == Layer else other
        return Layer(self.name, self.weight_array * weights)

    def __truediv__(self, other):
        """
        This function is used to divide the weights of two layers.
        It works only if the weights of the other layer are a number or a layer object (weights with the same shape).

        :param other: the other layer object

        :return: the division result in the form of a layer object
        """
        weights = other.get_weight() if type(other) == Layer else other
        weights = self.weight_array * (1 / weights)
        return Layer(self.name, weights)

    def __len__(self):
        """
        This function is used to get the number of weights of the layer (the size of the layer).
        """
        somme = 1
        for elem in self.weight_array.shape():
            somme *= elem
        return somme

    def shape(self):
        """
        This function is used to get the shape of the layer.
        """
        return self.weight_array.shape()

    def sum(self, axis=0):
        """
        This function is used to get the sum of the weights of the layer.

        :param axis: the axis of the sum (default: 0 for the sum of the columns)

        :return: the sum of the weights of the layer in the form of a layer object
        """
        return Layer(f"sum_{self.name}", self.weight_array.sum(axis=axis))

    def mean(self, axis=0):
        """
        This function is used to get the mean of the weights of the layer.

        :param axis: the axis of the mean (default: 0 for the mean of the columns)

        :return: the mean of the weights of the layer in the form of a layer object
        """
        weights = self.weight_array.sum(axis=axis) * (1 / self.weight_array.shape[axis])
        return Layer(f"sum_{self.name}", weights)

    def decrypt(self, sk=None):
        """
        This function is used to decrypt the weights of the layer.

        :param sk: the secret key used to decrypt the weights (default: None).
        Not used in this class but this is to respect the same structure as the CryptedLayer class.

        :return: the decrypted weights of the layer in the form of a list of weights
        """
        return self.weight_array.tolist()

    def serialize(self):
        """
        This function is used to serialize the weights of the layer.

        :return: the serialized weights of the layer in the form of a dictionary
        with the name of the layer as key and the weights as value
        """

        return {self.name: self.weight_array}