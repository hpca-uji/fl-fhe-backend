from src.nn.layers.layer import Layer
from src.encryption.factory import HomomorphicEncrytionFactory


class EncryptedLayer(Layer):
    """
    This class is used to represent the crypted weights of a layer of a neural network.

    :param name_layer: the name of the layer
    :param weight: the crypted weights of the layer
    :param he_backend: the homomorphic encryption backend
    """
    def __init__(self, name_layer, weight, he_backend=None):
        super(EncryptedLayer, self).__init__(name_layer, weight)
        self.he_backend = he_backend
        if type(weight) == bytes or hasattr(weight, 'decrypt') or hasattr(weight, 'serialize'):
            # If the weights are already encrypted or if they are bytes (serialized weights)
            self.weight_array = weight

        else:
            # If the weights are not encrypted, we encrypt them with the backend
            if hasattr(weight, 'cpu'):
                weight = weight.cpu().detach().numpy()
            self.weight_array = self.he_backend.encrypt(weight)

    def __add__(self, other):
        """
        This function is used to add the crypted weights of two layers.

        :param other: the other layer object

        :return: the addition result in the form of a crypted layer object
        """
        weights = other.get_weight() if type(other) == EncryptedLayer else other
        if self.he_backend and hasattr(self.he_backend, 'enc_add'):
            res = self.he_backend.enc_add(self.weight_array, weights)
        else:
            res = self.weight_array + weights
        return EncryptedLayer(self.name, res, self.he_backend)

    def __sub__(self, other):
        """
        This function is used to substract the crypted weights of two layers.

        :param other: the other layer object

        :return: the substraction result in the form of a crypted layer object
        """
        weights = other.get_weight() if type(other) == EncryptedLayer else other
        return EncryptedLayer(self.name, self.weight_array - weights, self.he_backend)

    def __mul__(self, other):
        """
        This function is used to multiply the crypted weights of two layers.

        :param other: the other layer object

        :return: the multiplication result in the form of a crypted layer object
        """
        weights = other.get_weight() if type(other) == EncryptedLayer else other
        if self.he_backend and hasattr(self.he_backend, 'enc_mul'):
            res = self.he_backend.enc_mul(self.weight_array, weights)
        else:
            res = self.weight_array * weights
        return EncryptedLayer(self.name, res, self.he_backend)

    def __truediv__(self, other):
        """
        This function is used to divide the crypted weights of two layers.

        :param other: the other layer object

        :return: the division result in the form of a crypted layer object
        """
        try:
            # We try to divide the weights of the layer by the weights of the other layer or by a number
            # It is possible only if the denominator is a number or a tensor of non-crypted weights
            weights = other.get_weight() if type(other) == EncryptedLayer else other
            weights = self.weight_array * (1 / weights)

        except:
            print("Error: the division operator isn't supported by the backend")
            weights = []

        return EncryptedLayer(self.name, weights, self.he_backend)

    def shape(self):
        """
        This function is used to get the shape of the layer.

        :return: the shape of the layer in the form of a tuple
        """
        return self.weight_array.shape

    def sum(self, axis=0):
        """
        This function is used to get the sum of the weights of the layer.

        :param axis: the axis of the sum (default: 0 for the sum of the columns)

        :return: the sum of the weights of the layer in the form of a crypted layer object
        """
        return EncryptedLayer(f"sum_{self.name}", self.weight_array.sum(axis=axis), self.he_backend)

    def mean(self, axis=0):
        """
        This function is used to get the mean of the weights of the layer.

        :param axis: the axis of the mean (default: 0 for the mean of the columns)

        :return: the average of the weights of the layer in the form of a crypted layer object
        """
        weights = self.weight_array.sum(axis=axis) * (1 / self.weight_array.shape[axis])
        return EncryptedLayer(f"sum_{self.name}", weights, self.he_backend)

    def decrypt(self, sk=None):
        """
        This function is used to decrypt the weights of the layer.

        :param sk: the secret key used to decrypt the weights (default: None)

        :return: the decrypted weights of the layer in the form of a list of weights
        """
        if self.he_backend and hasattr(self.he_backend, 'decrypt'):
            decrypted = self.he_backend.decrypt(self.weight_array, sk=sk)
            return decrypted.tolist() if hasattr(decrypted, 'tolist') else decrypted
        return self.weight_array.decrypt(sk).tolist() if sk else self.weight_array.decrypt().tolist()

    def serialize(self):
        """
        This function is used to serialize the weights of the layer.

        :return: the serialized weights of the layer in the form of a dictionary
        """
        return {self.name: self.weight_array.serialize()}
    
def encrypt_weights(weights, he_backend):
    encrypted = []
    for name_layer, weight_array in weights.items():
        if name_layer == 'fc3.weight':
            encrypted.append(EncryptedLayer(name_layer, weight_array, he_backend))
        else:
            encrypted.append(Layer(name_layer, weight_array))
    return encrypted