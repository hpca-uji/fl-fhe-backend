from src.nn.layers.layer import Layer
import tenseal as ts


class EncryptedLayer(Layer):
    """
    This class is used to represent the crypted weights of a layer of a neural network.

    :param name_layer: the name of the layer
    :param weight: the crypted weights of the layer
    :param contexte: the context of the encryption
    """
    def __init__(self, name_layer, weight, context=None):
        super(EncryptedLayer, self).__init__(name_layer, weight)
        if type(weight) == ts.tensors.CKKSTensor or type(weight) == bytes:
            # If the weights are already encrypted or if they are bytes (serialized weights)
            self.weight_array = weight

        else:
            # If the weights are not encrypted, we encrypt them with the context
            self.weight_array = ts.ckks_tensor(context, weight.cpu().detach().numpy())

    def __add__(self, other):
        """
        This function is used to add the crypted weights of two layers.

        :param other: the other layer object

        :return: the addition result in the form of a crypted layer object
        """
        weights = other.get_weight() if type(other) == EncryptedLayer else other
        return EncryptedLayer(self.name, self.weight_array + weights)

    def __sub__(self, other):
        """
        This function is used to substract the crypted weights of two layers.

        :param other: the other layer object

        :return: the substraction result in the form of a crypted layer object
        """
        weights = other.get_weight() if type(other) == EncryptedLayer else other
        return EncryptedLayer(self.name, self.weight_array - weights)

    def __mul__(self, other):
        """
        This function is used to multiply the crypted weights of two layers.

        :param other: the other layer object

        :return: the multiplication result in the form of a crypted layer object
        """
        weights = other.get_weight() if type(other) == EncryptedLayer else other
        return EncryptedLayer(self.name, self.weight_array * weights)

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
            print("Error: the division operator isn't supported by SEAL")
            weights = []

        return EncryptedLayer(self.name, weights)

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
        return EncryptedLayer(f"sum_{self.name}", self.weight_array.sum(axis=axis))

    def mean(self, axis=0):
        """
        This function is used to get the mean of the weights of the layer.

        :param axis: the axis of the mean (default: 0 for the mean of the columns)

        :return: the average of the weights of the layer in the form of a crypted layer object
        """
        weights = self.weight_array.sum(axis=axis) * (1 / self.weight_array.shape[axis])
        return EncryptedLayer(f"sum_{self.name}", weights)

    def decrypt(self, sk=None):
        """
        This function is used to decrypt the weights of the layer.

        :param sk: the secret key used to decrypt the weights (default: None)

        :return: the decrypted weights of the layer in the form of a list of weights
        """
        return self.weight_array.decrypt(sk).tolist() if sk else self.weight_array.decrypt().tolist()

    def serialize(self):
        """
        This function is used to serialize the weights of the layer.

        :return: the serialized weights of the layer in the form of a dictionary
        """
        return {self.name: self.weight_array.serialize()}
    
def encrypt_weights(weights, enc_context):
    encrypted = []
    for name_layer, weight_array in weights.items():
        if name_layer == 'fc3.weight':
            encrypted.append(EncryptedLayer(name_layer, weight_array, enc_context))
        else:
            encrypted.append(Layer(name_layer, weight_array))
    return encrypted