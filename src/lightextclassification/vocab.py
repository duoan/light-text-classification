from torchtext.vocab import Vectors


class LocalVectors(Vectors):

  def __init__(self, vector_path, **kwargs):
    super(LocalVectors, self).__init__(vector_path, **kwargs)