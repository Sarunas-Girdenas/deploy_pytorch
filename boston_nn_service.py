import torch
from torch.autograd import Variable
import mxnet as mx
import numpy as np
import os
import json

from mms.model_service.mxnet_model_service import MXNetBaseService

class BostonNNService(MXNetBaseService):
    """NN service class
    """
    def __init__(self, model_name, model_dir, manifest, gpu=None):
        self.model_name = model_name
        self.ctx = mx.gpu(int(gpu)) if gpu is not None else mx.cpu()
        signature_file_path = os.path.join(model_dir, manifest['Model']['Signature'])
        if not os.path.isfile(signature_file_path):
            raise RuntimeError('Signature file is not found. Please put signature.json '
                               'into the model file directory...' + signature_file_path)
        try:
            signature_file = open(signature_file_path)
            self._signature = json.load(signature_file)
        except:
            raise Exception('Failed to open model signiture file: %s' % signature_file_path)

        data_names = []
        data_shapes = []
        for input in self._signature['inputs']:
            data_names.append(input['data_name'])
            # Replace 0 entry in data shape with 1 for binding executor.
            # Set batch size as 1
            data_shape = input['data_shape']
            data_shape[0] = 1
            for idx in range(len(data_shape)):
                if data_shape[idx] == 0:
                    data_shape[idx] = 1
            data_shapes.append((input['data_name'], tuple(data_shape)))

        # Load MXNet module
        epoch = 0
        try:
            param_filename = manifest['Model']['Parameters']
            epoch = int(param_filename[len(model_name) + 1: -len('.params')])
        except Exception as e:
            logger.warning('Failed to parse epoch from param file, setting epoch to 0')

        sym, arg_params, aux_params = mx.model.load_checkpoint('%s/%s' % (model_dir, manifest['Model']['Symbol'][:-12]), epoch)
        self.mx_model = mx.mod.Module(symbol=sym, context=self.ctx,
                                      data_names=data_names, label_names=None)
        self.mx_model.bind(for_training=False, data_shapes=data_shapes)
        self.mx_model.set_params(arg_params, aux_params, allow_missing=True)

        # Read synset file
        # If synset is not specified, check whether model archive contains synset file.
        archive_synset = os.path.join(model_dir, 'synset.txt')

        if os.path.isfile(archive_synset):
            synset = archive_synset
            self.labels = [line.strip() for line in open(synset).readlines()]

    def _preprocess(self, data):
        """Could do some preprocessing here
        """
        data = mx.io.NDArrayIter(np.asarray(data))
        return data

    def _inference(self, data):
        pred = self.mx_model.predict(data)
        return pred

    def _postprocess(self, data):
        """Could do some postprocessing here
        """
        # convert to list
        data_out = data.asnumpy()[0]
        data_out = str(data_out)
        return data_out

