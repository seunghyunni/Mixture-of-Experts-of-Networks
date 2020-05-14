from data.constants import *


def resolve_data_config(model, verbose=True):
    new_config = {}
    

    # Resolve input/image size
    # FIXME grayscale/chans arg to use different # channels?
    in_chans = 3
    input_size = (in_chans, 224, 224)

    new_config['input_size'] = input_size

    # resolve interpolation method
    new_config['interpolation'] = 'bicubic'
    
    # resolve dataset + model mean for normalization
    
    new_config['mean'] = IMAGENET_DEFAULT_MEAN

    # resolve dataset + model std deviation for normalization
    
    new_config['std'] = IMAGENET_DEFAULT_STD

    # resolve default crop percentage
    new_config['crop_pct'] = DEFAULT_CROP_PCT
    
    if verbose:
        print('Data processing configuration for current model + dataset:')
        for n, v in new_config.items():
            print('\t%s: %s' % (n, str(v)))

    return new_config


def get_mean_by_name(name):
    if name == 'dpn':
        return IMAGENET_DPN_MEAN
    elif name == 'inception' or name == 'le':
        return IMAGENET_INCEPTION_MEAN
    else:
        return IMAGENET_DEFAULT_MEAN


def get_std_by_name(name):
    if name == 'dpn':
        return IMAGENET_DPN_STD
    elif name == 'inception' or name == 'le':
        return IMAGENET_INCEPTION_STD
    else:
        return IMAGENET_DEFAULT_STD


def get_mean_by_model(model_name):
    model_name = model_name.lower()
    if 'dpn' in model_name:
        return IMAGENET_DPN_STD
    elif 'ception' in model_name or 'nasnet' in model_name:
        return IMAGENET_INCEPTION_MEAN
    else:
        return IMAGENET_DEFAULT_MEAN


def get_std_by_model(model_name):
    model_name = model_name.lower()
    if 'dpn' in model_name:
        return IMAGENET_DEFAULT_STD
    elif 'ception' in model_name or 'nasnet' in model_name:
        return IMAGENET_INCEPTION_STD
    else:
        return IMAGENET_DEFAULT_STD
