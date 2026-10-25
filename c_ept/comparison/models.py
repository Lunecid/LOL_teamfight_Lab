"""Explicit architecture registry and float32 policy for the new profile only."""
from __future__ import annotations

import json

ARCHITECTURES = ('ept_s_v1','n2_teamset_v1')


def validate_model_spec(spec):
    if not isinstance(spec,dict) or set(spec) != {'architecture','n2_hidden','prefetch','precision'}:
        raise ValueError('model specification fields mismatch')
    if spec['architecture'] not in ARCHITECTURES or spec['precision'] != 'float32_strict':
        raise ValueError('unsupported comparison architecture/precision')
    if type(spec['n2_hidden']) is not int or not 1 <= spec['n2_hidden'] <= 4096:
        raise ValueError('invalid N2 hidden width')
    if type(spec['prefetch']) is not int or not 0 <= spec['prefetch'] <= 2:
        raise ValueError('prefetch must be bounded by2')
    return json.loads(json.dumps(spec))


def specification(model, *, hidden=512, prefetch=2):
    architecture={'ept':'ept_s_v1','n2':'n2_teamset_v1'}[model]
    return validate_model_spec({'architecture':architecture,'n2_hidden':hidden,
                                'prefetch':prefetch,'precision':'float32_strict'})


def build_model(cfg, static, spec):
    spec=validate_model_spec(spec)
    if spec['architecture']=='ept_s_v1':
        from c_ept.ept.heads import EPT
        return EPT(cfg,static)
    from c_ept.models.neural.team_set import TeamSetModel
    return TeamSetModel(cfg,static,hidden=spec['n2_hidden'])


def supported_model(model):
    from c_ept.ept.heads import EPT
    from c_ept.models.neural.team_set import TeamSetModel
    return isinstance(model,(EPT,TeamSetModel))


def configure_float32():
    import torch
    if torch.get_default_dtype()!=torch.float32:
        raise ValueError('float32 default required')
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)
    return {'dtype':'float32','amp':False,'tf32':False,'matmul_precision':'highest',
            'cudnn_benchmark':False,'deterministic':True}
