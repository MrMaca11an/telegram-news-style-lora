"""Check user environment, avoid downloading weights."""
import importlib
import platform
import sys


def main():
    print('python:',platform.python_version())
    for module in ['torch','transformers','trl','peft','datasets','unsloth','unsloth_zoo']:
        try:
            obj=importlib.import_module(module)
            print(module, getattr(obj,'__version__','installed'))
        except Exception as e:
            print(module, 'MISSING/BROKEN:',type(e).__name__,str(e)[:400])
            if module in ('unsloth','unsloth_zoo'): sys.exit(2)
    import torch
    if not torch.cuda.is_available():
        sys.exit('CUDA unavailable')
    print('GPU:',torch.cuda.get_device_name(0))
    print('BF16:',torch.cuda.is_bf16_supported())
    print('VRAM GiB:',round(torch.cuda.get_device_properties(0).total_memory/2**30,2))
    if not torch.cuda.is_bf16_supported(): sys.exit('BF16 required by this recipe')
    x=torch.randn(256,256,device='cuda',dtype=torch.bfloat16)
    print('GPU compute test:',float((x@x).float().abs().mean()))
    del x

if __name__=='__main__': main()
