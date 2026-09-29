import sys
from pathlib import Path


def main():
    import torch
    import nemo
    from nemo.collections.asr.models import SortformerEncLabelModel
    print('Python:', sys.version.split()[0])
    print('PyTorch:', torch.__version__)
    print('PyTorch CUDA:', torch.version.cuda)
    print('CUDA available:', torch.cuda.is_available())
    print('NeMo source:', nemo.__file__)
    print('Sortformer:', SortformerEncLabelModel.__name__)
    expected = Path(__file__).resolve().parents[1] / 'third_party/Speech'
    actual = Path(nemo.__file__).resolve()
    if expected.resolve() not in actual.parents:
        raise RuntimeError('NeMo was not imported from third_party/Speech. Check the activated environment.')
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
    else:
        print('GPU inference unavailable in this process; check driver / Slurm GPU allocation.')


if __name__ == '__main__':
    main()
