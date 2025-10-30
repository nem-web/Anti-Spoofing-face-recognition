import importlib

libraries = ['torch', 'torchvision', 'torchaudio']

for lib in libraries:
    try:
        module = importlib.import_module(lib)
        version = getattr(module, '__version__', 'Unknown version')
        print(f"{lib} is installed. Version: {version}")
    except ImportError:
        print(f"{lib} is NOT installed.")
