"""Download official, pinned DPDFNet ONNX files into this experiment only."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent
REPOSITORY = 'Ceva-IP/DPDFNet'
REVISION = 'dd6818d00f50c836fed43a6243ebe49116de5964'
MODELS = ('dpdfnet2','dpdfnet4')


def download(models):
    destination = ROOT/'models'
    destination.mkdir(exist_ok=True)
    metadata_path = destination/'upstream_metadata.json'
    subprocess.run(['curl','-fsSL','--retry','2',
        f'https://huggingface.co/api/models/{REPOSITORY}/revision/{REVISION}?blobs=true',
        '-o',str(metadata_path)],check=True)
    metadata = json.loads(metadata_path.read_text())
    assert metadata['sha'] == REVISION
    upstream = {item['rfilename']:item for item in metadata['siblings']}
    manifest_path = ROOT/'model_manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {
        'repository':REPOSITORY,'revision':REVISION,'models':{}}
    for model in models:
        relative = f'onnx/{model}.onnx'
        expected = upstream[relative]['lfs']['sha256']
        path = destination/f'{model}.onnx'
        actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
        if actual != expected:
            temporary = path.with_suffix('.onnx.part')
            print(f'Downloading {model} from official pinned revision...',flush=True)
            subprocess.run(['curl','-fL','--retry','2',
                f'https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/{relative}',
                '-o',str(temporary)],check=True)
            actual = hashlib.sha256(temporary.read_bytes()).hexdigest()
            if actual != expected:
                raise RuntimeError(f'SHA256 mismatch for {model}; download not installed')
            temporary.replace(path)
        manifest['models'][model] = {'file':str(path.relative_to(ROOT)),
            'sha256':actual,'bytes':path.stat().st_size}
        print(f'Verified {model}: {actual}',flush=True)
        manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models',nargs='+',choices=MODELS,default=list(MODELS))
    download(parser.parse_args().models)
