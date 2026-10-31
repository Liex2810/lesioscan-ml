"""Research-only single-image inference. Predictions are not diagnoses."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from lesioscan.inference import load_checkpoint,predict_image

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--image',required=True)
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    a=p.parse_args();model,ckpt=load_checkpoint(a.checkpoint,a.device)
    print(json.dumps({'warning':'RESEARCH ONLY — NOT A DIAGNOSIS',
                      'ranked_model_scores':predict_image(model,ckpt,a.image,a.device)},indent=2))
if __name__=='__main__':main()
