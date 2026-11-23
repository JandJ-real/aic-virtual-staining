import io,json,sys,tempfile,unittest,zipfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import reproduce
import numpy as np
import torch
from PIL import Image
from infer_io import predict,output_name

class Tiny(torch.nn.Module):
 marker_conditioned=True
 def __init__(self):super().__init__();self.scale=torch.nn.Parameter(torch.ones(()));self.calls=[]
 def forward(self,x,marker_id):
  self.calls.append((tuple(x.shape),marker_id,x.dtype,self.training))
  return x*self.scale+marker_id*.1
class Inputs:
 def input(self,row):return np.full((1,256,256),{'colon':.1,'liver':.2,'stomach':.3}[row['organ']],np.float32)
 def pair(self,row):raise AssertionError('Test target access forbidden')

def rows():
 return [dict(organ=o,subset='test',marker='',target='',target_sha256='',stem='ROI001_0_0',
  dapi=o+'/test/DAPI/ROI001_0_0.png',dapi_sha256='a'*64) for o in ('colon','liver','stomach')]

class InferenceTests(unittest.TestCase):
 def test_real_prediction_routes_one_model_and_native_fp32_inputs(self):
  torch.set_num_threads(1);net=Tiny()
  with tempfile.TemporaryDirectory() as td:
   p=Path(td)/'results.zip';result=predict(net,Inputs(),rows(),'cpu',p)
   self.assertEqual(result['images'],3)
   self.assertEqual(net.calls,[((1,1,256,256),0,torch.float32,False),((1,1,256,256),1,torch.float32,False),((1,1,256,256),0,torch.float32,False)])
   with zipfile.ZipFile(p) as z:
    self.assertEqual(set(z.namelist()),{'results/colon/test/CD68/ROI001_0_0_fake.png','results/liver/test/CD45RO/ROI001_0_0_fake.png','results/stomach/test/CD68/ROI001_0_0_fake.png'})
    for name in z.namelist():
     with Image.open(io.BytesIO(z.read(name))) as image:self.assertEqual((image.size,image.mode),((256,256),'RGB'))
   with self.assertRaises(FileExistsError):predict(net,Inputs(),rows(),'cpu',p)
 def test_nonfinite_prediction_never_publishes_submission(self):
  torch.set_num_threads(1);net=Tiny();net.scale.data.fill_(float('nan'))
  with tempfile.TemporaryDirectory() as td:
   p=Path(td)/'results.zip'
   with self.assertRaises(ValueError):predict(net,Inputs(),rows(),'cpu',p)
   self.assertFalse(p.exists())

if __name__=='__main__':unittest.main()
