import numpy as np, torch, sys
from pathlib import Path
sys.path.insert(0,"analyses/arch2")
from qmi.eval_shots import gen
from clf_arch2 import make_model
r=np.load("runs_eq/pary/loss_0_N100.npz"); names=list(r["names"]); j=names.index("clf_Q_shots_s0")
files=sorted(Path("data_eq/qmi/test/sweep").glob("part_loss_000_*.npz"))
s=torch.load("runs_eq/clf_Q_shots_s0/best.pt",map_location="cpu",weights_only=False)
for dev,dt in (("cuda",torch.float32),("cpu",torch.float32),("cpu",torch.float64)):
    m=make_model("v2").to(dev).to(dt); m.load_state_dict(s["model"]); m.eval()
    C=[];marg=[]
    for f in files:
        _,_,_,Xw,Xq,y=gen((str(f),100,36,"img"))
        with torch.no_grad(): lo=m(torch.from_numpy(Xq/s["scale"]).unsqueeze(1).to(dev).to(dt)).cpu().double().numpy()
        C.append(lo.argmax(1)==y); t=np.sort(lo,1); marg.append(t[:,-1]-t[:,-2])
    C=np.concatenate(C); marg=np.concatenate(marg); bad=np.where(C!=r["correct"][j])[0]
    print(dev,dt,"roznice:",bad,"margines tam:",marg[bad],"min margines:",marg.min())
print("allow_tf32 matmul/cudnn", torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
