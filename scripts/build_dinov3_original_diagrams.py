"""Draw implementation diagrams only; no empirical result charts are generated."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'docs/figures/dinov3_original';OUT.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','savefig.facecolor':'white'})
P='#7c3aed';B='#2563eb';G='#047857';INK='#172033'

def canvas(w,h,title):
 fig,ax=plt.subplots(figsize=(w,h));ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
 ax.text(.5,1.025,title,ha='center',fontsize=18,weight='bold',color=INK)
 return fig,ax

def box(ax,x,y,w,h,title,detail,color=P):
 ax.add_patch(FancyBboxPatch((x-w/2,y-h/2),w,h,boxstyle='round,pad=0.008,rounding_size=0.015',edgecolor=color,facecolor=color+'0c',linewidth=1.5))
 ax.text(x,y+h*.25,title,ha='center',va='center',weight='bold',fontsize=12,color=INK)
 ax.text(x,y-h*.23,detail,ha='center',va='center',fontsize=10,color='#475569',linespacing=1.6)

def arrow(ax,a,b):ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'-|>','color':'#64748b','lw':1.5})

def save(fig,name):
 for suffix in ['png','svg']:fig.savefig(OUT/f'{name}.{suffix}',dpi=170,bbox_inches='tight')
 plt.close(fig)

fig,ax=canvas(12,11,'CA-TCC–DINOv3: audio features to predictions')
for y,title,detail,color in [(.92,'Audio frontend','Mono 16 kHz → log-mel → training-set normalization',B),(.78,'Numerical image adapter','Clip ±4 → [0,1] → resize → 3 channels → RGB normalization',B),(.64,'DINOv3 ViT-S/16','Patch tokens [B,4×L,384]; no CLS/register output',P),(.50,'Frequency pooling + learned projection','Mean over 4 frequency rows → Conv1d 384→128 → [B,128,L]',P)]:box(ax,.5,y,.72,.095,title,detail,color)
for y in [.92,.78,.64]:arrow(ax,(.5,y-.055),(.5,y-.085))
box(ax,.26,.30,.43,.12,'Temporal/contextual path','Sequence Transformer + future prediction\n32-D contextual vectors',G)
box(ax,.74,.30,.43,.12,'Classification path','Flatten [B,128,L] → linear classifier\nLogits [B,35] or [B,8]',B)
arrow(ax,(.4,.445),(.26,.37));arrow(ax,(.6,.445),(.74,.37))
box(ax,.26,.115,.43,.1,'Contrastive training','Temporal + NT-Xent / original SupCon',G)
box(ax,.74,.115,.43,.1,'Fine-tuning / inference','Cross-Entropy for training; softmax for display',B)
arrow(ax,(.26,.23),(.26,.175));arrow(ax,(.74,.23),(.74,.175))
ax.text(.5,.005,'Speech Commands: L=15, C=35  |  RAVDESS: L=77, C=8\nImplementation diagram; not an experimental result.',ha='center',fontsize=10,color='#64748b');save(fig,'architecture')

# Keep the training pipeline in the separately maintained monochrome style.
from build_dinov3_training_pipeline import main as build_training_pipeline
build_training_pipeline()

# Use the same monochrome style as the training pipeline.
from build_dinov3_loss_paths import main as build_loss_paths
build_loss_paths()

print('Generated three implementation diagrams only; no result charts.')
