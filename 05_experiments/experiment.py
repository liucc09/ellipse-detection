from shapely.geometry.point import Point
from shapely import affinity
from matplotlib.patches import Polygon
from shapely.ops import unary_union
import numpy as np

def create_ellipse(x,y,a,b,theta,scale):
    """
    create a shapely ellipse. adapted from
    https://gis.stackexchange.com/a/243462
    """
    x,y,a,b = x*scale,y*scale,a*scale,b*scale
    circ = Point((int(x),int(y))).buffer(1)
    ell =  affinity.scale(circ, int(a), int(b))
    ellr = affinity.rotate(ell, theta)
    return ellr

def draw_ellipse(params,ax,color='blue',alpha=0.5,wire=False):
    for param in params:
        ellipse1 = create_ellipse(*param,100)
        verts1 = np.array(ellipse1.exterior.coords.xy)/100
        if wire:
            ax.plot(verts1[0,:],verts1[1,:],color=color,linewidth=2.0)
        else:
            patch1 = Polygon(verts1.T, color = color, alpha = alpha)
            ax.add_patch(patch1)

def cal_ratio(param1,param2):
    if param1[2]==0 or param1[3]==0 or param2[2]==0 or param2[3]==0:
        return 0
    
    ellipse1 = create_ellipse(*param1,100)
    ellipse2 = create_ellipse(*param2,100)
    
    ##the intersect will be outlined in black
    intersect = ellipse1.intersection(ellipse2)
    union = unary_union([ellipse1,ellipse2])
    
    return intersect.area/union.area
    
def cal_ratio_all(params1,params2):
    ratio1 = dict()
    ratio2 = dict()
    
    for i in range(len(params1)):
        for j in range(len(params2)):
            rr = cal_ratio(params1[i],params2[j])
            
            if rr>ratio1.get(i,0) and rr>ratio2.get(j,0):
                ratio1[i] = rr
                ratio2[j] = rr
            else:
                ratio1[i] = ratio1.get(i,0)
                ratio2[j] = ratio2.get(j,0)
            
    return ratio1,ratio2

def cal_metrics(prs,gts,threshold=0.7):
    TP = 0 #正确拟合的椭圆
    PN = 0 #预测的总椭圆数
    GN = 0 #真实的总椭圆数

    pr_ratio = [] #记录每个已识别椭圆的覆盖率
    gt_ratio = [] #记录每个目标椭圆的覆盖率

    for (pr,gt) in zip(prs,gts):
        r1,r2 = cal_ratio_all(pr,gt)

        PN+= len(pr)
        GN+= len(gt)

        TP+= len([k for k in r2 if r2[k]>threshold])
    
        pr_ratio.append(r1)
        gt_ratio.append(r2)

    precision = TP/PN if PN>0 else 0
    recall = TP/GN if GN>0 else 0
    if precision==0:
        F1 = 0
    else:
        F1 = 2*precision*recall/(precision+recall)
    
    return precision,recall,F1,pr_ratio,gt_ratio