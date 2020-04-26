import imageio
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
from scipy.linalg import eig
from skimage.morphology import thin

#全局参数
#GLOB = {
#    'MIN_LEN' : 20,
#    'MIN_COVER_ANGLE' : 150,
#    'MIN_COVER_RHO1' : 0.3,
#    'MIN_COVER_RHO2' : 0.3,
#}

#用于生成唯一id
class Uid(object):
    def __init__(self):
        self.uid = 0
        
    def get_uid(self):
        
        self.uid+=1
        return self.uid
            
    def reset(self):
        self.uid = 0

#曲线模型
class Path(object):
    def __init__(self,x=None,y=None):
        
        self.uid = uuid.get_uid()
        
        self.complete = False #是否自成环
        
        self.neibors = set() #头尾能连接的path,格式：uid-0[头/尾]-1[头/尾]
        
        if x is not None and y is not None:
            self.xs = [x]
            self.ys = [y]
        
        
    def copy(self):
        p = Path()
        p.xs = self.xs.copy()
        p.ys = self.ys.copy()
        p.angles = self.angles.copy()
        p.omegas = self.omegas.copy()
        return p
    
    def append(self,x,y):
        self.xs.append(x)
        self.ys.append(y)
        
    def cal_angle(self):
        l = self.__len__()
        
        assert(l>2)
        
        self.angles = np.zeros(l,dtype=np.float32)
        self.omegas = np.zeros(l,dtype=np.float32)
        self.alphas = np.zeros(l,dtype=np.float32)
                
        gap = 5
        for i in range(l):
            
            if i<gap:
                x1,y1 = self.xs[min(i+gap,l-1)],self.ys[min(i+gap,l-1)]
                x0,y0 = self.xs[0],self.ys[0]
            elif l-i<=gap:
                x1,y1 = self.xs[-1],self.ys[-1]
                x0,y0 = self.xs[max(0,l-gap)],self.ys[max(0,l-gap)]
            else:
                x1,y1 = self.xs[i+gap],self.ys[i+gap]
                x0,y0 = self.xs[i-gap],self.ys[i-gap]
            
            cangle = np.arctan2(y1-y0,x1-x0)*180/np.pi
            
            if i==0:
                self.omegas[i] = 0
                self.angles[i] = cangle
                dang0 = 0
            else:
                dang = cangle-self.angles[i-1]
            
                if dang>180:
                    dang=dang-360
                elif dang<-180:
                    dang=dang+360
                    
                dang = dang*0.2+dang0*0.8
                dang0 = dang
                 
                self.angles[i] = self.angles[i-1] + dang*0.5
                self.omegas[i] = self.angles[i] - self.angles[i-1]
        
        self.mean_o = np.mean(self.omegas)
                
        self.alphas = np.abs(self.omegas)/np.maximum(np.abs(self.mean_o),1)
        
        mean_a = np.median(self.angles)
        dis_a = np.abs(self.angles-mean_a)
        
        self.line_rho = len(np.where(dis_a<5)[0])/l #直线度
        
        x0,y0,x1,y1 = self.xs[0],self.ys[0],self.xs[-1],self.ys[-1]
        
        if np.sqrt((x1-x0)**2+(y1-y0)**2)<l/8 and abs(self.mean_o)*l>270:
            self.complete = True
    
    def cal_S(self):
        x = np.array(self.xs)
        y = np.array(self.ys)
        
        x2 = x**2
        y2 = y**2
        
        xy = x*y
        
        X = np.stack((x2,xy,y2,x,y,np.ones(len(x))),axis=0) #n*5
        
        self.S = np.matmul(X,X.T)
        
    
    def get_head_tail(self):
        l = self.__len__()
        
        x1,y1 = self.xs[0],self.ys[0]
        x0,y0 = self.xs[min(5,l-1)],self.ys[min(5,l-1)]
            
        hangle = np.arctan2(y1-y0,x1-x0)*180/np.pi
        
        x1,y1 = self.xs[-1],self.ys[-1]
        x0,y0 = self.xs[max(0,l-5)],self.ys[max(0,l-5)]
            
        tangle = np.arctan2(y1-y0,x1-x0)*180/np.pi
        
        
        head = {'x':self.xs[0],
                'y':self.ys[0],
                'angle':hangle,
                'omega':-self.mean_o*l}
        
        tail = {'x':self.xs[-1],
                'y':self.ys[-1],
                'angle':tangle,
                'omega':self.mean_o*l}
        
        return head,tail
    
        
    def concatenate(self,path2):
        path2.xs.reverse()
        path2.ys.reverse()
        self.xs = path2.xs[:-1]+self.xs
        self.ys = path2.ys[:-1]+self.ys
        
    def iter_xy(self):
        for x,y in zip(path.xs,path.ys):
            yield x,y
    
    
    def __len__(self):
        assert len(self.xs)==len(self.ys)
        return len(self.xs)

#从一个点x,y为起点识别出一条连续的曲线，不能有分叉
def segment(im,label,l_id,x,y):
       
    label[y,x] = l_id
    h,w = im.shape    
          
    dirc = np.array([[1,1,1,0,0,-1,-1,-1],
                    [0,-1,1,-1,1,0,-1,1]])
    
    
    cdd_num = 0
    cdd_pt = []
    for i in range(8):
        x1,y1 = x+dirc[0,i],y+dirc[1,i]
        
        if not (x1>=0 and y1>=0 and y1<h and x1<w):
            continue

        if im[y1,x1]:
            cdd_pt.append((x1,y1))
            cdd_num+=1
            
        if cdd_num>2:
            break
            
    if cdd_num>2 or cdd_num==0:
        return None
    
    paths = []
    
    for x1,y1 in cdd_pt:
        if label[y1,x1]==0:
            path1 = Path(x,y)
            path1.append(x1,y1)
            label[y1,x1] = l_id

            grow_arc(im,label,l_id,path1)

            paths.append(path1)
        
    if len(paths)==2:
        paths[0].concatenate(paths[1])
        return paths[0]
    elif len(paths)==1:
        return paths[0]
    else:
        return None    

#沿一个方向生长曲线
def grow_arc(im,label,l_id,path):
    
    h,w = im.shape
    
    assert len(path)>1
    
    while True:
        dirc = np.array([[1,1,1,0,0,-1,-1,-1],
                        [0,-1,1,-1,1,0,-1,1]])
        
        x,y = path.xs[-1],path.ys[-1]
        x_,y_ = path.xs[-2],path.ys[-2]        
        cdd_num = 0
        
        for i in range(8):
            x1,y1 = x+dirc[0,i],y+dirc[1,i]
            
            if not (x1>=0 and y1>=0 and y1<h and x1<w):
                continue

            if im[y1,x1] and not (x_==x1 and y_==y1):
                xn,yn = x1,y1
                cdd_num+=1
                
            if cdd_num>1:
                break
                
        if cdd_num!=1: #多于一条可行路径,或没有可行路径
            break
        elif label[yn,xn]==0:
            path.append(xn,yn)
            label[yn,xn] = l_id
        else:
            break

#将曲线在拐点断开成两条曲线
#p_ori:Path
def cut_inflexion(p_ori):
    paths = []

    peaks, _ = find_peaks(np.abs(p_ori.alphas),height=2,distance=20)
    
    if len(peaks)==0:
        return None
    
    peaks = np.append(peaks, len(p_ori))
    
    id1 = 0
    for id2 in peaks:
        id2-=4
        if id2-id1<=10:
            id1 = id2
            continue
        
        path = Path()
        path.xs = p_ori.xs[id1:id2]
        path.ys = p_ori.ys[id1:id2]
        path.cal_angle()
        
        if path.line_rho<0.9 or len(path)<30:
            paths.append(path)
        
        id1 = id2+2

        
    return paths

#将覆盖角度大于360的曲线断开
def cut_over_spiral(p_ori):
    paths = []
    l = len(p_ori)
    
    cover_angle = abs(p_ori.mean_o)*l
    if cover_angle<360:
        return None
    
    if cover_angle<720:
        l_cut = int((cover_angle-360)/abs(p_ori.mean_o))
        cuts = [l_cut,l-l_cut,l]
    else:
        l_cut = int(l/cover_angle*360)
        cuts = list(range(l_cut,l+l_cut,l_cut))
        cuts[-1] = min(l,cust[-1])
    
    
    id1 = 0
    for id2 in cuts:
    
        if id2-id1<=10:
            id1 = id2
            continue
        
        path = Path()
        path.xs = p_ori.xs[id1:id2]
        path.ys = p_ori.ys[id1:id2]
        path.cal_angle()
        
        if path.line_rho<0.9 or len(path)<30:
            paths.append(path)
        
        id1 = id2

        
    return paths

#判断曲线p1和p2是否能够拼接起来
#p1:Path
#p2:Path
def extend_path(p1,p2):
        
    ht1 = list(p1.get_head_tail())
    ht2 = list(p2.get_head_tail())
    
    len_thr = max(min(len(p1),len(p2)),10)
        
    tags = ['h','t']
    
    for i in range(2):
        for j in range(2):
            e1,e2 = ht1[i],ht2[j]
            
            if p1.uid==37 and p2.uid==136 and i==1 and j==0:
                a=1
            
            res = judge_connect(e1,e2,len_thr)
            
            if res is not None:
                p1.neibors.add(f'{tags[i]}:{tags[j]}:{p2.uid}:{res[0]}:{res[2]}:{len(p2)}') #记录自己是用头/尾连接的下一条路径
                p2.neibors.add(f'{tags[j]}:{tags[i]}:{p1.uid}:{res[1]}:{res[2]}:{len(p1)}')

                return True
    
#判断两个端点是否能够拼接
def judge_connect(pt1,pt2,len_thr):
    x1,y1,a1,o1 = pt1['x'],pt1['y'],pt1['angle'],pt1['omega']
    x2,y2,a2,o2 = pt2['x'],pt2['y'],pt2['angle'],pt2['omega']
    
    dis = np.sqrt((x2-x1)**2+(y2-y1)**2)
    
    angle_thr = 45
    
    if o1*o2>0:
        return None
    
    #空缺距离不能太大
    if dis>len_thr:
        return None
    
    #如果两点距离很近则不再校验连线和路径两端的夹角
    if (dis<=2):
        a2 = a2+180 if a2<0 else a2-180
        d12 = abs(a1-a2)
        d12 = 360-d12 if d12>180 else d12
        return (int(d12),int(d12),int(dis))
    
    a12 = np.arctan2(y2-y1,x2-x1)*180/np.pi
    a21 = a12-180 if a12>0 else a12+180
    
    da1 = abs(a12-a1)
    da1 = 360-da1 if da1>180 else da1
    
    da2 = abs(a21-a2)
    da2 = 360-da2 if da2>180 else da2
    
    if da1>angle_thr or da2>angle_thr:
        return None
    
    return (int(da1),int(da2),int(dis))

#对多种拼接方案排列，只选择最佳的拼接方案
#p:Path
def rank_neibor(p):
    best_h = ""
    best_t = ""
    best_hl = 0
    best_tl = 0
    for nb in p.neibors:
        tag1,tag2,uid2,da,dis,l2 = nb.split(":")
        
        l2 = float(l2)/max(float(dis),1)
        
        if l2>best_hl and tag1=='h':
            best_h = f'{tag1}:{tag2}:{uid2}'
            best_hl = l2
        elif l2>best_tl and tag1=='t':
            best_t = f'{tag1}:{tag2}:{uid2}'
            best_tl = l2
    
    p.neibors = []
    if best_h!="":
        p.neibors.append(best_h)
        
    if best_t!="":
        p.neibors.append(best_t)
    
#将曲线组合转为唯一的可区分的Id
def group2uid(group):
    g_list = [uid for uid in group.split(':') if uid.isdigit()]
    g_list.sort()
    
    return '-'.join(g_list)

#将曲线连接成可能的组合，中间组合也保留
#t:8:h->t:9:h->h:10:t->t:11:h
#consume：记录已经出现的组合
def find_groups(group,consume,paths):
    groups1 = []
    #groups2 = []
    
    grp_uids = [g for g in group.split(':') if g.isdigit()]
    
    
    #从尾端连接
    _,cuid,tag = group.split('->')[-1].split(':')

    ctail = paths[int(cuid)]

    for p in ctail.neibors:

        source,end1,uid = p.split(':')

        if (source==tag) and (uid not in grp_uids):
            end2 = 'h' if end1=='t' else 't'
            g = f'{group}->{end1}:{uid}:{end2}'

            if g not in consume:
                consume.add(g)
                groups1.append(g)
                gs = find_groups(g,consume,paths)
                groups1 += gs
    
    
    return groups1

#将group转成Path list
def group2list(g,paths):
    return [paths[int(uid)] for uid in g.split(':') if uid.isdigit()]
    
#计算每个组覆盖的角度范围和长度
def cangle_length(g,paths):
    
    cangle = 0
    length = 0
    for pstr in g.split('->'):
        
        tag1,uid,tag2 = pstr.split(':')
        
        uid = int(uid)
        
        path = paths[uid]
        
        ll = len(path)
        length+=ll
        
        oa = path.mean_o*ll
                
        if tag1=='h' and tag2=='t':
            cangle+=oa
        else:
            cangle-=oa
            
    return abs(cangle),length


#计算拟合的椭圆和原始点的贴合程度
def ellipse_dis(x,y,a,b,theta,xs,ys):
    xs = np.array(xs)-x
    ys = np.array(ys)-y
    
    ell = np.stack((xs,ys),axis=0)
    
    alpha = -theta*np.pi/180
    
    R_rot = np.array([[np.cos(alpha) , -np.sin(alpha)],[np.sin(alpha) , np.cos(alpha)]])  
         
    ell = np.dot(R_rot,ell) #反向补偿
    
    assert b>0
    
    ell[1,:] = ell[1,:]*a/b
    
    dis = np.abs(np.sqrt(np.sum(ell**2,axis=0))-a)
    
    rho = len(np.where(dis<1)[0])/len(dis) #误差小于两个像素的点所占比例
    
    return np.mean(dis),rho,dis
    
#从椭圆通用表达式解算椭圆标准参数
#A*x.^2 + B*x.*y + C*y.^2 + D*x + E*y + 1
def solve_ellipse(A,B,C,D,E,F):
    
        
    Xc = (B*E-2*C*D)/(4*A*C-B**2)
    Yc = (B*D-2*A*E)/(4*A*C-B**2)
        
    FA1 = 2*(A*Xc**2+C*Yc**2+B*Xc*Yc-F)
    FA2 = np.sqrt((A-C)**2+B**2)
    
    MA = np.sqrt(FA1/(A+C+FA2)) #长轴
    SMA= np.sqrt(FA1/(A+C-FA2)) if A+C-FA2!=0 else 0#半长轴
    
    if B==0 and F*A<F*C:
        Theta = 0
    elif B==0 and F*A>=F*C:
        Theta = 90
    elif B!=0 and F*A<F*C:
        alpha = np.arctan((A-C)/B)*180/np.pi
        Theta = 0.5*(-90-alpha) if alpha<0 else 0.5*(90-alpha)
    else:
        alpha = np.arctan((A-C)/B)*180/np.pi
        Theta = 90+0.5*(-90-alpha) if alpha<0 else 90+0.5*(90-alpha)
    
        
    if MA<SMA:
        MA,SMA = SMA,MA
    
    return [Xc,Yc,MA,SMA,Theta]

#拟合椭圆
#grp:group
#width:图片宽度，用于排除不合理的椭圆长短轴
def fit_ellipse(grp,width,glob):
    S = 0
    xs = []
    ys = []
    for p in grp['paths']:
        S+=p.S
        xs+=p.xs
        ys+=p.ys
        
    H = np.zeros((6,6))
    H[0,2] = 2
    H[2,0] = 2
    H[1,1] = -1
    
    Ls,Vs = eig(H,S,overwrite_a=True,overwrite_b=False)
    
    re = dict()
    
    err0=1e10
    
    for i in range(6):
        mu = np.matmul(Vs[:,i].T,H).dot(Vs[:,i])
        if Ls[i]>0 and mu>0:
            u = np.sqrt(1/mu)
            w = u*Vs[:,i]
            
            skip = False
            w_new = []
            for vv in w:
                if isinstance(vv, complex) and vv.imag!=0:
                    skip = True
                    print(vv)
                    break
                    
                if isinstance(vv, complex):
                    w_new.append(vv.real)
                else:
                    w_new.append(vv)
                    
            if skip:
                continue
                
            A,B,C,D,E,F = w_new
            
            det = np.array([[A,B/2,D/2],
                            [B/2,C,E/2],
                            [D/2,E/2,F]])
            
            #判别是否椭圆
            if F==0 or B**2>=4*A*C or np.linalg.det(det)==0:
                continue
            
            params = solve_ellipse(A,B,C,D,E,F)
            
            if params is None:
                continue
                
            Xc,Yc,MA,SMA,Theta = params
            
            #排除畸形椭圆
            if abs(Xc)>2*width or abs(Yc)>2*width or MA<=2 or SMA<=2 or MA>2*width or SMA>2*width or MA/SMA>3:
                continue
            
            #拟合误差
            e = np.matmul(w.T,S).dot(w)
            
            if e<err0:
                err0 = e
                re['mean_dis'],re['rho'],re['dises'] = ellipse_dis(*params,xs,ys)
                
                perimeter = 2*np.pi*SMA+4*(MA-SMA)
                
                re['cover_rho'] = re['rho']*grp['length']/perimeter
                re['param'] = params
                re['group'] = grp 
                
    if err0>1e9 or re['cover_rho']<glob['MIN_COVER_RHO2'] or re['rho']<glob['MIN_COVER_RHO1']:
        return None
    else:
        return re
    
def draw_ellipse(x,y,a,b,o):
    o = o*np.pi/180

    t = np.linspace(0, 2*np.pi, 100)
    Ell = np.array([a*np.cos(t) , b*np.sin(t)])  
         
    R_rot = np.array([[np.cos(o) , -np.sin(o)],[np.sin(o) , np.cos(o)]])  
    
    Ell = np.dot(R_rot,Ell)
    
    Ell = Ell + np.array([[x],[y]])
        
    return Ell
    

def detect_ellipses(file_path,glob):
    #读取图片
    im = imageio.imread(file_path)
    return detect_ellipses_im(im,glob)

def detect_ellipses_im(im,glob):
    img = thin(im)

    h,w = img.shape

    
    #用于储存所有曲线
    paths = dict()

    #---------------------------生长得到所有曲线-------------------------------
    label = np.zeros_like(img,dtype=np.int)
    l_id = 0

    for y in range(h):
        for x in range(w):
            if img[y,x] and label[y,x]==0:
                l_id+=1
                path = segment(img,label,l_id,x,y)
                if path is None or len(path)<=max(glob['MIN_LEN'],2):
                    continue

                path.cal_angle()
                if path.line_rho<0.9:
                    paths[path.uid]=path


    #--------------------------从拐点断开曲线-----------------------------------
    keys = list(paths.keys())

    for k in keys:
        paths_new = cut_inflexion(paths[k])

        if paths_new is not None and len(paths_new)>0:

            path = paths.pop(k)

            for p in paths_new:
                paths[p.uid]=p
                
                
    #--------------------------将旋转大于360的曲线断开----------------------------
    keys = list(paths.keys())


    for k in keys:
        paths_new = cut_over_spiral(paths[k])

        if paths_new is not None and len(paths_new)>0:

            path = paths.pop(k)

            for p in paths_new:
                paths[p.uid]=p



    #-------------------------将曲线建立邻接关系--------------------------------- 
    path_list = [p for _,p in paths.items()]

    for p in path_list:
        p.neibors = set()

    for i in range(len(path_list)):
        for j in range(i+1,len(path_list),1):
            p1,p2 = path_list[i],path_list[j]
            if (not p1.complete) and (not p2.complete):
                extend_path(p1,p2)
    
    
    #------------------------排列并选择最优的邻居---------------------------------
    for k in paths:
        rank_neibor(paths[k])


    #-----------------------将曲线连接起来形成曲线组------------------------------
    groups = []
    consume = set()
    for p in path_list:

        if p.uid==131:
            a = 1

        if len(p.neibors)==0:
            pstr = f'h:{p.uid}:t'
            groups.append(f'h:{p.uid}:t')

        else:
            pstr = f'h:{p.uid}:t'
            groups.append(pstr)

            gs =  find_groups(f'h:{p.uid}:t',consume,paths)
            groups += gs

            gs =  find_groups(f't:{p.uid}:h',consume,paths)
            groups += gs

    #去除重复的组合
    groups_t = groups
    groups = []
    consume = set()
    for g in groups_t:
        guid = group2uid(g)
        if guid not in consume:
            groups.append(g)
            consume.add(guid)


    #-------------------------------对每个组进行覆盖角度和长度的分析-----------------------------------
    groups_info = []
    for g in groups:

        cangle,length = cangle_length(g,paths)
        if (cangle>glob['MIN_COVER_ANGLE'] and cangle<400 and length>glob['MIN_LEN']):

            groups_info.append({'paths':group2list(g,paths),
                                'group':g,
                                'cover':cangle,
                                'length':length})



    #-----------------------------拟合所有椭圆-------------------------------------------
    for uid in paths:
        paths[uid].cal_S()


    ellipses = []

    h,w = img.shape

    width = max(h,w)

    for g in groups_info:

        if '77' in g['group']:
            a=1

        re = fit_ellipse(g,width,glob)
        if re is not None:
            ellipses.append(re)

    #按覆盖率排序
    ellipses = sorted(ellipses, key=lambda x:x['cover_rho'])
    ellipses.reverse()


    #----------------------------竞争椭圆，覆盖率高的占有曲线---------------------------------
    consume = set()
    ell_valid = []

    for ell in ellipses:
        valid = True
        for p in ell['group']['paths']:
            if p in consume:
                valid = False

        if valid:
            consume = consume | set(ell['group']['paths'])
            ell_valid.append(ell)

    
    return [e['param'] for e in ell_valid]


uuid = Uid()