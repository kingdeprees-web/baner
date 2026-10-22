#!/usr/bin/env python3
"""
ساخت اینترو کمیته از روی عکس بنر (انجمن سازندگان ذی‌صلاح ارومیه)

  python make_intro.py banners/legal.jpg --out videos
  python make_intro.py --all banners --out videos
  python make_intro.py --write-workflow      (برای اجرا روی GitHub Actions)

اگر کنار بنر فایل هم‌نام .txt باشد (مثلاً banners/legal.txt) اسم‌ها از آن خوانده می‌شود:
  عنوان: کارشناسی و حقوقی
  فرشاد موسوی مقدم | لیسانس عمران | کارشناس دادگستری
  ناصر عادلی | کارشناس ارشد عمران | کارشناس دادگستری
در غیر این صورت (اگر tesseract فارسی نصب باشد) اسم‌ها خودکار خوانده می‌شود.
"""
import argparse, base64, glob, hashlib, json, os, shutil, subprocess, sys, tempfile, time
import cv2, numpy as np

RT = None  # پوشه موقت که کد رندر و فایل‌ها در آن پیاده می‌شود


# ---------------------------------------------------------------- تشخیص لوگو و عکس اعضا
def detect(img):
    """img: BGR. خروجی: (logo, members) هرکدام dict با x,y,r بر حسب پیکسل بنر"""
    H, W = img.shape[:2]
    f = max(1, round(W / 556))
    sm = img[f // 2::f, f // 2::f][:, :, ::-1].astype(np.int32)  # RGB
    r, g, b = sm[..., 0], sm[..., 1], sm[..., 2]
    gold = ((r > 140) & (r >= g) & (g > b) & (r - b > 70) & (g > 85)).astype(np.uint8)
    h, w = gold.shape
    n, _, st, _ = cv2.connectedComponentsWithStats(gold, connectivity=8)
    logo, mem = None, []
    for i in range(1, n):
        x0, y0, bw, bh, cnt = st[i]
        sz = (bw + bh) / 2
        if abs(bw - bh) / max(bw, bh) > 0.12 or cnt < sz * 3:
            continue
        cx, cy, rel = (x0 + bw / 2) * f, (y0 + bh / 2) * f, sz / w
        if 0.2 < rel < 0.34 and cy < H * 0.3:
            if logo is None or sz > logo['sz']:
                logo = dict(x=cx, y=cy, r=sz / 2 * f, sz=sz)
        elif 0.10 < rel < 0.19 and cx > W * 0.7:
            mem.append(dict(x=cx, y=cy, r=sz / 2 * f))
    mem.sort(key=lambda m: m['y'])
    if logo:
        mem = [m for m in mem if m['y'] > logo['y'] + logo['r'] * 1.1]
    out = []
    for m in mem:
        if not out or m['y'] - out[-1]['y'] > m['r']:
            out.append(m)
    return logo, out


# ---------------------------------------------------------------- خواندن متن (اختیاری)
def fix_fa(s):
    return s.replace('ي', 'ی').replace('ك', 'ک').replace('|', '').strip()


def ocr_names(img, mem):
    try:
        import pytesseract
        from PIL import Image
        pytesseract.get_languages()
        if 'fas' not in pytesseract.get_languages():
            return None
    except Exception:
        return None
    H, W = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def rd(x, y, w, h, th, psm):
        c = gray[int(max(0, y)):int(y + h), int(x):int(x + w)]
        c = cv2.resize(c, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
        c = np.where(c < th, 0, 255).astype(np.uint8)
        t = pytesseract.image_to_string(c, lang='fas', config=f'--psm {psm}')
        return [fix_fa(z) for z in t.split('\n') if z.strip()]

    title = ' '.join(rd(.07 * W, mem[0]['y'] - .186 * W, .55 * W, .06 * W, 115, 7))
    rows = []
    for m in mem:
        n = rd(.551 * W, m['y'] - .02 * W, .227 * W, .092 * W, 150, 6)
        c = rd(.308 * W, m['y'] - .02 * W, .225 * W, .092 * W, 150, 6)
        role = (n[1] if len(n) > 1 else (c[1] if len(c) > 1 else '')).replace('(', '').replace(')', '').strip()
        rows.append(dict(n=n[0] if n else '', c=c[0] if c else '', r=role))
    return title, rows


def read_txt(path):
    title, rows = '', []
    for line in open(path, encoding='utf-8'):
        line = line.strip()
        if not line:
            continue
        if line.startswith('عنوان:') or line.lower().startswith('title:'):
            title = line.split(':', 1)[1].strip()
            continue
        p = [x.strip() for x in line.split('|')]
        rows.append(dict(n=p[0], c=p[1] if len(p) > 1 else '', r=p[2] if len(p) > 2 else ''))
    return title, rows


# ---------------------------------------------------------------- ساخت ویدیو
def run(cmd, **kw):
    r = subprocess.run(cmd, **kw)
    if r.returncode:
        sys.exit(f'خطا در اجرای: {" ".join(map(str, cmd))}')


def build(banner, out_dir, title_arg, workers):
    global RT
    RT = ensure_runtime()
    name = os.path.splitext(os.path.basename(banner))[0]
    img = cv2.imread(banner)
    if img is None:
        sys.exit(f'نمی‌توانم عکس را باز کنم: {banner}')
    logo, mem = detect(img)
    if not logo or not mem:
        sys.exit(f'[{name}] لوگو یا عکس اعضا پیدا نشد؛ مطمئن شو بنر کمیته با همان قالب انجمن است.')
    print(f'[{name}] {len(mem)} عضو پیدا شد')

    title, rows = '', None
    txt = os.path.splitext(banner)[0] + '.txt'
    if os.path.exists(txt):
        title, rows = read_txt(txt)
        print(f'[{name}] اسم‌ها از {os.path.basename(txt)} خوانده شد')
    if rows is None:
        res = ocr_names(img, mem)
        if res:
            title, rows = res
            print(f'[{name}] اسم‌ها با OCR خوانده شد (حتماً بررسی کن)')
    if rows is None:
        tpl = txt
        with open(tpl, 'w', encoding='utf-8') as f:
            f.write('عنوان: \n' + ''.join('نام | مدرک | توضیح اختیاری\n' for _ in mem))
        sys.exit(f'[{name}] OCR فارسی در دسترس نیست. فایل {tpl} را با اسم‌ها پر کن و دوباره اجرا کن.')
    if title_arg:
        title = title_arg
    rows = (rows + [dict(n='', c='', r='')] * len(mem))[:len(mem)]
    if len(rows) != len(mem):
        print(f'[{name}] هشدار: تعداد اسم‌ها با تعداد عکس‌ها فرق دارد')

    work = tempfile.mkdtemp(prefix='intro_')
    cfg = dict(banner=os.path.abspath(banner), title=title or 'کمیته تخصصی',
               members=rows, circles=mem, logo=logo)
    cfgp = os.path.join(work, 'cfg.json')
    json.dump(cfg, open(cfgp, 'w', encoding='utf-8'), ensure_ascii=False)
    env = dict(os.environ, INTRO_CFG=cfgp)
    eng = os.path.join(RT, 'engine.py')

    info = subprocess.run([sys.executable, eng, 'info'], env=env, capture_output=True, text=True, check=True).stdout
    I = json.loads(info.strip().splitlines()[-1])
    total = int(I['TEND'] * I['FPS'])
    print(f'[{name}] مدت ویدیو {I["TEND"]:.1f} ثانیه ({total} فریم)')

    # رندر موازی قطعه‌ها
    chunk = max(30, total // (workers * 2))
    parts, procs = [], []
    a = 0
    while a < total:
        b = min(total, a + chunk)
        parts.append((a, b, os.path.join(work, f'p_{len(parts):03d}.mp4')))
        a = b
    pending, running = list(parts), []
    while pending or running:
        while pending and len(running) < workers:
            p = pending.pop(0)
            running.append((subprocess.Popen([sys.executable, eng, str(p[0]), str(p[1]), p[2]], env=env), p))
        for pr, p in list(running):
            if pr.poll() is not None:
                if pr.returncode:
                    sys.exit('رندر قطعه ناموفق بود')
                running.remove((pr, p))
                done = sum(1 for q in parts if os.path.exists(q[2]) and q not in [x[1] for x in running] and q not in pending)
                print(f'  رندر {done}/{len(parts)} قطعه')
        time.sleep(0.5)
    lst = os.path.join(work, 'list.txt')
    open(lst, 'w').write(''.join(f"file '{p[2]}'\n" for p in parts))
    silent = os.path.join(work, 'silent.mp4')
    run(['ffmpeg', '-loglevel', 'error', '-y', '-f', 'concat', '-safe', '0', '-i', lst, '-c', 'copy', silent])

    # صدا
    run([sys.executable, os.path.join(RT, 'audio_gen.py'), json.dumps(I), work])
    os.makedirs(out_dir, exist_ok=True)
    for kind, wav, lufs in (('', 'music.wav', -14), ('_NoMusic', 'sfx.wav', -16)):
        outp = os.path.join(out_dir, f'Intro_{name}{kind}.mp4')
        run(['ffmpeg', '-loglevel', 'error', '-y', '-i', silent, '-i', os.path.join(work, wav), '-map', '0:v', '-map', '1:a',
             '-c:v', 'copy', '-c:a', 'aac', '-b:a', '256k', '-af', f'loudnorm=I={lufs}:TP=-1.5:LRA=11', '-shortest', outp])
        print(f'[{name}] ساخته شد: {outp}')
    shutil.rmtree(work, ignore_errors=True)


def ensure_runtime():
    """کد رندر و فایل‌های لازم (که پایین همین فایل هستند) را در پوشه موقت پیاده می‌کند."""
    h = hashlib.md5((SRC_INTRO3 + SRC_ENGINE + SRC_AUDIO).encode()).hexdigest()[:8]
    d = os.path.join(tempfile.gettempdir(), 'urmia_intro_' + h)
    if not os.path.exists(os.path.join(d, '.ok')):
        os.makedirs(os.path.join(d, 'assets'), exist_ok=True)
        for fn, src in (('intro3.py', SRC_INTRO3), ('engine.py', SRC_ENGINE), ('audio_gen.py', SRC_AUDIO)):
            open(os.path.join(d, fn), 'w', encoding='utf-8').write(src)
        open(os.path.join(d, 'assets', 'B-NAZANIN.TTF'), 'wb').write(base64.b64decode(FONT_B64))
        open(os.path.join(d, 'assets', 'logo_source.webp'), 'wb').write(base64.b64decode(LOGO_B64))
        open(os.path.join(d, '.ok'), 'w').write('1')
    return d


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('banner', nargs='?', help='عکس بنر کمیته')
    ap.add_argument('--all', metavar='DIR', help='همه بنرهای یک پوشه')
    ap.add_argument('--out', default='videos')
    ap.add_argument('--title', help='عنوان کمیته (اختیاری)')
    ap.add_argument('--write-workflow', action='store_true', help='فایل GitHub Actions را می‌سازد')
    ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2)))
    a = ap.parse_args()
    if a.write_workflow:
        os.makedirs('.github/workflows', exist_ok=True)
        open('.github/workflows/make-video.yml', 'w').write(WORKFLOW)
        print('ساخته شد: .github/workflows/make-video.yml')
        return
    if a.all:
        files = sorted(p for e in ('jpg', 'jpeg', 'png', 'webp') for p in glob.glob(os.path.join(a.all, f'*.{e}')))
        if not files:
            sys.exit('هیچ بنری در پوشه نیست')
        for f in files:
            build(f, a.out, a.title, a.workers)
    elif a.banner:
        build(a.banner, a.out, a.title, a.workers)
    else:
        ap.print_help()



# ======================================================================
# کد رندر ویدیو (intro3.py) 
# ======================================================================
SRC_INTRO3 = r'''import cv2, numpy as np, math, sys, subprocess, os
from PIL import Image, ImageDraw, ImageFont
W,H,FPS=1080,1920,30
TEND=13.0
rng=np.random.RandomState(11)
def sstep(x): x=np.clip(x,0,1); return x*x*(3-2*x)
def eo3(x): x=np.clip(x,0,1); return 1-(1-x)**3
def eio3(x): x=np.clip(x,0,1); return 4*x**3 if x<.5 else 1-(-2*x+2)**3/2
def eisine(x): x=np.clip(x,0,1); return -(math.cos(math.pi*x)-1)/2
yy,xx=np.mgrid[0:H,0:W].astype(np.float32)
GOLD=np.array([255,205,120],np.float32)
FB='/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'

# ---------------- coin sprite
src=Image.open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'assets','logo_source.webp')).convert('RGB')
cx0,cy0,R0=311.5,308.5,292.5
SD=int(2*R0)+6
x0=int(round(cx0-SD/2)); y0=int(round(cy0-SD/2))
coin=src.crop((x0,y0,x0+SD,y0+SD)).convert('RGBA')
m=Image.new('L',(SD*4,SD*4),0)
ImageDraw.Draw(m).ellipse(((SD/2-R0)*4,(SD/2-R0)*4,(SD/2+R0)*4,(SD/2+R0)*4),fill=255)
coin.putalpha(m.resize((SD,SD),Image.LANCZOS))
coinA=np.array(coin).astype(np.float32); coinA[...,3]/=255.0
gy,gx=np.mgrid[0:SD,0:SD].astype(np.float32)
shine_u=(gx+0.55*gy)/(SD*1.55)
coin_mask=np.where(coinA[...,3]>0,1,0).astype(np.float32)

# ---------------- text
def render_text(txt,size,top,bot):
    font=ImageFont.truetype(FB,size,layout_engine=ImageFont.Layout.RAQM)
    wlen=int(font.getlength(txt,direction='rtl',language='fa'))+size
    hh=int(size*2.2)
    im=Image.new('L',(wlen,hh),0)
    ImageDraw.Draw(im).text((wlen//2,hh//2),txt,font=font,fill=255,anchor='mm',direction='rtl',language='fa')
    a=np.array(im).astype(np.float32)/255
    ys,xs=np.where(a>0.02)
    pad=8
    a=a[max(0,ys.min()-pad):ys.max()+pad, max(0,xs.min()-pad):xs.max()+pad]
    h,w=a.shape
    g=np.linspace(0,1,h)[:,None,None]
    rgb=(np.array(top,np.float32)*(1-g)+np.array(bot,np.float32)*g)*np.ones((1,w,1),np.float32)
    return rgb,a
TXT={}
def prep(name,txt,size,top,bot,col):
    rgb,al=render_text(txt,size,top,bot)
    rgbb=cv2.GaussianBlur(rgb*al[...,None],(0,0),7); alb=cv2.GaussianBlur(al,(0,0),7)
    PADG=80
    glowp=cv2.GaussianBlur(np.pad(al,PADG),(0,0),26)
    TXT[name]=dict(rgb=rgb,al=al,rgbb=rgbb,alb=alb,glowp=glowp,col=np.array(col,np.float32))
GT,GB=(255,232,160),(224,162,58)
WT,WB=(255,255,255),(208,222,238)
prep('label','دوره تخصصی مقدماتی',64,WT,WB,(180,210,255))
prep('t1','بهسازی لرزه‌ای',126,GT,GB,(255,200,100))
prep('t2','سازه',126,GT,GB,(255,200,100))
prep('with','با تدریس',50,WT,WB,(180,210,255))
prep('dr','جناب دکتر فریدونی',88,GT,GB,(255,200,100))
prep('org1','توسط انجمن سازندگان ذی‌صلاح',50,WT,WB,(180,210,255))
prep('org2','ساختمان شهرستان ارومیه',50,WT,WB,(180,210,255))
prep('a1','انجمن سازندگان ذی‌صلاح',60,WT,WB,(180,210,255))
prep('a2','ساختمان شهرستان ارومیه',60,WT,WB,(180,210,255))

def draw_text(cv,name,t,t_in,t_out,cx,cy,dur=1.1,fade_out=0.6):
    T=TXT[name]
    env=sstep((t-t_in)/0.25)*(1-sstep((t-t_out)/fade_out))
    if env<=0.003: return
    p=eo3((t-t_in)/dur)
    h,w=T['al'].shape
    xs=np.arange(w,dtype=np.float32)
    front=(1-p)*(w+110)-110
    mloc=sstep((xs-front)/110)[None,:]
    rise=int(16*(1-eo3((t-t_in)/1.4)))-int(10*sstep((t-t_out)/fade_out))
    x=int(cx-w/2); y=int(cy-h/2)+rise
    ya,yb=max(0,y),min(H,y+h); xa,xb=max(0,x),min(W,x+w)
    if ya>=yb or xa>=xb: return
    sl=(slice(ya-y,yb-y),slice(xa-x,xb-x))
    m_=mloc[:,sl[1]]
    al=T['al'][sl]; rgb=T['rgb'][sl]; alb=T['alb'][sl]; rgbb=T['rgbb'][sl]
    edge=np.clip(1-np.abs(m_-0.5)*2,0,1)
    k=edge*0.9
    a_use=(al*(1-k)+alb*k)*m_*env
    rgb_use=rgb*(1-k[...,None])+(rgbb/np.maximum(alb[...,None],1e-3))*k[...,None]
    xs2=xs[sl[1]][None,:]
    band=np.exp(-((xs2-front-55)/38.0)**2)*(p<0.999)
    rgb_use=np.clip(rgb_use+band[...,None]*np.array([255,235,180],np.float32)*0.7,0,255)
    reg=cv[ya:yb,xa:xb]
    reg[:]=reg*(1-a_use[...,None])+rgb_use*a_use[...,None]
    G=T['glowp']; gh,gw=G.shape; gx_=x-80; gy_=y-80
    ya2,yb2=max(0,gy_),min(H,gy_+gh); xa2,xb2=max(0,gx_),min(W,gx_+gw)
    if ya2<yb2 and xa2<xb2:
        gsl=G[ya2-gy_:yb2-gy_, xa2-gx_:xb2-gx_]
        xs_g=(np.arange(xa2,xb2)-x).astype(np.float32)[None,:]
        mg=sstep((xs_g-front)/110)
        cv[ya2:yb2,xa2:xb2]+=(gsl*mg)[...,None]*T['col']*(0.55*env)

def draw_line(cv,t,t_in,t_out,cy,halfw):
    env=sstep((t-t_in)/0.2)*(1-sstep((t-t_out)/0.5))
    if env<=0.003: return
    p=eo3((t-t_in)/1.2); hw_=int(halfw*p)
    if hw_>4:
        y0_=max(0,cy-16); y1_=min(H,cy+16)
        ov=np.zeros((y1_-y0_,W,3),np.float32); c=cy-y0_
        cv2.line(ov,(540-hw_,c),(540-16,c),(235,190,95),2,cv2.LINE_AA)
        cv2.line(ov,(540+16,c),(540+hw_,c),(235,190,95),2,cv2.LINE_AA)
        pts=np.array([[540,c-10],[550,c],[540,c+10],[530,c]],np.int32)
        cv2.fillPoly(ov,[pts],(245,205,120),cv2.LINE_AA)
        cv[y0_:y1_]+=ov*env+cv2.GaussianBlur(ov,(0,0),5)*0.5*env

# ---------------- sprites / particles
def spr(r):
    s=int(r*6)|1; c=s//2
    y,x=np.mgrid[0:s,0:s]; g=np.exp(-((x-c)**2+(y-c)**2)/(2*(r**2)))
    return g.astype(np.float32)
SPR={r:spr(r) for r in (1.6,2.6,4,22,40)}
def add_sprite(canvas,spr_,x,y,col,a):
    s=spr_.shape[0]; h=s//2; x=int(x); y=int(y)
    xa,ya=max(0,x-h),max(0,y-h); xb,yb=min(W,x-h+s),min(H,y-h+s)
    if xa>=xb or ya>=yb: return
    sp=spr_[ya-(y-h):yb-(y-h), xa-(x-h):xb-(x-h)]
    canvas[ya:yb,xa:xb]+=sp[...,None]*(col*a)
NP=110
pk=[]
for i in range(NP):
    big=i<8
    pk.append(dict(x=rng.uniform(0,W),y=rng.uniform(0,H),v=rng.uniform(8,30)*(0.5 if big else 1),
        r=(40 if i<3 else 22) if big else rng.choice([1.6,2.6,4]),ph=rng.uniform(0,6.28),
        a=(0.12 if big else rng.uniform(0.35,1.0)),dx=rng.uniform(-8,8),big=big))
grain=[cv2.resize((rng.randn(H//2,W//2).astype(np.float32)*5),(W,H))[...,None] for _ in range(6)]
SPK=[(rng.uniform(0,2*math.pi),rng.uniform(90,420),rng.uniform(0.7,1.5),rng.choice([1.6,2.6,4])) for _ in range(70)]
def draw_sparks(cv,t,t0,px,py):
    k=t-t0
    if k<0 or k>1.6: return
    for a_,v,life,r in SPK:
        if k>life: continue
        x=px+math.cos(a_)*v*k*(1-0.35*k); y=py+math.sin(a_)*v*k*(1-0.35*k)+120*k*k
        add_sprite(cv,SPR[r],x,y,GOLD,(1-k/life)**1.5*1.4)
def draw_shock(cv,t,t0,cx,cy):
    k=t-t0
    if k<0 or k>1.0: return
    r=int(60+900*eo3(k)); al=(1-k)**2
    ov=np.zeros((H//2,W//2,3),np.float32)
    cv2.circle(ov,(cx//2,cy//2),r//2,(255,215,140),max(2,int(26*(1-k))),cv2.LINE_AA)
    cv+=cv2.resize(cv2.GaussianBlur(ov,(0,0),6),(W,H))*(0.55*al)

# ---------------- coin renderer
def coin_patch(theta,D,shine):
    c=math.cos(theta); sn=math.sin(theta)
    D=int(round(D)); w=max(int(D*abs(c)),3)
    face=cv2.resize(coinA,(max(w,2),D),interpolation=cv2.INTER_AREA if D<SD else cv2.INTER_LINEAR)
    rgb=face[...,:3].copy(); a=face[...,3]
    if shine is not None:
        band=np.exp(-((shine_u-shine)/0.07)**2)*coin_mask
        rgb=rgb+cv2.resize(band.astype(np.float32),(max(w,2),D))[...,None]*np.array([255,235,190],np.float32)*0.55
    rgb*=(0.62+0.38*abs(c))
    edge=int(round(D*0.045*abs(sn)))
    if edge>0:
        tw=w+edge
        patch=np.zeros((D,tw,4),np.float32)
        ed=np.zeros((D,tw),np.uint8)
        cv2.ellipse(ed,(tw//2,D//2),(tw//2,D//2),0,0,360,255,-1,cv2.LINE_AA)
        gradx=np.linspace(0,1,tw)[None,:]*np.ones((D,1))
        patch[...,:3]=np.stack([130+60*gradx,95+45*gradx,22+10*gradx],-1); patch[...,3]=ed/255.0
        off=-edge//2 if c*sn>0 else edge//2
        fx=min(max((tw-w)//2+off,0),tw-w)
        reg=patch[:,fx:fx+w]; fa=a[:,:w][...,None]
        reg[...,:3]=rgb[:,:w]*fa+reg[...,:3]*(1-fa); reg[...,3]=np.maximum(reg[...,3],a[:,:w])
        return patch
    return np.dstack([rgb[:,:w],a[:,:w]])
def blit(canvas,patch,cx,cy,alpha=1.0):
    h,w=patch.shape[:2]; x=int(round(cx-w/2)); y=int(round(cy-h/2))
    xa,ya=max(0,x),max(0,y); xb,yb=min(W,x+w),min(H,y+h)
    if xa>=xb or ya>=yb: return
    p=patch[ya-y:yb-y, xa-x:xb-x]; al=p[...,3:4]*alpha
    canvas[ya:yb,xa:xb]=canvas[ya:yb,xa:xb]*(1-al)+p[...,:3]*al

# ---------------- scene A (intro stage)
cxs,cys=540,820
FLOOR=1230
bgA=np.zeros((H,W,3),np.float32)
d=np.hypot((xx-cxs)*1.2,(yy-cys))/1100
glow=np.clip(1-d,0,1)**1.8
bgA[...,0]=4+16*glow; bgA[...,1]=9+44*glow; bgA[...,2]=20+78*glow
fl=np.clip((yy-FLOOR)/(H-FLOOR),0,1)
bgA*=(1-0.35*fl[...,None])
fg=np.exp(-((xx-cxs)/480)**2)*np.clip(1-abs(yy-FLOOR-60)/260,0,1)*(yy>FLOOR)
bgA[...,0]+=14*fg; bgA[...,1]+=11*fg
cv2.line(bgA,(0,FLOOR),(W,FLOOR),(60,75,95),1)
bgA=cv2.GaussianBlur(bgA,(0,0),1.0)
hw,hh=W//2,H//2
hy,hx=np.mgrid[0:hh,0:hw].astype(np.float32)
ang=np.arctan2(hy-cys/2,hx-cxs/2); rad=np.hypot(hx-cxs/2,hy-cys/2)
rayfall=np.exp(-rad/380).astype(np.float32)
sxs=np.arange(W)-W/2
streak=(np.exp(-(np.arange(-40,41)/3.2)**2)[:,None]*np.exp(-np.abs(sxs)/250)[None,:]).astype(np.float32)

# ---------------- scene B (course card)
bgB=np.zeros((H,W,3),np.float32)
dB=np.hypot((xx-540)*0.9,(yy-900)*0.62)/900
gB=np.clip(1-dB,0,1)**1.5
bgB[...,0]=6+12*gB; bgB[...,1]=15+42*gB; bgB[...,2]=34+66*gB
# grid pattern
gridl=np.zeros((H,W),np.float32)
for gx_ in range(0,W,135): gridl[:,gx_:gx_+1]=1
for gy_ in range(0,H,135): gridl[gy_:gy_+1,:]=1
bgB+=gridl[...,None]*np.array([6,14,22],np.float32)*(0.4+0.6*gB[...,None])
# skyline
BASE=1585; RY0=1290; RH=BASE+30-RY0
bl=[]
x=-10
heights=[130,200,150,95,250,170,120,210,145,100,175,135]
i=0
while x<W+20:
    w=int(rng.uniform(78,128)); h=heights[i%len(heights)]+int(rng.uniform(-15,25)); bl.append((x,w,h)); x+=w+int(rng.uniform(4,14)); i+=1
sky_body=np.zeros((RH,W,4),np.float32)
win_dim=np.zeros((RH,W,4),np.float32); win_bright=np.zeros((RH,W,4),np.float32)
for (bx,bw,bh) in bl:
    top=BASE-bh-RY0
    cv2.rectangle(sky_body,(bx,top),(bx+bw,BASE-RY0+30),(10,26,46,1.0),-1)
    cv2.rectangle(sky_body,(bx,top),(bx+bw,top+3),(40,80,120,1.0),-1)
    for wy in range(top+16,BASE-RY0-8,30):
        for wx in range(bx+12,bx+bw-14,22):
            r_=rng.rand()
            if r_<0.55:
                cv2.rectangle(win_dim,(wx,wy),(wx+9,wy+14),(205,170,95,0.35),-1)
                cv2.rectangle(win_bright,(wx,wy),(wx+9,wy+14),(255,215,120,1.0),-1)
            elif r_<0.8:
                cv2.rectangle(win_bright,(wx,wy),(wx+9,wy+14),(235,195,105,0.85),-1)
# brace segments for tall buildings
segs=[]
tall=sorted(bl,key=lambda b:-b[2])[:4]
for (bx,bw,bh) in tall:
    floors=int((bh-20)//58)
    for f in range(min(floors,3)):
        yb_=BASE-RY0-10-f*58; yt_=yb_-52
        segs.append(((bx+6,yb_),(bx+bw-6,yt_))); segs.append(((bx+bw-6,yb_),(bx+6,yt_)))
NSEG=len(segs)
def sky_layer(t,shear_amp,phase,bracep,winb):
    lay=sky_body.copy()
    wd=win_dim*(1-winb)+win_bright*winb
    a=wd[...,3:4]; lay[...,:3]=lay[...,:3]*(1-a)+wd[...,:3]*a
    lay[...,3]=np.maximum(lay[...,3],wd[...,3])
    # braces
    if bracep>0:
        n=bracep*NSEG
        for i,(p0,p1) in enumerate(segs):
            f=np.clip(n-i,0,1)
            if f<=0: break
            q=(int(p0[0]+(p1[0]-p0[0])*f),int(p0[1]+(p1[1]-p0[1])*f))
            cv2.line(lay,p0,q,(255,205,95,1.0),4,cv2.LINE_AA)
    # shear
    if shear_amp>0.05:
        ys=np.arange(RH,dtype=np.float32)
        hfac=np.clip((BASE-RY0-ys)/260.0,0,1.2)
        dx=shear_amp*phase*hfac
        mapx=(np.arange(W,dtype=np.float32)[None,:]+dx[:,None]).astype(np.float32)
        mapy=np.repeat(ys[:,None],W,1).astype(np.float32)
        lay=cv2.remap(lay,mapx,mapy,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
    return lay
def quake_env(t): return sstep((t-6.6)/0.4)*(1-sstep((t-8.7)/1.5))
def draw_frame(cv,p):
    if p<=0: return
    ov=np.zeros((H//2,W//2,3),np.float32)
    col=(235,190,95)
    for ins,th,pp in ((26,3,p),(44,1,p)):
        pts=[(W/2,ins),(W-ins,ins),(W-ins,H-ins),(ins,H-ins),(ins,ins),(W/2,ins)]
        segl=[math.hypot(pts[i+1][0]-pts[i][0],pts[i+1][1]-pts[i][1]) for i in range(5)]
        tot=sum(segl)
        # clockwise from top-center to p*tot/2, counter to same
        for dirn in (1,-1):
            order=range(5) if dirn==1 else range(4,-1,-1)
            remain=pp*tot/2*1.0
            # path param: first half runs top-center -> ... meet bottom center
            L=0
            ptsd=pts if dirn==1 else [pts[0]]+[pts[5-i] for i in range(1,5)]+[pts[0]]
            # build counter-clockwise explicit
            if dirn==-1: ptsd=[(W/2,ins),(ins,ins),(ins,H-ins),(W-ins,H-ins),(W-ins,ins),(W/2,ins)]
            for i in range(5):
                a=ptsd[i]; b=ptsd[i+1]; sl=math.hypot(b[0]-a[0],b[1]-a[1])
                if remain<=0: break
                f=min(1.0,remain/sl); q=(a[0]+(b[0]-a[0])*f,a[1]+(b[1]-a[1])*f)
                cv2.line(cv,(int(a[0]),int(a[1])),(int(q[0]),int(q[1])),col,th,cv2.LINE_AA)
                cv2.line(ov,(int(a[0]/2),int(a[1]/2)),(int(q[0]/2),int(q[1]/2)),col,th,cv2.LINE_AA)
                remain-=sl
    if p>0.97:
        for (px_,py_) in ((26,26),(W-26,26),(W-26,H-26),(26,H-26)):
            pts=np.array([[px_,py_-16],[px_+16,py_],[px_,py_+16],[px_-16,py_]],np.int32)
            cv2.fillPoly(cv,[pts],(245,205,120),cv2.LINE_AA)
            cv2.fillPoly(ov,[pts//2],(245,205,120),cv2.LINE_AA)
    cv+=cv2.resize(cv2.GaussianBlur(ov,(0,0),4),(W,H))*0.6
def wave(t,amp):
    xs=np.arange(70,1011,5,dtype=np.float32)
    y=(np.sin(xs*0.045+t*17)*0.6+np.sin(xs*0.11-t*23+1.3)*0.3+np.sin(xs*0.021+t*9.0)*0.5+np.sin(xs*0.33+t*31)*0.12)
    env=np.exp(-((xs-540)/520)**2*0.7)
    return xs,(y*env*amp).astype(np.float32)

T_APP=0.8; T_SPIN=3.4; T_SETTLE=4.2
T_FLY0=5.0; T_FLY1=6.4
FINAL=(540,420,300)   # coin final (cx,cy,D)
def renderB(t,pb):
    cv=bgB.copy()
    # ambient pulsing glow behind titles
    if t>6.5:
        pulse=0.5+0.5*math.sin((t-6.5)*1.6)
        gg=np.exp(-(np.hypot((xx-540)*0.9,(yy-800)*0.7)/420)**1.5)
        cv+=gg[...,None]*np.array([22,34,52],np.float32)*(0.5+0.5*sstep((t-6.5)/1.5))*(0.8+0.2*pulse)
    draw_frame(cv,eo3((t-5.3)/1.7))
    # skyline + quake
    q=quake_env(t)
    shear=26*q
    phase=math.sin(2*math.pi*3.1*t)+0.5*math.sin(2*math.pi*5.3*t+1.0)
    bracep=eo3((t-8.0)/1.8)
    winb=sstep((t-9.6)/0.9)
    lay=sky_layer(t,shear,phase,bracep,winb)
    skyop=0.95*sstep((t-5.6)/0.8)
    reg=cv[RY0:RY0+RH]
    a=lay[...,3:4]*skyop
    reg[:]=reg*(1-a)+lay[...,:3]*a
    # stable gold edge glow after retrofit
    if t>9.8:
        gl=sstep((t-9.8)/0.8)*(0.7+0.3*math.sin((t-9.8)*2))
        e=cv2.GaussianBlur(lay[...,3]*sstep(1)*0+lay[...,:3].sum(-1)*0,(0,0),1)
    # seismograph
    ybase=1470
    amp=70*q+ (3.0+2.0*math.sin(t*2.0))*sstep((t-6.4)/0.6)
    xs,ys=wave(t,amp if t>6.4 else 0)
    if t>5.8:
        env=sstep((t-5.8)/0.6)
        y0_,y1_=ybase-120,ybase+120
        ov=np.zeros((y1_-y0_,W,3),np.float32)
        pts=np.stack([xs,ybase-y0_-ys],1).astype(np.int32)
        col=(120,220,255) if q>0.15 else (255,205,110)
        cv2.polylines(ov,[pts],False,col,3,cv2.LINE_AA)
        cv[y0_:y1_]+=ov*env+cv2.GaussianBlur(ov,(0,0),7)*0.9*env
        cv2.line(cv,(70,ybase),(1010,ybase),(40,70,100),1,cv2.LINE_AA)
    return cv
def zoomB(t): return 1.0
def render(t):
    cv=bgA.copy()
    renv=sstep((t-0.2)/1.0)*(1-sstep((t-4.6)/1.2))
    if renv>0.01:
        rot=0.35*t
        rr=(np.abs(np.sin(7*ang+rot))**6*0.75+np.abs(np.sin(11*ang-rot*1.3))**9*0.55+np.abs(np.sin(4*ang+rot*0.6))**4*0.25)*rayfall
        cv+=cv2.resize(rr,(W,H))[...,None]*GOLD*(0.30*renv)
    u=(t-T_APP)/T_SPIN
    theta=2*math.pi*3*eo3(u) if t>=T_APP else 0
    sc=0.45+0.55*eo3((t-T_APP)/1.1)
    D0=700
    appear=sstep((t-T_APP)/0.45)
    bob=6*math.sin(2*math.pi*(t-T_APP)*0.45)*sstep((t-T_APP)/1.0)*(1-sstep((t-T_FLY0)/0.4)) if t>=T_APP else 0
    ccx,ccy=cxs,cys+bob+(1-eo3((t-T_APP)/1.1))*40
    genv=appear*(1-sstep((t-5.0)/1.0))
    if genv>0.01:
        gl=np.exp(-(np.hypot(xx-ccx,yy-ccy)/400.0)**1.6).astype(np.float32)
        cv+=gl[...,None]*GOLD*(0.28*genv)
    shine=None
    if T_SETTLE<=t<=T_SETTLE+0.9: shine=-0.3+1.6*eio3((t-T_SETTLE)/0.9)
    patch=None
    if appear>0.001 and t<T_FLY0:
        Dc=D0*sc
        patch=coin_patch(theta,Dc,shine)
        ph,pw=patch.shape[:2]
        yb=ccy+Dc/2; top_r=2*FLOOR-yb
        refl=patch[::-1].copy()
        refl[...,3]*=np.linspace(0.38,0.0,ph)[:,None]*appear*(1-sstep((t-T_FLY0+0.2)/0.6))
        refl=cv2.GaussianBlur(refl,(0,0),2.2); refl[...,:3]*=0.8
        blit(cv,refl,ccx,top_r+ph/2)
    pb=sstep((t-(T_FLY0+0.1))/1.0)
    if pb>0:
        B=renderB(t,pb)
        cv=cv*(1-pb)+B*pb
    # coin
    if t>=T_FLY0:
        u2=eio3((t-T_FLY0)/(T_FLY1-T_FLY0))
        tx,ty,tD=FINAL
        px=cxs+(tx-cxs)*u2; py=cys+(ty-cys)*u2-90*math.sin(math.pi*u2)
        Dc=D0+(tD-D0)*u2
        # halo behind coin
        halo=0.28+0.12*math.sin(t*2.0)
        if t>T_FLY1: add_sprite(cv,cv2.resize(SPR[40],(560,560)),px,py,np.array([255,190,100],np.float32),halo*sstep((t-T_FLY1)/0.5))
        blit(cv,coin_patch(0.0,Dc,None),px,py,1.0)
    elif patch is not None:
        blit(cv,patch,ccx,ccy,appear)
    # flares & effects
    f1=math.exp(-((t-0.9)/0.22)**2)*0.9
    f2=math.exp(-((t-T_SETTLE)/0.28)**2)
    f3=math.exp(-((t-6.45)/0.25)**2)*0.8
    for fv,(fx,fy) in ((f1,(cxs,cys)),(f2,(cxs,cys)),(f3,(FINAL[0],FINAL[1]))):
        if fv>0.02:
            y0_=int(fy)-40
            M=np.float32([[1,0,fx-W/2],[0,1,0]])
            s=cv2.warpAffine(streak,M,(W,81),flags=cv2.INTER_LINEAR)
            ya=max(0,y0_); yb=min(H,y0_+81)
            cv[ya:yb]+=s[ya-y0_:yb-y0_,:,None]*np.array([255,215,150],np.float32)*(0.9*fv)
            add_sprite(cv,cv2.resize(SPR[40],(460,460)),fx,fy,np.array([255,200,110],np.float32),0.45*fv)
    draw_shock(cv,t,T_SETTLE,cxs,cys)
    draw_sparks(cv,t,6.4,FINAL[0],FINAL[1])
    # ---- texts: phase 1 (organizer)
    draw_text(cv,'a1',t,3.2,5.0,540,1440)
    draw_text(cv,'a2',t,3.5,5.0,540,1515)
    draw_line(cv,t,3.9,5.0,1385,300)
    # ---- phase 2 titles
    draw_text(cv,'label',t,6.0,99,540,655)
    draw_text(cv,'t1',t,6.5,99,540,795,dur=1.2)
    draw_text(cv,'t2',t,6.9,99,540,930,dur=1.2)
    draw_line(cv,t,7.6,99,1012,330)
    draw_text(cv,'with',t,7.9,99,540,1078)
    draw_text(cv,'dr',t,8.3,99,540,1168,dur=1.3)
    draw_line(cv,t,9.2,99,1250,330)
    draw_text(cv,'org1',t,9.5,99,540,1318)
    draw_text(cv,'org2',t,9.8,99,540,1378)
    # particles
    for p in pk:
        y_=(p['y']-p['v']*t)%H; x_=(p['x']+p['dx']*t+14*math.sin(t*0.6+p['ph']))%W
        tw=0.6+0.4*math.sin(t*3+p['ph']*3); a=p['a']*tw
        if p['big']: a=p['a']*(0.7+0.3*math.sin(t*0.7+p['ph']))
        add_sprite(cv,SPR[p['r']],x_,y_,GOLD,a*sstep(t/1.2))
    # glint sweep over everything bright
    gs=sstep((t-10.6)/0.1)*(1-sstep((t-12.2)/0.1))
    if gs>0:
        pos=-300+1900*eio3((t-10.6)/1.6)
        band=np.exp(-((xx*0.6+yy*0.45-pos*0.9-300)/90.0)**2).astype(np.float32)
        lum=np.clip(cv.mean(-1)/255.0,0,1)**2.2
        cv+=(band*lum*1.2)[...,None]*np.array([255,230,170],np.float32)*gs
    # shakes
    sh=np.zeros(2)
    if T_SETTLE<=t<T_SETTLE+0.7:
        k=t-T_SETTLE; amp=11*math.exp(-k*7); sh+=(amp*math.sin(k*70),amp*math.cos(k*61))
    if 6.6<=t<8.8:
        k=t-6.6; amp=5*quake_env(t); sh+=(amp*math.sin(k*47),amp*math.cos(k*53))
    if abs(sh).sum()>0.01:
        cv=cv2.warpAffine(cv,np.float32([[1,0,sh[0]],[0,1,sh[1]]]),(W,H),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT)
    vig=np.clip(1-(np.hypot((xx-W/2)/(W*0.75),(yy-H/2)/(H*0.62)))**2.2*0.7,0.25,1)[...,None]
    cv*=vig
    cv+=grain[int(t*FPS)%6]*0.6
    cv*=1-sstep((t-(TEND-0.5))/0.5)
    return np.clip(cv,0,255).astype(np.uint8)

if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='test':
        for tt in [float(a) for a in sys.argv[2:]]:
            Image.fromarray(render(tt)).save(f'/tmp/v3_{tt:.2f}.png')
    else:
        n=int(TEND*FPS)
        ff=subprocess.Popen(['ffmpeg','-y','-f','rawvideo','-pix_fmt','rgb24','-s',f'{W}x{H}','-r',str(FPS),'-i','-','-c:v','libx264','-preset','medium','-crf','17','-pix_fmt','yuv420p','/home/claude/video3_noaudio.mp4'],stdin=subprocess.PIPE,stderr=open('/tmp/ff3.log','w'))
        for i in range(n):
            ff.stdin.write(render(i/FPS).tobytes())
            if i%15==0: open('/tmp/prog3.txt','w').write(f'{i}/{n}')
        ff.stdin.close(); ff.wait(); open('/tmp/prog3.txt','w').write('done')
'''


# ======================================================================
# کد رندر ویدیو (engine.py) 
# ======================================================================
SRC_ENGINE = r'''import cv2, numpy as np, math, sys, subprocess
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import os, json, tempfile
HERE=os.path.dirname(os.path.abspath(__file__))
CFG=json.load(open(os.environ['INTRO_CFG'],encoding='utf-8'))
import intro3 as I
I.FB=os.path.join(HERE,'assets','B-NAZANIN.TTF')
def _render_text(txt,size,top,bot):
    font=ImageFont.truetype(I.FB,size,layout_engine=ImageFont.Layout.RAQM)
    sw=max(1,int(size*0.028))
    wlen=int(font.getlength(txt,direction='rtl',language='fa'))+size
    hh=int(size*2.2)
    im=Image.new('L',(wlen,hh),0)
    ImageDraw.Draw(im).text((wlen//2,hh//2),txt,font=font,fill=255,anchor='mm',direction='rtl',language='fa',stroke_width=sw,stroke_fill=255)
    a=np.array(im).astype(np.float32)/255
    ys,xs=np.where(a>0.02); pad=8
    a=a[max(0,ys.min()-pad):ys.max()+pad, max(0,xs.min()-pad):xs.max()+pad]
    h,w=a.shape; g=np.linspace(0,1,h)[:,None,None]
    rgb=(np.array(top,np.float32)*(1-g)+np.array(bot,np.float32)*g)*np.ones((1,w,1),np.float32)
    return rgb,a
I.render_text=_render_text
from intro3 import sstep,eo3,eio3,eisine,W,H,FPS,GOLD
xx,yy=I.xx,I.yy
TEND=0
def eob(x):
    x=min(max(x,0),1); c1=1.70158; c3=c1+1
    return 1+c3*(x-1)**3+c1*(x-1)**2

# ---------------- texts
def fit_prep(name,txt,start,maxw,top,bot,col):
    size=start
    while size>20:
        f=ImageFont.truetype(I.FB,size,layout_engine=ImageFont.Layout.RAQM)
        if f.getlength(txt,direction='rtl',language='fa')<=maxw: break
        size-=4
    I.prep(name,txt,size,top,bot,col)
GT,GB=I.GT,I.GB; WT,WB=I.WT,I.WB
CY=((200,230,255),(170,205,240))
fit_prep('a1','انجمن صنفی کارفرمایی سازندگان',54,700,WT,WB,(255,215,150))
fit_prep('a2','ذی‌صلاح ساختمان شهرستان ارومیه',54,700,WT,WB,(255,215,150))
fit_prep('label','اعضای کمیته تخصصی',84,900,WT,WB,(255,215,150))
fit_prep('title',CFG['title'],150,900,GT,GB,(255,200,100))
MEM=[(m['n'],m['c'],m.get('r') or None) for m in CFG['members']]
for i,(n,c,r) in enumerate(MEM):
    fit_prep(f'n{i}',n,112,900,GT,GB,(255,200,100))
    fit_prep(f'c{i}',c,70,900,WT,WB,(255,215,150))
    if r: fit_prep(f'r{i}','('+r+')',60,900,(255,232,160),(224,170,70),(255,200,100))

# ---------------- banner + avatars
BANFILE=CFG['banner']
BAN=np.array(Image.open(BANFILE).convert('RGB')).astype(np.float32)
BW0,BH0=BAN.shape[1],BAN.shape[0]
TALL=BH0/BW0>1.9
SV=960.0/BW0 if TALL else min(960.0/BW0,1330.0/BH0)
VW=int(round(BW0*SV)); VH=1330 if TALL else int(round(BH0*SV))
VX0=(W-VW)//2; VY0=250 if TALL else int(915-VH/2)
BANS=cv2.resize(BAN,(VW,int(round(BH0*SV))),interpolation=cv2.INTER_AREA)
SCROLL_MAX=max(0,BANS.shape[0]-VH)
CIR=CFG['circles']; N=len(CIR); LOGO=CFG['logo']
T_M0=6.9; DT=1.0
T_P4=T_M0+DT*N+0.2; TB=T_P4+0.6
if TALL: S0=TB+2.9; S1=S0+DT*N; TEND=S1+2.9
else: S0=S1=TB+3.0; TEND=TB+9.8
def scroll(t): return SCROLL_MAX*eisine((t-S0)/(S1-S0)) if SCROLL_MAX>0 else 0
def slot(i,s=None,cx=None,cy=None): return VX0+CIR[i]['x']*SV, VY0+CIR[i]['y']*SV
def logo_slot(s=None,cx=None,cy=None): return VX0+LOGO['x']*SV, VY0+LOGO['y']*SV
VISIBLE=[(CIR[i]['y']*SV>40 and CIR[i]['y']*SV<VH-40) for i in range(N)]
NVIS=sum(VISIBLE)
DSLOT=2*0.922*float(np.mean([c['r'] for c in CIR]))*SV
def _rr(w,h,r,ss=3):
    im=Image.new('L',(w*ss,h*ss),0); ImageDraw.Draw(im).rounded_rectangle((0,0,w*ss-1,h*ss-1),radius=r*ss,fill=255)
    return np.array(im.resize((w,h),Image.LANCZOS)).astype(np.float32)/255
VMASK=_rr(VW,VH,30)
_fy=np.minimum(np.arange(VH),VH-1-np.arange(VH)).astype(np.float32)
if TALL: VMASK=VMASK*np.clip(_fy/30.0,0,1)[:,None]
_bo=_rr(VW+12,VH+12,34); _bi=np.zeros_like(_bo); _bi[5:5+_rr(VW+2,VH+2,30).shape[0],5:5+_rr(VW+2,VH+2,30).shape[1]]=_rr(VW+2,VH+2,30)
_bd=np.clip(_bo-_bi,0,1)
BORDER=np.zeros((VH+12,VW+12,4),np.float32)
_g=np.linspace(0,1,VH+12)[:,None,None]
BORDER[...,:3]=(np.array([255,228,150],np.float32)*(1-_g)+np.array([196,140,42],np.float32)*_g)*np.ones((1,VW+12,1),np.float32)
BORDER[...,3]=_bd
srcim=Image.open(BANFILE).convert('RGB')
SP=min(190,int(900/(N-1))) if N>1 else 0
ROWD=int(min(112,SP*0.77)) if N>1 else 112
AV=[]
for i in range(N):
    cx_=CIR[i]['x']; cy_=CIR[i]['y']; r=int(round(CIR[i]['r']*0.99))
    crop=srcim.crop((int(round(cx_-r)),int(round(cy_-r)),int(round(cx_+r)),int(round(cy_+r)))).resize((440,440),Image.LANCZOS)
    crop=crop.filter(ImageFilter.UnsharpMask(radius=2.2,percent=110,threshold=2))
    m=Image.new('L',(440*2,440*2),0); RM=71*440/154.0
    ImageDraw.Draw(m).ellipse((2*(220-RM),2*(220-RM),2*(220+RM),2*(220+RM)),fill=255)
    a=np.array(m.resize((440,440),Image.LANCZOS)).astype(np.float32)/255
    rgb=np.array(crop).astype(np.float32)
    AV.append(np.dstack([rgb,a]))
def avatar_patch(i,Din,ring_p,glow=0.0):
    Din=int(round(Din)); S=Din+40
    p=np.zeros((S,S,4),np.float32)
    av=cv2.resize(AV[i],(int(Din*77/71),)*2,interpolation=cv2.INTER_AREA if Din<440 else cv2.INTER_LINEAR)
    h=av.shape[0]; o=(S-h)//2
    p[o:o+h,o:o+h]=av
    ring=np.zeros((S,S),np.uint8)
    rr=int(Din/2+9)
    if ring_p>0:
        cv2.ellipse(ring,(S//2,S//2),(rr,rr),0,-90,-90+360*min(ring_p,1)-0.01,255,5,cv2.LINE_AA)
    rg=ring.astype(np.float32)/255
    g=np.linspace(0,1,S)[:,None]
    col=np.stack([255-55*g,226-86*g,140-100*g],-1)*np.ones((1,S,1),np.float32)
    p[...,:3]=p[...,:3]*(1-rg[...,None])+col*rg[...,None]
    p[...,3]=np.maximum(p[...,3],rg)
    return p
def slot_ring(S,Din,alpha):
    # faint placeholder ring
    S=int(Din+40); r=np.zeros((S,S),np.uint8)
    cv2.ellipse(r,(S//2,S//2),(int(Din/2+9),int(Din/2+9)),0,0,360,255,3,cv2.LINE_AA)
    p=np.zeros((S,S,4),np.float32); p[...,:3]=(235,190,95); p[...,3]=r/255.0*alpha
    return p

# ---------------- backgrounds
cxs,cys=I.cxs,I.cys; FLOOR=I.FLOOR
bgA=np.zeros((H,W,3),np.float32)
d=np.hypot((xx-cxs)*1.2,(yy-cys))/1100
glow=np.clip(1-d,0,1)**1.8
bgA[...,0]=3+34*glow; bgA[...,1]=3+22*glow; bgA[...,2]=4+7*glow
fl=np.clip((yy-FLOOR)/(H-FLOOR),0,1)
bgA*=(1-0.35*fl[...,None])
fg=np.exp(-((xx-cxs)/480)**2)*np.clip(1-abs(yy-FLOOR-60)/260,0,1)*(yy>FLOOR)
bgA[...,0]+=16*fg; bgA[...,1]+=11*fg
cv2.line(bgA,(0,FLOOR),(W,FLOOR),(80,64,40),1)
bgA=cv2.GaussianBlur(bgA,(0,0),1.0)
# ---------- new opening assets
bgN=np.zeros((H,W,3),np.float32)
dN=np.hypot((xx-cxs)*1.1,(yy-cys))/1150; gN=np.clip(1-dN,0,1)**1.7
bgN[...,0]=3+30*gN; bgN[...,1]=3+20*gN; bgN[...,2]=4+7*gN
_rng=np.random.RandomState(21)
_pts=np.argwhere(I.coinA[...,3]>0.5)
_sel=_rng.choice(len(_pts),22000,replace=False); _pts=_pts[_sel]
_s=700/(2*I.R0)
_tx=(_pts[:,1]-I.SD/2)*_s; _ty=(_pts[:,0]-I.SD/2)*_s
PTH=np.arctan2(_ty,_tx).astype(np.float32); PRT=np.hypot(_tx,_ty).astype(np.float32)
PR0=_rng.uniform(420,1500,len(_pts)).astype(np.float32)
PDELTA=_rng.uniform(2.2,6.0,len(_pts)).astype(np.float32)
PD_=_rng.uniform(0,0.4,len(_pts)).astype(np.float32)
_col=I.coinA[_pts[:,0],_pts[:,1],:3]
PCOL=[np.clip(_col[:,c]*1.1+25,0,255)*0.85 for c in range(3)]
bgB=np.zeros((H,W,3),np.float32)
dB=np.hypot((xx-540)*0.9,(yy-860)*0.62)/950
gB=np.clip(1-dB,0,1)**1.6
bgB[...,0]=4+22*gB; bgB[...,1]=4+15*gB; bgB[...,2]=5+5*gB
gl_=np.zeros((H,W),np.float32)
for gx_ in range(0,W,135): gl_[:,gx_:gx_+1]=1
for gy_ in range(0,H,135): gl_[gy_:gy_+1,:]=1
bgB+=gl_[...,None]*np.array([5,4,1.5],np.float32)*(0.3+0.7*gB[...,None])

T_FLY0,T_FLY1=5.0,6.4
FIN=(540,350,250)
T_M0=6.9; DT=1.0
ROW_Y=1470
def row_slot(i): return 540+((N-1)/2-i)*SP, ROW_Y
def coin_state(t):
    if t<T_FLY0: return None
    if t<T_P4:
        u=eio3((t-T_FLY0)/(T_FLY1-T_FLY0))
        return (cxs+(FIN[0]-cxs)*u, cys+(FIN[1]-cys)*u-90*math.sin(math.pi*u), 700+(FIN[2]-700)*u, 1.0)
    u=eio3((t-T_P4)/1.3); lx,ly=logo_slot()
    al=1-sstep((t-(TB+1.6))/0.5)
    return (FIN[0]+(lx-FIN[0])*u, FIN[1]+(ly-FIN[1])*u-70*math.sin(math.pi*u), FIN[2]+(2*LOGO['r']*SV-FIN[2])*u, al)

def banner_scale(t): return SB if t<13.5 else SB*(1+0.02*eisine((t-13.5)/7.0))

def add_glow(cv,cx,cy,w,h,col,a):
    g=cv2.resize(I.SPR[40],(int(w),int(h)))
    x=int(cx-w/2); y=int(cy-h/2)
    xa,ya=max(0,x),max(0,y); xb,yb=min(W,x+int(w)),min(H,y+int(h))
    if xa>=xb or ya>=yb: return
    cv[ya:yb,xa:xb]+=g[ya-y:yb-y,xa-x:xb-x,None]*(col*a)

def render(t):
    cv=bgA.copy()
    renv=sstep((t-0.2)/1.0)*(1-sstep((t-4.6)/1.2))
    if renv>0.01:
        rot=0.35*t
        rr=(np.abs(np.sin(7*I.ang+rot))**6*0.75+np.abs(np.sin(11*I.ang-rot*1.3))**9*0.55+np.abs(np.sin(4*I.ang+rot*0.6))**4*0.25)*I.rayfall
        cv+=cv2.resize(rr,(W,H))[...,None]*GOLD*(0.30*renv)
    T_APP,T_SPIN,T_SETTLE=0.8,3.4,4.2
    u=(t-T_APP)/T_SPIN
    theta=2*math.pi*3*eo3(u) if t>=T_APP else 0
    sc=0.45+0.55*eo3((t-T_APP)/1.1)
    D0=700
    appear=sstep((t-T_APP)/0.45)
    bob=6*math.sin(2*math.pi*(t-T_APP)*0.45)*sstep((t-T_APP)/1.0)*(1-sstep((t-T_FLY0)/0.4)) if t>=T_APP else 0
    ccx,ccy=cxs,cys+bob+(1-eo3((t-T_APP)/1.1))*40
    genv=appear*(1-sstep((t-5.0)/1.0))
    if genv>0.01:
        gl=np.exp(-(np.hypot(xx-ccx,yy-ccy)/400.0)**1.6).astype(np.float32)
        cv+=gl[...,None]*GOLD*(0.28*genv)
    shine=None
    if T_SETTLE<=t<=T_SETTLE+0.9: shine=-0.3+1.6*eio3((t-T_SETTLE)/0.9)
    patch=None
    if appear>0.001 and t<T_FLY0:
        Dc=D0*sc
        patch=I.coin_patch(theta,Dc,shine)
        ph=patch.shape[0]; yb=ccy+Dc/2; top_r=2*FLOOR-yb
        refl=patch[::-1].copy()
        refl[...,3]*=np.linspace(0.38,0.0,ph)[:,None]*appear*(1-sstep((t-T_FLY0+0.2)/0.6))
        refl=cv2.GaussianBlur(refl,(0,0),2.2); refl[...,:3]*=0.8
        I.blit(cv,refl,ccx,top_r+ph/2)
    # ---- scene B
    pb=sstep((t-(T_FLY0+0.1))/1.0)
    if pb>0:
        B=bgB.copy()
        if t>6.0:
            pulse=0.5+0.5*math.sin((t-6.0)*1.5)
            gg=np.exp(-(np.hypot((xx-540)*0.9,(yy-860)*0.7)/430)**1.5)
            B+=gg[...,None]*np.array([30,22,8],np.float32)*(0.5+0.5*sstep((t-6.0)/1.5))*(0.8+0.2*pulse)
        I.draw_frame(B,eo3((t-5.3)/1.7))
        # banner viewport (tall banner scrolls)
        if t>=TB-0.1:
            sc_=int(scroll(t))
            band_=BANS[sc_:sc_+VH]
            p=eo3((t-TB)/1.4)
            ys_=np.arange(VH,dtype=np.float32)
            front=(VH+160)*p
            mrow=sstep((front-ys_)/160)*sstep((t-(TB-0.1))/0.3)
            a_=(VMASK*mrow[:,None])[...,None]
            B[VY0:VY0+VH,VX0:VX0+VW]=B[VY0:VY0+VH,VX0:VX0+VW]*(1-a_)+band_*a_
            if p<0.999:
                ly=int(VY0+front-80)
                if VY0<=ly<VY0+VH:
                    bnd=np.exp(-((yy-ly)/16.0)**2)[...,None]*np.array([255,215,140],np.float32)*0.6
                    ins=((xx>=VX0)&(xx<VX0+VW)).astype(np.float32)[...,None]
                    B+=bnd*ins
            bo_=BORDER; bal=sstep((t-TB)/0.5)
            B[VY0-6:VY0+VH+6,VX0-6:VX0+VW+6]=B[VY0-6:VY0+VH+6,VX0-6:VX0+VW+6]*(1-bo_[...,3:4]*bal)+bo_[...,:3]*bo_[...,3:4]*bal
        cv=cv*(1-pb)+B*pb
    # ---- slot placeholders + avatars
    if t>6.6:
        for i in range(N):
            rx,ry=row_slot(i)
            ph=sstep((t-6.6-0.08*i)/0.5)*(1-sstep((t-T_P4)/0.3))
            if ph>0.01: I.blit(cv,slot_ring(0,ROWD,0.35),rx,ry,ph)
    for i in range(N):
        ti=T_M0+DT*i; a=t-ti
        if a<0: continue
        rx,ry=row_slot(i)
        if a<0.8:
            D=270*(0.55+0.45*eob(min(a/0.45,1))); cx_,cy_=540,980; al=sstep(a/0.25); rp=eo3(a/0.55)
            if a<0.8 and a>0.05:
                add=I.add_sprite
                add(cv,cv2.resize(I.SPR[40],(int(D*1.9),)*2),cx_,cy_,np.array([255,190,100],np.float32),0.30*al)
        elif a<1.12:
            u2=eio3((a-0.8)/0.32); cx_=540+(rx-540)*u2; cy_=980+(ry-980)*u2; D=270+(ROWD-270)*u2; al=1; rp=1
        else:
            cx_,cy_,D,al,rp=rx,ry,ROWD,1,1
        if t>=T_P4:
            if VISIBLE[i]:
                u3=eio3((t-T_P4-0.06*i)/1.0); sx,sy=slot(i)
                cx_=rx+(sx-rx)*u3; cy_=ry+(sy-ry)*u3; D=ROWD+(DSLOT-ROWD)*u3
                al=1-sstep((t-(TB+1.6))/0.5)
            else:
                cx_,cy_,D=rx,ry,ROWD*(1-0.35*sstep((t-T_P4)/0.6))
                al=1-sstep((t-T_P4-0.12*(i-NVIS))/0.6)
        if al>0.01:
            I.blit(cv,avatar_patch(i,D,rp),cx_,cy_,al)
        # landing glow in row
        if 1.0<a<1.7 and t<T_P4:
            I.add_sprite(cv,cv2.resize(I.SPR[40],(300,300)),rx,ry,np.array([255,200,110],np.float32),0.5*(1-(a-1.0)/0.7))
    # ---- coin
    cs=coin_state(t)
    if cs is not None:
        px,py,Dc,al=cs
        if t>T_FLY1 and t<T_P4+0.3:
            I.add_sprite(cv,cv2.resize(I.SPR[40],(520,520)),px,py,np.array([255,190,100],np.float32),(0.26+0.1*math.sin(t*2.0))*sstep((t-T_FLY1)/0.5))
        if al>0.01: I.blit(cv,I.coin_patch(0.0,Dc,None),px,py,al)
    elif patch is not None:
        I.blit(cv,patch,ccx,ccy,appear)
    # ---- flares
    f1=math.exp(-((t-0.9)/0.22)**2)*0.9
    f2=math.exp(-((t-T_SETTLE)/0.28)**2)
    f3=math.exp(-((t-6.45)/0.25)**2)*0.8
    for fv,(fx,fy) in ((f1,(cxs,cys)),(f2,(cxs,cys)),(f3,(FIN[0],FIN[1]))):
        if fv>0.02:
            y0_=int(fy)-40
            M=np.float32([[1,0,fx-W/2],[0,1,0]])
            s_=cv2.warpAffine(I.streak,M,(W,81),flags=cv2.INTER_LINEAR)
            ya=max(0,y0_); yb=min(H,y0_+81)
            cv[ya:yb]+=s_[ya-y0_:yb-y0_,:,None]*np.array([255,215,150],np.float32)*(0.9*fv)
            I.add_sprite(cv,cv2.resize(I.SPR[40],(460,460)),fx,fy,np.array([255,200,110],np.float32),0.45*fv)
    I.draw_shock(cv,t,T_SETTLE,cxs,cys)
    I.draw_sparks(cv,t,6.4,FIN[0],FIN[1])
    for i in range(N):
        if 0<t-(T_M0+DT*i)<1.6 and False: pass
    # ---- text
    I.draw_text(cv,'a1',t,3.2,5.0,540,1450)
    I.draw_text(cv,'a2',t,3.5,5.0,540,1510)
    I.draw_line(cv,t,3.9,5.0,1395,230)
    I.draw_text(cv,'label',t,5.9,T_P4,540,590,fade_out=0.5)
    I.draw_text(cv,'title',t,6.2,T_P4,540,690,dur=1.3,fade_out=0.5)
    I.draw_line(cv,t,6.7,T_P4,790,330)
    for i in range(N):
        ti=T_M0+DT*i
        I.draw_text(cv,f'n{i}',t,ti+0.2,ti+0.9,540,1200,dur=0.7,fade_out=0.22)
        I.draw_text(cv,f'c{i}',t,ti+0.4,ti+0.9,540,1272,dur=0.6,fade_out=0.22)
        if f'r{i}' in I.TXT: I.draw_text(cv,f'r{i}',t,ti+0.5,ti+0.9,540,1336,dur=0.5,fade_out=0.22)
    # glint over banner
    for g0 in [g for g in (TB+4.8,TB+10.3) if g<TEND-1.6]:
        gs=sstep((t-g0)/0.1)*(1-sstep((t-g0-1.5)/0.1))
        if gs>0:
            pos=-300+1900*eio3((t-g0)/1.5)
            band=np.exp(-((xx*0.6+yy*0.45-pos*0.9-300)/90.0)**2).astype(np.float32)
            lum=np.clip(cv.mean(-1)/255.0,0,1)**1.6
            cv+=(band*lum*0.9)[...,None]*np.array([255,230,170],np.float32)*gs
    sh=np.zeros(2)
    if T_SETTLE<=t<T_SETTLE+0.7:
        k=t-T_SETTLE; amp=11*math.exp(-k*7); sh+=(amp*math.sin(k*70),amp*math.cos(k*61))
    if abs(sh).sum()>0.01:
        cv=cv2.warpAffine(cv,np.float32([[1,0,sh[0]],[0,1,sh[1]]]),(W,H),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT)
    vig=np.clip(1-(np.hypot((xx-W/2)/(W*0.75),(yy-H/2)/(H*0.62)))**2.2*0.7,0.25,1)[...,None]
    cv*=vig
    cv+=I.grain[int(t*FPS)%6]*0.6
    cv*=1-sstep((t-(TEND-0.5))/0.5)
    return np.clip(cv,0,255).astype(np.uint8)

def run(a,b,name):
    ff=subprocess.Popen(['ffmpeg','-y','-f','rawvideo','-pix_fmt','rgb24','-s',f'{W}x{H}','-r',str(FPS),'-i','-','-c:v','libx264','-preset','medium','-crf','19','-pix_fmt','yuv420p',name],stdin=subprocess.PIPE,stderr=subprocess.DEVNULL)
    for i in range(a,b):
        ff.stdin.write(render(i/FPS).tobytes())
    ff.stdin.close(); ff.wait()
if __name__=='__main__':
    if sys.argv[1]=='info':
        print(json.dumps(dict(TEND=TEND,N=N,TB=TB,S0=S0,S1=S1,TALL=bool(TALL),FPS=FPS)))
    elif sys.argv[1]=='test':
        for tt in [float(x) for x in sys.argv[2:]]:
            Image.fromarray(render(tt)).save(os.path.join(tempfile.gettempdir(),f'intro_{tt:.2f}.png'))
    else:
        run(int(sys.argv[1]),int(sys.argv[2]),sys.argv[3])
'''


# ======================================================================
# کد رندر ویدیو (audio_gen.py) 
# ======================================================================
SRC_AUDIO = r'''import numpy as np, scipy.signal as sg, scipy.io.wavfile as wf, sys, json, os
INFO=json.loads(sys.argv[1]); OUTD=sys.argv[2]
SR=44100; T=INFO['TEND']; N=INFO['N']; TB=INFO['TB']; S0=INFO['S0']; S1=INFO['S1']; TALL=INFO['TALL']; n=int(SR*T); t=np.arange(n)/SR
rng=np.random.RandomState(12)
def noise(d): return rng.randn(int(d*SR))
def bp(x,lo,hi): b,a=sg.butter(2,[lo/(SR/2),hi/(SR/2)],'band'); return sg.lfilter(b,a,x)
def lp(x,f): b,a=sg.butter(2,f/(SR/2)); return sg.lfilter(b,a,x)
class Bus:
    def __init__(s): s.L=np.zeros(n); s.R=np.zeros(n)
    def add(s,sig,t0,pan=0.0,g=1.0):
        i=int(t0*SR)
        if i<0 or i>=n: return
        m=min(len(sig),n-i)
        s.L[i:i+m]+=sig[:m]*g*(1-max(pan,0)); s.R[i:i+m]+=sig[:m]*g*(1+min(pan,0))
SFX=Bus(); LEAD=Bus(); PAD=Bus(); DRUM=Bus(); BASS=Bus()
def adsr(d,a,r):
    k=np.arange(int(d*SR))/SR
    return np.minimum(k/a,1)*np.clip((d-k)/r,0,1)
def piano2(f,vel=1.0,d=2.4):
    k=np.arange(int(d*SR))/SR; dec=0.9+f/500.0; x=0
    for h,a in ((1,1.0),(2,0.55),(3,0.28),(4,0.16),(5,0.08),(6,0.05)):
        x=x+a*np.sin(2*np.pi*f*h*(1+0.0004*h*h)*k)*np.exp(-k*dec*(1+0.35*(h-1)))
    ham=lp(noise(0.03),5000)*np.exp(-np.arange(int(0.03*SR))/SR*120)*0.25
    x[:len(ham)]+=ham
    return lp(x,5200)*np.minimum(k/0.003,1)*vel
def strings2(f,d,att=0.5,rel=0.9):
    k=np.arange(int(d*SR))/SR
    vib=1+0.003*np.sin(2*np.pi*5.1*k)*np.minimum(k/0.9,1)
    ph=2*np.pi*f*np.cumsum(vib)/SR; x=0
    for c in (-9,0,9): x=x+sg.sawtooth(ph*2**(c/1200))
    return (lp(x/3,2600)+0.25*np.sin(ph*0.5))*adsr(d,att,rel)
def harp(f,d=1.6):
    k=np.arange(int(d*SR))/SR
    return (np.sin(2*np.pi*f*k)+0.25*np.sin(2*np.pi*2*f*k))*np.exp(-k*3.0)*np.minimum(k/0.002,1)
def softkick():
    k=np.arange(int(0.45*SR))/SR
    return np.sin(2*np.pi*np.cumsum(42+70*np.exp(-k*30))/SR)*np.exp(-k*9)
def brush(op=0.09):
    k=np.arange(int(op*SR))/SR; b_,a_=sg.butter(2,5500/(SR/2),'high'); return sg.lfilter(b_,a_,noise(op))*np.exp(-k*38)
def timp(f=90.0,d=1.2):
    k=np.arange(int(d*SR))/SR
    return np.sin(2*np.pi*np.cumsum(f*(1+0.5*np.exp(-k*25)))/SR)*np.exp(-k*3.0)+0.25*lp(noise(d),400)*np.exp(-k*12)
def sub(f,d):
    k=np.arange(int(d*SR))/SR
    return np.tanh(np.sin(2*np.pi*np.cumsum(f*(1+1.0*np.exp(-k*60)))/SR)*np.exp(-k*1.6)*2.0)*0.6
def crash(d=3.0):
    k=np.arange(int(d*SR))/SR; b_,a_=sg.butter(2,3500/(SR/2),'high'); return sg.lfilter(b_,a_,noise(d))*np.exp(-k*1.6)
def whoosh(bus,t_end,d=0.5,lo=400,hi=3500,up=True,g=0.5,pan=0.0):
    k=np.arange(int(d*SR))/SR; nz=noise(d); out=np.zeros_like(nz)
    for j in range(0,len(nz),1024):
        fr=j/len(nz); f=(lo+(hi-lo)*fr) if up else (hi-(hi-lo)*fr)
        out[j:j+1024]=bp(nz[j:j+1024],f*0.6,f*1.5)
    bus.add(out*np.sin(np.pi*np.clip(k/d,0,1))**1.5*g,t_end-d,pan=pan)
def chime(bus,tt,f,g=0.12,pan=0.0):
    k=np.arange(int(1.6*SR))/SR
    bus.add((np.sin(2*np.pi*f*k)+0.3*np.sin(2*np.pi*2*f*k)+0.1*np.sin(2*np.pi*3.01*f*k))*np.exp(-k*3.0)*np.minimum(k/0.002,1),tt,pan=pan,g=g)
def shimmer(bus,tt,d=1.5,g=0.3,pan=0.2):
    kk=np.arange(int(d*SR))/SR
    bus.add(bp(noise(d),3000,10000)*np.sin(np.pi*np.clip(kk/d,0,1))**2,tt,pan=pan,g=g)
def riser(d,lo,hi,pw=1.5):
    k=np.arange(int(d*SR))/SR; nz=noise(d); sw=np.zeros_like(nz)
    for j in range(0,len(nz),2048):
        f=lo+(hi-lo)*(j/len(nz))**pw; sw[j:j+2048]=bp(nz[j:j+2048],f*0.6,f*1.4)
    return sw*(k/d)**1.6
PENT=[1396.91,1567.98,1760.0,2093.0,2349.3,2793.0]
# ================= SFX: spinning-coin entrance =================
PENT=[1396.91,1567.98,1760.0,2093.0,2349.3,2793.0]
kk0=np.arange(int(2.4*SR))/SR
SFX.add(np.sin(2*np.pi*(40+40*np.exp(-kk0*6))*kk0)*np.exp(-kk0*2.0),0.85,g=0.8)      # coin enters
SFX.add(lp(noise(2.4),600)*np.exp(-kk0*3),0.85,g=0.4); shimmer(SFX,0.85,1.6,0.3,0.0)
SFX.add(riser(3.0,500,8000,1.4),1.2,g=0.45)
for h in range(1,7):                                   # half-turn swishes (spin decelerates)
    x_=(0.5*h)/3.0; u_=1-(1-x_)**(1/3.0) if x_<1 else 1.0
    tt=0.8+3.4*u_
    whoosh(SFX,tt,0.45,500+h*250,2500+h*600,True,0.45,pan=(-1)**h*0.5)
    chime(SFX,tt,[1396.91,1567.98,1760.0,2093.0,2349.3,2793.0][h-1],0.07,pan=(-1)**h*0.4)
kk2=np.arange(int(2.6*SR))/SR
SFX.add(np.sin(2*np.pi*(40+34*np.exp(-kk2*7))*kk2)*np.exp(-kk2*2.2),4.2,g=0.9)     # settle impact
SFX.add(lp(noise(2.6),700)*np.exp(-kk2*3),4.2,g=0.5); shimmer(SFX,4.2,2.4,0.4,0.0)
for j,f in enumerate((1396.91,1760.0,2093.0,2637.0)): chime(SFX,4.22+0.1*j,f,0.12,pan=-0.3+0.2*j)
chime(SFX,3.2,1760.0,0.10); chime(SFX,3.5,2349.3,0.10)
whoosh(SFX,6.4,1.4,400,3500,True,0.5)
chime(SFX,6.4,2349.3,0.12); shimmer(SFX,6.4,1.4,0.25,-0.3)
chime(SFX,5.9,1760.0,0.10); chime(SFX,6.3,2093.0,0.10); chime(SFX,6.7,2349.3,0.10)
# ---- members: pop + chime + flight whoosh + landing tick
for i in range(N):
    ti=6.9+1.0*i
    kk=np.arange(int(0.5*SR))/SR
    SFX.add(np.sin(2*np.pi*(90+60*np.exp(-kk*12))*kk)*np.exp(-kk*8),ti,g=0.4)
    chime(SFX,ti+0.05,PENT[i%6]*(1.0 if i<6 else 0.5),0.14,pan=0.3*(i-4)/4)
    whoosh(SFX,ti+1.0,0.45,1200,5000,True,0.22)
    d_=0.05; k2=np.arange(int(d_*SR))/SR
    SFX.add(bp(noise(d_),1800,6500)*np.exp(-k2*80),ti+1.0,g=0.3)
# ---- banner reveal at 16.7 (mosaic)
TBv=TB
SFX.add(riser(1.0,1500,9000),TBv-0.9,g=0.5)
whoosh(SFX,TBv+0.3,1.0,300,3800,True,0.4)
kk3=np.arange(int(2.4*SR))/SR
SFX.add(np.sin(2*np.pi*(46+34*np.exp(-kk3*8))*kk3)*np.exp(-kk3*2.6),TBv,g=0.85)
shimmer(SFX,TBv,2.0,0.35,0.0)
for j,f in enumerate((1396.91,1760.0,2093.0,2793.0)): chime(SFX,TBv+0.1+0.12*j,f,0.12,pan=-0.3+0.2*j)
for j in range(14): chime(SFX,TBv+0.2+0.1*j,PENT[j%6]*1.5,0.05,pan=-0.5+0.08*j)
for g0 in [g for g in ([S0,S1] if TALL else [TB+4.8,TB+10.3]) if g<T-1.6]: shimmer(SFX,g0,1.5,0.25); whoosh(SFX,g0+0.5,0.9,500,3000,True,0.15)
# ================= MUSIC =================
for fq,gg in ((87.31,0.8),(174.61,1.0),(261.63,0.9),(440.0,0.6),(523.25,0.5)):
    PAD.add(strings2(fq,3.6,2.8,0.4),0.7,g=0.10*gg)
d=3.3; k=np.arange(int(d*SR))/SR
ph=2*np.pi*np.cumsum(174.61*(4**(k/d)))/SR
LEAD.add(lp(np.sin(ph)*(k/d)**2*0.12,3000),0.9)
for j in range(10): DRUM.add(timp(70.0,0.6),1.6+2.5*(1-(1-j/10)**1.6),g=0.06+0.06*j/10)
DRUM.add(timp(60.0,2.0),4.2,g=0.9); BASS.add(sub(43.65,2.4),4.2,g=0.9); DRUM.add(crash(3.2),4.2,g=0.4)
for fq in (174.61,220.0,261.63,349.23,440.0,523.25): PAD.add(strings2(fq,2.4,0.05,1.2),4.2,g=0.09)
for fq in (174.61,261.63,329.63,440.0): LEAD.add(piano2(fq,1.0,3.0),4.2,g=0.25)
for j,fq in enumerate((261.63,329.63,392.0,523.25,659.25,783.99)): LEAD.add(harp(fq),5.0+0.07*j,pan=-0.2+0.08*j,g=0.16)
CHD={'Fmaj7':dict(r=87.31,ch=(174.61,220.0,261.63,329.63),ar=[174.61,261.63,329.63,440.0,329.63,261.63,220.0,261.63]),
     'Am7':dict(r=110.0,ch=(164.81,196.0,261.63,329.63),ar=[164.81,261.63,329.63,392.0,329.63,261.63,196.0,261.63]),
     'Bbmaj7':dict(r=116.54,ch=(174.61,220.0,293.66,349.23),ar=[174.61,293.66,349.23,440.0,349.23,293.66,220.0,293.66]),
     'C':dict(r=130.81,ch=(196.0,261.63,329.63,392.0),ar=[196.0,329.63,392.0,523.25,392.0,329.63,261.63,329.63]),
     'Dm7':dict(r=146.83,ch=(174.61,220.0,261.63,349.23),ar=[174.61,261.63,349.23,440.0,349.23,261.63,220.0,261.63]),
     'Gm7':dict(r=98.0,ch=(196.0,233.08,293.66,349.23),ar=[196.0,293.66,349.23,466.16,349.23,293.66,233.08,293.66])}
SEQ=['Fmaj7','Am7','Bbmaj7','C','Dm7','Bbmaj7','Gm7','C']
MELS=[[(0,1.5,523.25),(1.5,0.5,587.33),(2,2,659.25)],
      [(0,1.5,659.25),(1.5,0.5,587.33),(2,1,523.25),(3,1,440.0)],
      [(0,1,587.33),(1,1,698.46),(2,2,880.0)],
      [(0,1.5,783.99),(1.5,0.5,698.46),(2,2,659.25)],
      [(0,1.5,698.46),(1.5,0.5,659.25),(2,1,587.33),(3,1,523.25)],
      [(0,1,698.46),(1,1,587.33),(2,2,466.16)],
      [(0,1.5,466.16),(1.5,0.5,523.25),(2,2,587.33)],
      [(0,2,659.25),(2,2,783.99)]]
B=0.46875
def bar(t0,name,level=1.0,arp=True,mel=None,mel_g=1.0,pad_g=0.05,oct2=False):
    c=CHD[name]
    for fq in c['ch']: PAD.add(strings2(fq,2.7),t0,g=pad_g*level)
    for b in (0,2):
        kb=np.arange(int(1.4*SR))/SR
        BASS.add((np.sin(2*np.pi*c['r']*kb)+0.3*np.sin(2*np.pi*2*c['r']*kb))*np.exp(-kb*1.4)*np.minimum(kb/0.01,1),t0+b*B,g=0.32*level)
    if arp:
        vels=[0.9,0.5,0.6,0.75,0.8,0.5,0.55,0.5]
        for sx in range(8): LEAD.add(piano2(c['ar'][sx],vels[sx]*level,1.8),t0+sx*B/2,pan=0.25*(-1)**sx,g=0.17)
    if mel:
        for (bo,bd,fq) in mel:
            LEAD.add(piano2(fq,1.0,2.4),t0+bo*B,g=0.26*mel_g); PAD.add(strings2(fq,bd*B+0.5,0.18,0.5),t0+bo*B,g=0.06*mel_g)
            if oct2: LEAD.add(piano2(fq*2,0.8,1.6),t0+bo*B+0.02,g=0.08*mel_g)

# bars: 1.875 s each; downbeats at 4.2 (settle) and 11.7 (banner reveal)
MM={'Fmaj7':0,'Am7':1,'Bbmaj7':2,'C':3,'Dm7':4,'Gm7':6}
def B_(t0,nm,lvl=1.0,mg=1.0,pad_g=0.05,oct2=False,arp=True,mel=True):
    bar(t0,nm,lvl,arp=arp,mel=MELS[MM[nm]] if mel else None,mel_g=mg,pad_g=pad_g,oct2=oct2)
BL=1.875
SQ1=['Fmaj7','Bbmaj7','C','Gm7','Am7','Dm7','Bbmaj7','C']
n1=max(1,int(round((TB-1.25-4.2)/BL)))
for k_ in range(n1): B_(4.2+k_*BL,SQ1[k_%8],0.7 if k_==0 else min(1.0,0.8+0.05*k_),0.8 if k_==0 else 1.0,arp=(k_>0))
for fq in (196.0,261.63,329.63,392.0,523.25): PAD.add(strings2(fq,1.6,1.0,0.1),TB-1.6,g=0.09)
DRUM.add(timp(60.0,2.2),TB,g=0.8); BASS.add(sub(43.65,2.4),TB,g=0.6)
for j,fq in enumerate((261.63,329.63,392.0,523.25,659.25,783.99,1046.5)): LEAD.add(harp(fq,2.0),TB+0.05+0.08*j,pan=-0.3+0.1*j,g=0.14)
SQ=['Fmaj7','Bbmaj7','Am7','C','Dm7','Bbmaj7','Gm7','C']
n2=max(1,int(round((T-1.7-TB)/BL)))
for k_ in range(n2): B_(TB+k_*BL,SQ[k_%8],1.05 if k_%2==0 else 1.0,1.1,pad_g=0.07,oct2=(k_<4))
tf=TB+n2*BL
for fq in (174.61,220.0,261.63,349.23,440.0): PAD.add(strings2(fq,2.4,0.8,1.3),tf,g=0.09)
for fq in (174.61,261.63,329.63,440.0,523.25): LEAD.add(piano2(fq,0.9,3.0),tf,g=0.20)
LEAD.add(piano2(659.25,0.9,2.6),tf+0.1,g=0.22)
BASS.add(sub(43.65,2.0),tf,g=0.5)
for j,fq in enumerate((523.25,659.25,783.99,1046.5,1318.5)): LEAD.add(harp(fq,2.0),tf+0.1*j,pan=-0.3+0.15*j,g=0.12)
# ================= mix =================
def reverb(x,sec=3.6,seed=1):
    r=np.random.RandomState(seed); m=int(sec*SR)
    ir=r.randn(m)*np.exp(-np.arange(m)/SR*2.8); ir[:int(0.012*SR)]*=0; ir=lp(ir,6500)
    return sg.fftconvolve(x,ir)[:len(x)]/np.abs(ir).sum()*40
fade=np.clip((T-t)/1.4,0,1)
mus=[None,None]
for ch,seed in ((0,1),(1,2)):
    g=lambda b:(b.L if ch==0 else b.R)
    dry=g(LEAD)+g(PAD)+g(DRUM)+g(BASS)
    wet=reverb(g(LEAD)*0.6+g(PAD)*0.5,seed=seed)
    mus[ch]=(dry+wet*0.6)*fade
pk=max(np.abs(mus[0]).max(),np.abs(mus[1]).max())
mL=mus[0]/pk*0.62; mR=mus[1]/pk*0.62
def fin(a,b):
    m=max(np.abs(a).max(),np.abs(b).max()); return a/m,b/m
L1,R1=fin(SFX.L*0.6+mL,SFX.R*0.6+mR)
L2,R2=fin(SFX.L,SFX.R)
wf.write(os.path.join(OUTD,'music.wav'),SR,(np.stack([L1,R1],1)*0.7*32767).astype(np.int16))
wf.write(os.path.join(OUTD,'sfx.wav'),SR,(np.stack([L2,R2],1)*0.7*32767).astype(np.int16))
print('ok')
'''


WORKFLOW = r'''name: Make intro videos
on:
  workflow_dispatch:
  push:
    paths: ['banners/**']
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: '3.11' }
      - name: Install system packages
        run: sudo apt-get update && sudo apt-get install -y ffmpeg libraqm0 libfribidi0 tesseract-ocr tesseract-ocr-fas
      - run: pip install numpy scipy opencv-python-headless Pillow pytesseract
      - name: Render
        run: python make_intro.py --all banners --out videos
      - uses: actions/upload-artifact@v4
        with:
          name: intro-videos
          path: videos/*.mp4
'''

# فایل‌های لازم (فونت B-Nazanin و تصویر سکه) به‌صورت base64
FONT_B64 = (
    'AAEAAAAQAQAABAAAR1NVQiToNn4AAAEMAAAG1k9TLzK2Rt/mAAAH5AAAAFZQQ0xUfEaxzgAACDwAAAA2Y21hcBsEsYcAAAh0AAAD'
    '7mN2dCBpY8t2AAAMZAAAAHRmcGdtgzPCTwAADNgAAAAUZ2x5ZlPI8rIAAAzsAADCemhkbXh3Ey5XAADPaAAADchoZWFk0r8HxwAA'
    '3TAAAAA2aGhlYQw1A6MAAN1oAAAAJGhtdHiicEQAAADdjAAAA2Rsb2NhAEcGGAAA4PAAAANobWF4cAGGAh0AAORYAAAAIG5hbWXT'
    'vWKMAADkeAAAApFwb3N0duxTjwAA5wwAAAYOcHJlcLAxLGMAAO0cAAAAdQABAAAACgAmAGgAAWFyYWIACAAEAAAAAP//AAUAAAAB'
    'AAIAAwAEAAVpbml0ACBtZWRpACZmaW5hACxsaWdhADJtc2V0ADoAAAABAAAAAAABAAEAAAABAAIAAAACAAMABAAAAAIABQAHAAkA'
    'FACSAQwBygIyAnoEKARqBlQAAQABAAEACAACADwAGwBXAF0AYQBlAGkAbQBxAHUAeQB7AIcAiwCPAJMAlwCbAJ8AowCnAKsArwCz'
    'ALcAuwC/AMUAywABABsAVQBbAF8AYwBnAGsAbwBzAHcAewCFAIkAjQCRAJUAmQCdAKEApQCpAK0AsQC1ALkAvQDDAMkAAQABAAEA'
    'CAACADoAGgBYAF4AYgBmAGoAbgByAHYAegCIAIwAkACUAJgAnACgAKQAqACsALAAtAC4ALwAwADGAMwAAQAaAFUAWwBfAGMAZwBr'
    'AG8AcwB3AIUAiQCNAJEAlQCZAJ0AoQClAKkArQCxALUAuQC9AMMAyQABAAEAAQAIAAIAXAArAE4AUABSAFQAVgBaAFwAYABkAGgA'
    'bABwAHQAeAB8AH4AgACCAIQAhgCKAI4AkgCWAJoAngCiAKYAqgCuALIAtgC6AL4AwgDEAMgAygDUAM4A0ADSANYAAQArAE0ATwBR'
    'AFMAVQBZAFsAXwBjAGcAawBvAHMAdwB7AH0AfwCBAIMAhQCJAI0AkQCVAJkAnQChAKUAqQCtALEAtQC5AL0AwQDDAMcAyQDTAM0A'
    'zwDRANUABAAJAAEACAABAFgAAgAKADYABQAMABIAGAAeACQAzQACAE4AzwACAFAA0QACAFQA0wACAFoA1wADALgAxAAEAAoAEAAW'
    'ABwAzgACAE4A0AACAFAA0gACAFQA1AACAFoAAQACALcAuAAEAAcAAQAIAAEAOgABAAgABgAOABQAGgAgACYALABIAAIAMABJAAIA'
    'MQBGAAIAMwBHAAIANABKAAIANQBLAAIAMgABAAEANgAFAAEAAQAIAAIAHAE2AAMAAAAAAA4AAQAEAAIAAQABAAEABgABAIsATABN'
    'AE4ATwBQAFEAUgBTAFQAVQBWAFcAWABZAFoAWwBcAF0AXgBfAGAAYQBiAGMAZABlAGYAZwBoAGkAagBrAGwAbQBuAG8AcABxAHIA'
    'cwB0AHUAdgB3AHgAeQB6AHsAfAB9AH4AfwCAAIEAggCDAIQAhQCGAIcAiACJAIoAiwCMAI0AjgCPAJAAkQCSAJMAlACVAJYAlwCY'
    'AJkAmgCbAJwAnQCeAJ8AoAChAKIAowCkAKUApgCnAKgAqQCqAKsArACtAK4ArwCwALEAsgCzALQAtQC2ALcAuAC5ALoAuwC8AL0A'
    'vgC/AMAAwQDCAMMAxADFAMYAxwDIAMkAygDLAMwAzQDOAM8A0ADRANIA0wDUANUA1gACABIATABMAAIATQBaAAEAVwBkAAIAawB2'
    'AAIAewB8AAIAfwB/AAIAgQCBAAIAhQCKAAIAjQCSAAIAnQCgAAIAogCiAAEApQCmAAIAsQCyAAIArQCuAAIAuQC8AAIAwQDGAAIA'
    'yQDMAAIA1QDWAAIAAQABAAEACAACAB4ADAA7ADwAOAA5AD0AOgBCAEMARQBAAEEARAABAAwAMAAxADMANAA2ADcASABJAEsARgBH'
    'AEoABQABAAEACAACABwBNgADAAAAAAAOAAEABAACAAEAAQABAAgAAQCLAEwATQBOAE8AUABRAFIAUwBUAFUAVgBXAFgAWQBaAFsA'
    'XABdAF4AXwBgAGEAYgBjAGQAZQBmAGcAaABpAGoAawBsAG0AbgBvAHAAcQByAHMAdAB1AHYAdwB4AHkAegB7AHwAfQB+AH8AgACB'
    'AIIAgwCEAIUAhgCHAIgAiQCKAIsAjACNAI4AjwCQAJEAkgCTAJQAlQCWAJcAmACZAJoAmwCcAJ0AngCfAKAAoQCiAKMApAClAKYA'
    'pwCoAKkAqgCrAKwArQCuAK8AsACxALIAswC0ALUAtgC3ALgAuQC6ALsAvAC9AL4AvwDAAMEAwgDDAMQAxQDGAMcAyADJAMoAywDM'
    'AM0AzgDPANAA0QDSANMA1ADVANYAAgAcAEwAUAACAFEAVgABAFcAWgACAFsAYgABAGMAagACAGsAbgABAG8AdAABAHUAdgACAHcA'
    'eAABAHkAfgACAH8AhgABAIcAiAACAIkAigABAIsAjAACAI0AjgABAI8AkAACAJEAkgABAJMAlAACAJUAnAACAJ8AoAACAKMAqAAC'
    'AKsAtAACALcAuAACALsAvAACAL8AwAACAMMAxQACAM0A0AACANMA1AACAAEAAQABAAgAAgAKAAIAPwA+AAEAAgAyADUAAAABAAAB'
    'kAAFAAAFmgUzAAABJQWaBTMAAAOgAGYCEgAAAAAEAAAAAAAAAAAAYACAAAAAAAAACAAAAABNWjczAEAAIP78B8b76wAAB8YEFQAA'
    'AEAAAAAAAAAAAQAAgAAAAAH0BAAAAGAABT8CdUIgTmF6YW5pbiAgICAgICD/////N////kIgTlIwMAAAAAAAAAAAAAIAAQAAAAAA'
    'FAADAAEAAAEaAAABBgAAAQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADBAAAAAUAAAYHCAkKCwwNDg8QERITFBUWFxgA'
    'ABkAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABoAGwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAHAAdAAAAAAAAAAAA'
    'AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAHgAAAAAAAAAAAAAA2AAAAB8AAAAAAAAAAAAAAAAAAAAAAAAAAAAA'
    'AAAAAAAgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACEAAAAAAAAAAAAEAtQAAABaAEAABQAaACEAJQA6AD0AWwBdAHsA'
    'fQCrALcAuwDXAPcC2QYMBhsGHwY6BlIGaQZ+BoYGmAapBq8GwAbMBvkgDyAZIB0gOiIZ6BjoLftZ+337i/uV+6X7//xi/fL+/P//'
    'AAAAIAAlACgAPQBbAF0AewB9AKsAtwC7ANcA9wLZBgwGGwYfBiEGQAZgBn4GhgaYBqkGrwbABswG8CAMIBggHCA5IhnoGOgg+1b7'
    'evuK+477pPv8/F798v6B////4//g/97/3P+//77/of+g/3MAIf9k/0n/Kv3/+iD6EvoPAAAAAPmu+eH56fnr+gT6AvoH+f35HuAW'
    '4A7gDN/x3r8YMBgYBQkE9QT5BR8FIwTNAAAC5QAAAAEAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADgAagAAAAAA'
    'AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAGIAAABoAAAATABNAE8AUQBTAFUAWQBbAMcAYwBnAGsAcwB3'
    'AHsAfQB/AIEAhQCJAI0AkQCVAJkAnQChAC8ApQCpAK0AtQC5AL0AwwDBANUAyQAwADEAMgAzADQANQA2ADcASQBLAEYARwBKAE0A'
    'TgBPAFAAUQBSAFMAVABVAFYAVwBYAFkAWgBbAFwAXQBeAMcAyABjAGQAZQBmAGcAaABpAGoAawBsAG0AbgBzAHQAdQB2AHcAeAB5'
    'AHoAewB8AH0AfgB/AIAAgQCCAIUAhgCHAIgAiQCKAIsAjACNAI4AjwCQAJEAkgCTAJQAlQCWAJcAmACZAJoAmwCcAJ0AngCfAKAA'
    'oQCiAKMApAClAKYApwCoAKkAqgCrAKwArQCuAK8AsAC1ALYAtwC4ALkAugC7ALwAvQC+AL8AwADDAMQAxQDGAMEAwgDVANYAyQDK'
    'AMsAzADNAM4AzwDQANEA0gDTANQAAAAAAF0AeQCpAOMArQBRAXUBPwLzAzQD8wINA4YArgJRAwMC1gN4AzQD619pX2lfaV9pX2lf'
    'aV9pX2lfaV9pX2lfaV9pX2lfaV9pX2lfaV9pX2lfaV9pX2lfaV9pX2lfaV9pX2lfaV9pX2lfaV9pX2kAIwAlQAEALHZFILADJUUj'
    'YWgYI2hgRC0AAgCAAAADgAU/AAMABwBWQCABCAhACQIHBAIBAAYFAgMCBQQDAAcGAwECAQMAAAEARnYvNxgAPzwvPBD9PBD9PAEv'
    'PP08Lzz9PAAxMAFJaLkAAAAISWhhsEBSWDgRN7kACP/AOFkzESERJSERIYADAP2AAgD+AAU/+sGABD8AAgCa//4BSgSDABEAHQBO'
    'QBwBHh5AHxIHBg4HAQYGAQkYAgASGwMVEBUAAQ5Gdi83GAA/LxD9AS88/S/9EP0uAC4uMTABSWi5AA4AHkloYbBAUlg4ETe5AB7/'
    'wDhZARQHBgcGByMmLwEmJyY1NDMyExQGIyImNTQ2MzIWAUgYGgUJAyMDCQIEHhhXVwIyIyIxMCMjMgOsPJ+qUY1fUX4cPMSePNT7'
    'ziMwMSIjMTEAAwBF/4YDMwTAAAoADgAaAFlAIAEbG0AcDw0LDQ4GDAsLDAACBRUCDwIDCBgDEg4MAQVGdi83GAAvLy/9L/0BL/0v'
    '/YcuDsQO/A7EAS4uADEwAUlouQAFABtJaGGwQFJYOBE3uQAb/8A4WRMUIyImNTQ2MzIWJQEnARMUBiMiJjU0NjMyFudSIi4vISIw'
    'AeP+MjgB0Z4uIiIvLyIiLgQAUS8iIS8uivraFAUm+4YiLy8iIi8vAAABAJz/AAI+BCgAGQA6QBABGhpAGwoKAAQCEhgMARJGdi83'
    'GAAvLwEv/S4uADEwAUlouQASABpJaGGwQFJYOBE3uQAa/8A4WQEGBwIVFBcWFxYVFCMiJyYnJjU0NzY3NhcWAj0OOctnIoQGHxta'
    'bUVcbmWrDQsLBAwXUf7f68C+P7MICRdYan+pqcGzo3UJBQUAAAEAm/78Aj0EJAAZADpAEAEaGkAbEgoABAISDBgBCkZ2LzcYAC8v'
    'AS/9Li4AMTABSWi5AAoAGkloYbBAUlg4ETe5ABr/wDhZFzY3EjU0JyYnJjU0MzIXFhcWFRQHBgcGJyacCj3MZyKFBiAbWW1FXG1k'
    'rQwLDOkSVgEg7cG8PrQICRhYbH6pqcSwonYIBQUAAAkAm//bBBMDUwAQAB0AKgA2AEUAVABhAG4AfQAAAQYHBiMiJyY3Njc2MzIW'
    'FRQlFAcGIyYnJjU0MzIWARQGIyInJjU0NzYzMgUUBiMiJjU0NjMyFhcWFRQGIyInJicmNzYXFiUGJyYnJjU0NjMyFxYXFgcGBwYj'
    'IjU0NjMyFxYTFCMiJjU0NzYzFhcWJwYHBiMiJjU0NzY3NhcWA5VRgwQGAwEDAydGGB8iMP8ASgUFBQVFUx4yAWotHW14CQl5aE7+'
    'ljEhITAvIiIw6RgyISITSyIDAwQKi/6rAwuKRhcxIiMSSiMDKAIJd2lOLR1seArVUx0zSgYEBQVFrCdIGB8iLlZISgsDAQJiTCAB'
    'AQILjUMXMSIilGx4CwIJeWdOLf5vHTJJBwMFBUVRIS8vISIvL+sZHyEwFFCECwQDAyfeAwMmSBgfIi8TTIkLYQUFRFIeMkoG/pFP'
    'Lh1reQoBCXqbiUYXMCIzMioTAwMEAAEANwA3AsMCwgALAGVAKwEMDEANAAsADgEGBQ4DCAcEAwMBCgkCAwELCgcDBgMFBAEDAAkI'
    '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'
    'ETe5AAj/wDhZBSERIRUjETMCG/67AUW2tmUGbT76DwAAAQCH/5sBowYGAAcAV0AhAQgIQAkABAMCAAcAAgYFAgMBAwIDAAUEAwYH'
    'BgEAAQFGdi83GAAvPC88EP08EP08AS8XPP08EP08ADEwAUlouQABAAhJaGGwQFJYOBE3uQAI/8A4WQUhNTMRIzUhAaP+5I2NARxl'
    'PgXwPQABAR/+wANABgYAKgBWQCABKytALAAqGBcAKAIEJAIICA8CIBwCEyIBDQwXAAEMRnYvNxgALy8BLzz9L/0v/TwQ/S/9Li4u'
    'LgAxMAFJaLkADAArSWhhsEBSWDgRN7kAK//AOFkBJicmNTQ3NjU0JyYnNTY1NCcmNTQ3NjcVBgcGFRQXFhUUBQQVFAcGFRQXA0CY'
    'YWcaE0hDY+8QG2VgmF49Qw0e/uIBGxsR4v7AG2Zrlj10VRthQj4LJjSnL0t+PpdqZhkoDjtAWy5FoDLgcHTqMYJSNbQ1AAABAJb+'
    'wAK3BgYAKgBWQCABKytALAAeHQsKDwIGEwICAiYCFxsCIhUBKgAeCgEKRnYvNxgALy8BLzz9L/0v/TwQ/S/9Li4uLgAxMAFJaLkA'
    'CgArSWhhsEBSWDgRN7kAK//AOFkBBhUUFxYVFAcGBzU2NzY1NCcmNTQlJDU0NzY1NCc1FhcWFRQHBhUUFxYXArfvDxxlYJhePUIM'
    'HgEd/uYaEuKYYWcbEklAZQJRNKg0RoM4l2pmGSgOOz9cMEKkL99xdOk3fVcwtDUkHGVqlzl4UCFhQToQAAACAEYAgQMkA5oABQAL'
    'AE1AGwEMDEANAAsKCAYFBAIACgkEAwMHBgEDAAEIRnYvNxgALxc8Lxc8AS4uLi4uLi4uADEwAUlouQAIAAxJaGGwQFJYOBE3uQAM'
    '/8A4WSUjCQEzCwEjCQEzAwMkXf6sAVRX4ktZ/q4BVFfigQGQAYn+d/5wAZABif53AAACADIAgQMQA5oABQALAE1AGwEMDEANAAoJ'
    'CAYEAwIACwoFAwQIBwIDAQEIRnYvNxgALxc8Lxc8AS4uLi4uLi4uADEwAUlouQAIAAxJaGGwQFJYOBE3uQAM/8A4WQkBIxMDMxMB'
    'IxMDMwMQ/q5Z4uJXJ/6sXejiVwIR/nABkAGJ/nf+cAGQAYkAAQBJADUCuQKkAAsAmEBHAQwMQA0ACAILCgYFBAALCwACAQIKCQoD'
    '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'
    'HzEvGxQ3JiY/RUATCQgHDAwOFCYZFgkDAQEDAAACAJv+XgUvAtgANwBcAG1ALAFdXUBeAllUSjseAltQSEA+PDsgAiQBGiwBEDIB'
    'CgYDNigDFkwDREQWARpGdi83GAAvLxD9EP0v/QEv/S/9L/0uLi4uLi4uLi4ALi4uLi4uMTABSWi5ABoAXUloYbBAUlg4ETe5AF3/'
    'wDhZARYHBicmIyIHBhUUFxYXFhUUBwYHBiMiJyY1NDc2NzYVFAcGFRQXFjMyNzY1NCcmJyY1NDc2MzIFBwYHNTY3JjU0NzYzMhcW'
    'BwYnJiMiBwYVFBcWMzI/ATY3FhcWBScIBQQcO1t+bFmywyEHQyyUt+ymbnl1EhEQC0tBWa+crfr1WyEvgoyJlfzhEIyTLChRNTo1'
    'QiAJBgcUITIXGR8zKikgIQsGBAQBAgHVJAUFECFrWTQ+ICQ6DBtpa0c7SVVeobOaFwUEEAwUindXRl8nOGUmLxITHEN9qLVnPxs1'
    'LxsUNyYmP0VAEwkIBwwMDhQmGRYJAwEBAQEDAAIAm/3MBRgCdwAuAFMAc0AvAVRUQFUAUEtBMhNSRz83NTMyLicVBQAZAQ4hAQgu'
    'LQMAHQMMQwM7OwwBAAABDkZ2LzcYAD88Ly8Q/RD9EP08AS/9L/0uLi4uLi4uLi4uLi4ALi4uLi4xMAFJaLkADgBUSWhhsEBSWDgR'
    'N7kAVP/AOFklISIHBhUUBBUUBwYhIBE0Njc2NzYVFAcGFRQXFjMyNzY1NCcmJyY1NDc2NzY7AQEHBgc1NjcmNTQ3NjMyFxYHBicm'
    'IyIHBhUUFxYzMj8BNjcWFxYFGP7cNjJFAXFXk/6N/kBALhIPDww6fF+Sm4HoZaEZZTEpLVqmsfzOEIyTGTtRNTo1Qx8JBgcUJC8W'
    'Gh8zKikgIQsGBAQBAgERFyoKUiiXSn4BS0u3PBgBAREPF252gz8wEB1GHREbBxw6LkU5JUkBDD8bNS8PIDcnJT9FQBMJCAcNDQ8T'
    'JhkWCQMBAQEBAwAC/+YAAQFMA94ADwAyAGhAKQEzM0A0ADEsIhMMMSggGBYUExAJBAMCAAcBAAUEAwIkAxwcAwIAAQNGdi83GAA/'
    'PC8Q/RD9PAEv/RD9PC4uLi4uLi4uLgAuLi4uLjEwAUlouQADADNJaGGwQFJYOBE3uQAz/8A4WSUUKwE1MzI1NCc3NjMyFxYDBwYH'
    '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'
    'KiwsLSMiIyQFJCUiIiMlJSInJicoBSgpJiYnKSkmLSwtKgUqKywsLSsrLBMBCSEgAwAVAwclBwEAAAEJRnYvNxgAPzwvLxD9EP08'
    'AS/9hy4IxAj8CMSHLgjECPwIxIcuCMQI/AjEhy4IxAj8CMSHLgjECPwIxIcuCMQI/AjEAS4uLi4uLi4uLi4uAC4uLi4uLi4uMTAB'
    'SWi5AAkALkloYbBAUlg4ETe5AC7/wDhZJSMiJyYnBiEgETQ3NjMyFRQHBhUUISA3NjMyFRQGFRQ7AQEHJzcTByc3DwEnNwYBNlo5'
    'QQHV/kT+NkcNDw4FHAGgAeWfIBEMDo02/VJdX17RXl5dhV1fXgEpLlbFARZeZRMSCAxAM7/TKhgRRBFnArVdXl3+2l1eXV5dXl0A'
    'AAT/5gABAaQEPgAPABMAFwAbAOBAcwEcHEAdFBsZFxURDBoYFhQSEAkQExARBRESExMQEhITFBcUFQUVFhcXFBYWFxgbGBkFGRob'
    'GxgaGhsREBESBRITEBARExMQFRQVFgUWFxQUFRcXFBkYGRoFGhsYGBkbGxgEAwIABwEABQQDAhMDAgABA0Z2LzcYAD88LxD9PAEv'
    '/RD9PIcuCMQI/AjEhy4IxAj8CMSHLgjECPwIxIcuCMQI/AjEhy4IxAj8CMSHLgjECPwIxAEuLi4uLi4uAC4uLi4uLjEwAUlouQAD'
    'ABxJaGGwQFJYOBE3uQAc/8A4WSUUKwE1MzI1NCc3NjMyFxYDByc3EwcnNw8BJzcBTP1pV79XNAgFBRBRG15eXtFeXl2FXV9e//6U'
    'RTBvfhMXdwJkXl9d/tpdXl1eXV5dAAAE/+YAAQKQA6gAEQAVABkAHQDlQHcBHh5AHwAdGxkXExwaGBYUEhEHBgASFRITBRMUFRUS'
    'FBQVFhkWFwUXGBkZFhgYGRodGhsFGxwdHRocHB0TEhMUBRQVEhITFRUSFxYXGAUYGRYWFxkZFh0cHRoFGhscHB0bGxwREAgDBwMA'
    'DAQDFQYFAQMAAAEGRnYvNxgAPxc8Ly/9EP0XPAGHLgjECPwIxIcuCMQI/AjEhy4IxAj8CMSHLgjECPwIxIcuCMQI/AjEhy4IxAj8'
    'CMQBLi4uLi4uLi4uLgAuLi4uLjEwAUlouQAGAB5JaGGwQFJYOBE3uQAe/8A4WSUjIicGKwE1MzI3NjMyFxY7AQEHJzcTByc3DwEn'
    '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'
    'FhcWFxY7AQMcNkYkAnVlQJJqIySWmKKGOwcFBgs7IDxSKAEYFBS0p5APBQwMBx+AiX5UgngPAQIPThovAAIARf4GAoUDDAAYABwA'
    'bUAuAR0dQB4AGhMbGRAIGRwZGgUaGxwcGRsbHBoZGhsFGxwZGRocHBkOAQAcBAEIRnYvNxgALy8BL/2HLgjECPwIxIcuCMQI/AjE'
    'AS4uLi4ALi4xMAFJaLkACAAdSWhhsEBSWDgRN7kAHf/AOFkFFAcGIyInJjU0NzY3NjU0Jzc2NxYXFhcWAwcnNwKFd2ZEkmojJJaY'
    'ooY6CAQGCy4lKF5oaGgLu6aODwUMDAcfgIl+VIJ5EAECDjhdYgJsaGlnAAACAEX+BgMcAwwAHwAjAIRAOwEkJEAlACEYAyIgFQ0g'
    'IyAhBSEiIyMgIiIjISAhIgUiIyAgISMjIB8AAhMTAQMFHx4DACMJAQAAAQ1Gdi83GAA/PC8vEP08AS88/RD9PIcuCMQI/AjEhy4I'
    'xAj8CMQBLi4uLgAuLi4xMAFJaLkADQAkSWhhsEBSWDgRN7kAJP/AOFklIyInFhUUBwYjIicmNTQ3Njc2NTQnNzY3FhcWFxY7AQMH'
    'JzcDHDZGJAJ1ZUCSaiMklpiihjsHBQYLOyA8Uij1aGhoARgUFLSnkA8FDAwHH4CJflSCeA8BAg9OGi8CD2hpZwAEAEX+BgKFA6EA'
    'GAAcACAAJADRQGoBJSVAJgAkIiAeGhMjIR8dGxkQCBkcGRoFGhscHBkbGxwdIB0eBR4fICAdHx8gISQhIgUiIyQkISMjJBoZGhsF'
    'GxwZGRocHBkeHR4fBR8gHR0eICAdIiEiIwUjJCEhIiQkIQ4BABwEAQhGdi83GAAvLwEv/YcuCMQI/AjEhy4IxAj8CMSHLgjECPwI'
    'xIcuCMQI/AjEhy4IxAj8CMSHLgjECPwIxAEuLi4uLi4uLgAuLi4uLi4xMAFJaLkACAAlSWhhsEBSWDgRN7kAJf/AOFkFFAcGIyIn'
    'JjU0NzY3NjU0Jzc2NxYXFhcWAwcnNxMHJzcPASc3AoV3ZkSSaiMklpiihjoIBAYLLiUokl5eXdJeX16FXl9fC7umjg8FDAwHH4CJ'
    'flSCeRABAg44XWIDC11eXf7aXV5dXl1eXQAEAEX+BgMcA6EAHwAjACcAKwDoQHcBLCxALQArKSclIRgDKigmJCIgFQ0gIyAhBSEi'
    'IyMgIiIjJCckJQUlJicnJCYmJygrKCkFKSorKygqKishICEiBSIjICAhIyMgJSQlJgUmJyQkJScnJCkoKSoFKisoKCkrKygfAAIT'
    'EwEDBR8eAwAjCQEAAAENRnYvNxgAPzwvLxD9PAEvPP0Q/TyHLgjECPwIxIcuCMQI/AjEhy4IxAj8CMSHLgjECPwIxIcuCMQI/AjE'
    'hy4IxAj8CMQBLi4uLi4uLi4ALi4uLi4uLjEwAUlouQANACxJaGGwQFJYOBE3uQAs/8A4WSUjIicWFRQHBiMiJyY1NDc2NzY1NCc3'
    'NjcWFxYXFjsBAQcnNxMHJzcPASc3Axw2RiQCdWVAkmojJJaYooY7BwUGCzsgPFIo/tdeXl3SXl9ehV5fXwEYFBS0p5APBQwMBx+A'
    'iX5UgngPAQIPThovAq5dXl3+2l1eXV5dXl0AAQCb/fcG7wHtAEwAYEAmAU1NQE4ALxgIRiwoGh4BEkQBAEADAgYDNSIDDgQEO0kO'
    'AgABEkZ2LzcYAD8vLy/9EP0v/RD9AS/9L/0uLi4uAC4uLjEwAUlouQASAE1JaGGwQFJYOBE3uQBN/8A4WQEQIyInBiMiJwYHBgcG'
    'IyInJjU0NzY3NjMyFRQHBhUUFxYzMjc2NzYnJicmJzc2MzIXFhcWMzI3Njc2MzIXHgEzMjc2NTQnNzY3FhcWBu/kfi4/eCwzBAoz'
    'qZGpqGx2Kx8hEA0MBjk/U6i+jXQeCgcHDBdlOAwGBQ01FDZMVSYZDwcRDwQMWGAoHSFKQQoFBgo/AQ3+8XBzEHOBj05DU1ykV2NH'
    'KBMOCQx1YV9FW0Q4QhY/QB88gmoXE1ATNS4fTiYiXUYVFyY3U28RAQMPYQAAAQCb/fcHpQFTAEsAakArAUxMQE0AMhsLSy8rHQAh'
    'ARVLSgMACQUDOEIlAxEDBEYHBD4+EQEAAAEVRnYvNxgAPzwvLxD9L/0Q/S88/TwQ/TwBL/0uLi4uLgAuLi4xMAFJaLkAFQBMSWhh'
    'sEBSWDgRN7kATP/AOFklIyInBiMiJwYjIicGBwYHBiMiJyY1NDc2NzYzMhUUBwYVFBcWMzI3Njc2JyYnJic3NjMyFxYXFjMyNzY3'
    'NjMyFxYzMjc2MzIXFjsBB6VGgTUufHQuP3YsMwQKM6mRqahsdisfIRANDAY5P1Oovo10HgoHBwwXZTgMBgUNNRQ2TFUmGQ8HEBAE'
    'FXaOEAQQEgUUg0oBaW5ycxBzgY9OQ1NcpFdjRygTDgkMdWFfRVtEOEIWP0AfPIJqFxNQEzUuHk8kIKGYJiWWAAAB/+YAAQQLAfMA'
    'KwBXQCIBLCxALQAlDg0jAQAhFw8DDgMEHRMECgYoDQwIAwQAAQ1Gdi83GAA/FzwvLzz9PBD9FzwBL/0uLi4AMTABSWi5AA0ALElo'
    'YbBAUlg4ETe5ACz/wDhZARQHBiMiJwYjIicGKwE1MzI3NjMyFxYzMjc2NzYzMhcWMzI1NCc3NjMyFxYECzU8bYA4R22ML0l+WWaL'
    'GQYSEAQPklYkGQsFEBEFFIaSSDoOBgYQPQESckpVaWlpaZSWJSaVLB9KJCOWRTdbbRodbwAAAf/mAAEEtwFQACkAXEAmASoqQCsA'
    'KQ8OACkoIBgQBQ8DAAsHAwQUJBwUDg0JBQEFAAABDkZ2LzcYAD8XPC88PBD9PDwQ/Rc8AS4uLi4AMTABSWi5AA4AKkloYbBAUlg4'
    'ETe5ACr/wDhZJSMiJwYjIicGIyInBisBNTMyNzYzMhcWMzI3NjMyFxYzMjc2MzIXFjsBBLdHgTQufGw6NnSML0l+WWaLGQYSEAQP'
    'ko4QBBASBRRyjhAEEBIFE4VKAWlpaWlpaZSWJSaVlSYllpUmJZYABACb/fcG7wQLAEwAUABUAFgA90CAAVlZQFoAWFZUUk5JLxgI'
    'V1VTUU9NRiwoGk1QTU4FTk9QUE1PT1BTUlNUBVRRUlJTUVFSVVhVVgVWV1hYVVdXWE5NTk8FT1BNTU5QUE1SUVJTBVNUUVFSVFRR'
    'VlVWVwVXWFVVVlhYVR4BEkQBAEADAjUDBiIDDjsEBFAOAgABEkZ2LzcYAD8vLy/9EP0v/RD9AS/9L/2HLgjECPwIxIcuCMQI/AjE'
    'hy4IxAj8CMSHLgjECPwIxIcuCMQI/AjEhy4IxAj8CMQBLi4uLi4uLi4uLgAuLi4uLi4uLi4xMAFJaLkAEgBZSWhhsEBSWDgRN7kA'
    'Wf/AOFkBECMiJwYjIicGBwYHBiMiJyY1NDc2NzYzMhUUBwYVFBcWMzI3Njc2JyYnJic3NjMyFxYXFjMyNzY3NjMyFx4BMzI3NjU0'
    'Jzc2NxYXFgEHJzcTByc3DwEnNwbv5H4uP3gsMwQKM6mRqahsdisfIRANDAY5P1Oovo10HgoHBwwXZTgMBgUNNRQ2TFUmGQ8HEQ8E'
    'DFhgKB0hSkEKBQYKP/7JXV9e0V5eXYVdX14BDf7xcHMQc4GPTkNTXKRXY0coEw4JDHVhX0VbRDhCFj9AHzyCahcTUBM1Lh9OJiJd'
    'RhUXJjdTbxEBAw9hAjNeX13+2l5fXV5eX10AAAQAm/33B6UDqgBLAE8AUwBXAP9AhAFYWEBZAFdVU1FNMhsLVlRSUE5MSy8rHQBM'
    'T0xNBU1OT09MTk5PUFNQUQVRUlNTUFJSU1RXVFUFVVZXV1RWVldNTE1OBU5PTExNT09MUVBRUgVSU1BQUVNTUFVUVVYFVldUVFVX'
    'V1QhARVLSgMAOEIDCQUlAxEDBEYHBD5PEQEAAAEVRnYvNxgAPzwvLy/9L/0Q/S88/TwQ/TwBL/2HLgjECPwIxIcuCMQI/AjEhy4I'
    'xAj8CMSHLgjECPwIxIcuCMQI/AjEhy4IxAj8CMQBLi4uLi4uLi4uLi4ALi4uLi4uLi4xMAFJaLkAFQBYSWhhsEBSWDgRN7kAWP/A'
    'OFklIyInBiMiJwYjIicGBwYHBiMiJyY1NDc2NzYzMhUUBwYVFBcWMzI3Njc2JyYnJic3NjMyFxYXFjMyNzY3NjMyFxYzMjc2MzIX'
    'FjsBAQcnNxMHJzcPASc3B6VGgTUufHQuP3YsMwQKM6mRqahsdisfIRANDAY5P1Oovo10HgoHBwwXZTgMBgUNNRQ2TFUmGQ8HEBAE'
    'FXaOEAQQEgUUg0r+E11fXtFeXl2FXV9eAWlucnMQc4GPTkNTXKRXY0coEw4JDHVhX0VbRDhCFj9AHzyCahcTUBM1Lh5PJCChmCYl'
    'lgK3Xl9d/tpdXl1eXV5dAAAE/+YAAQQLBAsAKwAvADMANwDuQHwBODhAOQA3NTMxLSg2NDIwLiwlDg0sLywtBS0uLy8sLi4vMjEy'
    'MwUzMDExMjAwMTQ3NDUFNTY3NzQ2NjctLC0uBS4vLCwtLy8sMTAxMgUyMzAwMTMzMDU0NTYFNjc0NDU3NzQjAQAhFw8DDgMEHRME'
    'CgYvDQwIAwQAAQ1Gdi83GAA/FzwvLzz9PBD9FzwBL/2HLgjECPwIxIcuCMQI/AjEhy4IxAj8CMSHLgjECPwIxIcuCMQI/AjEhy4I'
    'xAj8CMQBLi4uLi4uLi4uAC4uLi4uLjEwAUlouQANADhJaGGwQFJYOBE3uQA4/8A4WQEUBwYjIicGIyInBisBNTMyNzYzMhcWMzI3'
    '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'
    'JjU0NzY7ASU0JyYjIgcGFTIDNkJMQ2iZO7N2kAywqU9KLC08Nv75EHpNA0FdoTf+eisuHBojJ9kBGSdccGCGQHOMs5SSt6yikj7+'
    '8QRTmCYbGodehzsmZmxeaGwAAwCb//0CgAQQAA0AGwA9AGJAJgE+PkA/ADw4Lh8SDDw0LCQiIB8cDgEAFAEIMAMoGAMEKAQAAQhG'
    'di83GAA/LxD9EP0BL/0v/S4uLi4uLi4uAC4uLi4uLjEwAUlouQAIAD5JaGGwQFJYOBE3uQA+/8A4WQEUBwYjIicmNTQ3NjcEBzQn'
    'JicGFRQXFjMyNzYDBwYHNTY3JjU0NzYzMhcWBwYnJiMiBwYVFBcWMzI3NhcWAoBRToRRNzpMQlABB0lVTDtkLygwUzktLhCNkhg7'
    'UTU7NEQfCQcHFB80FhofMyopISUSAwMBDoNJRTEzUFmEclCJ7TVQRxZ/TC0cGB8YAnM/GzUvDyE0KSY+RT8TCQkIDA0PEyYZFgoF'
    'AgIAAAMAmwABA2wEnAAXACIARACLQDkBRUVARgBDPzUmGRADQzszKyknJiMdGBcPDgoAGQ4YDwYPEBIQEBIXFgMAIQMGBTcDLy8B'
    'AAABCkZ2LzcYAD88LxD9Lzz9EP08AYcuDsQI/A7EDsQOxAEuLi4uLi4uLi4uLi4uLi4ALi4uLi4uLjEwAUlouQAKAEVJaGGwQFJY'
    'OBE3uQBF/8A4WSUjIicGKwEiJyY1NDc2Nyc3FhMWFxY7ASUnBgcGFRQXFjMyEwcGBzU2NyY1NDc2MzIXFgcGJyYjIgcGFRQXFjMy'
    'NzYXFgNsX6QsWg5xPDtSZkSnGVYFUhkVIEVZ/sErQVp7XkIzP1EQjZIYO1E1OzRFHgkHBxQfNBYaHzMqKSElEgMDAcxZEhktkFo8'
    'QnthEP5teyU4ccsTLDwjHRALAss/GzUvDyE0KSY+RT8TCQkIDA0PEyYZFgoFAgIAAQCb/l4FLwJpADcAUkAeATg4QDkCHgIgAiQB'
    'GiwBEDIBCgYDNigDFjYWARpGdi83GAAvLxD9EP0BL/0v/S/9Li4ALi4xMAFJaLkAGgA4SWhhsEBSWDgRN7kAOP/AOFkBFgcGJyYj'
    'IgcGFRQXFhcWFRQHBgcGIyInJjU0NzY3NhUUBwYVFBcWMzI3NjU0JyYnJjU0NzYzMgUnCAUEHDtbfmxZssMhB0MslLfspm55dRIR'
    'EAtLQVmvnK369VshL4KMiZUB1SQFBRAha1k0PiAkOgwbaWtHO0lVXqGzmhcFBBAMFIp3V0ZfJzhlJi8SExxDfai1AAEAm/3MBRgA'
    'lQAuAFNAHgEvL0AwABMuJxUFABkBDiEBCB0DDC4tDAEAAAEORnYvNxgAPzwvLzwQ/QEv/S/9Li4uLi4ALjEwAUlouQAOAC9JaGGw'
    'QFJYOBE3uQAv/8A4WSUhIgcGFRQEFRQHBiEgETQ2NzY3NhUUBwYVFBcWMzI3NjU0JyYnJjU0NzY3NjsBBRj+3DYyRQFxV5P+jf5A'
    'QC4SDw8MOnxfkpuB6GWhGWUxKS1aprEBERcqClIol0p+AUtLtzwYAQERDxdudoM/MBAdRh0RGwccOi5FOSVJAAP/5v6EAcICCgAP'
    'ABMAFwCuQFUBGBhAGRAXExYUEAkSERITBRMQERESEBARFhUWFwUXFBUVFhQUFRMSExAFEBESEhMRERIXFhcUBRQVFhYXFRUWBAMC'
    'ABIHAQAFBAMCDBURAwIAAQNGdi83GAA/PC88LxD9PAEv/TwQ/TyHLgjECPwIxIcuCMQI/AjEhy4IxAj8CMSHLgjECPwIxAEuLi4u'
    'AC4uMTABSWi5AAMAGEloYbBAUlg4ETe5ABj/wDhZJRQrATUzMjU0Jzc2MzIXFhMHJzcPASc3AUz9aVe/VzQIBQUQUXZjY2N4Y2Ni'
    '//6URTBvfhMXd/1rY2RjZGNkYwAD/+b+hAKQAWAAEQAVABkAtUBaARoaQBsAGRUYFhQSEQcGABQTFBUFFRITExQSEhMYFxgZBRkW'
    'FxcYFhYXFRQVEgUSExQUFRMTFBkYGRYFFhcYGBkXFxgREAgDBwMAAwQMDBcTBgUBAwAAAQZGdi83GAA/FzwvPC8Q/RD9FzwBhy4I'
    'xAj8CMSHLgjECPwIxIcuCMQI/AjEhy4IxAj8CMQBLi4uLi4uLi4ALi4xMAFJaLkABgAaSWhhsEBSWDgRN7kAGv/AOFklIyInBisB'
    'NTMyNzYzMhcWOwEDByc3DwEnNwKQYrE8QsZTcrsRBxYWBg21ccpjZGN4YmRjAXR0lJE6PI/+UmNkY2RjZGMAA//g/88D2wU/ACcA'
    'MQBKAGdAJwFLS0BMAkA6NzAsJiEdFxYFBEk8LiUjGRQRCQIqAQs0A0NHEQE8RnYvNxgALy8v/QEv/S4uLi4uLi4uLi4ALi4uLi4u'
    'Li4uLi4uMTABSWi5ADwAS0loYbBAUlg4ETe5AEv/wDhZARYXBisBFAcGBxYVFAcGBwYnNzY3AicHJjU0NzYzMhcAEzY3Nic3FgM2'
    'NTQnBgcWMzIBBiMiJiMiBiMiNTQ3NjMyFjMyNzYzMhUUA9AEBwQSJSMtVywUGCWrwyq7U73wGYUYBgcHIQF0vkgQDhBsI9cBJEBd'
    'GBZb/sVUexJIEgsuCREWGRIZYhlwNBsLCQPzChYCoKXUd1VMLTQ/CSgOjVdyAQLwKpAiI2EZHv6r/qd60bXLZHT8SAkJNjNNRwIE'
    'uEUKORgRMDYRDwgECAAC/+D/lwSsBT8AMABJAHNALwFKSkBLAD85NiskHhYVBUg7KhoRCwUwAAIDKAEtAwEtMC8DADMDQkYJAQAA'
    'ATtGdi83GAA/PC8vL/0Q/TwBL/0Q/RD9PC4uLi4uLi4ALi4uLi4uLi4uMTABSWi5ADsASkloYbBAUlg4ETe5AEr/wDhZJSMiETQ3'
    'AgcGBwY1NDc2NzY3JicmJwcmJyYnJjc2MzIXFhcWFzY3NicmJzcWExY7AQEGIyImIyIGIyI1NDc2MzIWMzI3NjMyFRQErEzvA5Pg'
    'jcQqeExLhncyek6kDzQkPgECDgMGBhj8e5c1Ry4yAwUUbBAHAZhB/MJUexJIEgsuCREWGRIZYhlwNBsLCQEBUyVO/pOdIQQBBwg6'
    'IyRDfMCWYIUzLCQ+GSJfFhG0mbz6W3R/abLQXV788q0EiUUKORgRMDYRDwgECAAAAwAZ/88D2wYlACcAMQBTAHNALQFUVEBVAlJO'
    'RDUwLCYhHRcWBQRKQjo4NjUyLiUjGRQRCQIqAQtGAz4+EQE1RnYvNxgALy8Q/QEv/S4uLi4uLi4uLi4uLi4uLgAuLi4uLi4uLi4u'
    'Li4uMTABSWi5ADUAVEloYbBAUlg4ETe5AFT/wDhZARYXBisBFAcGBxYVFAcGBwYnNzY3AicHJjU0NzYzMhcAEzY3Nic3FgM2NTQn'
    'BgcWMzIBBwYHNTY3JjU0NzYzMhcWBwYnJiMiBwYVFBcWMzI3NhcWA9AEBwQSJSMtVywUGCWrwyq7U73wGYUYBgcHIQF0vkgQDhBs'
    'I9cBJEBdGBZb/p8QkI8YO1E1OzRFHgkHBxQfNBYaHyEpPCEgFQQFA/MKFgKgpdR3VUwtND8JKA6NV3IBAvAqkCIjYRke/qv+p3rR'
    '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'
    'AAAA3AkJBQACAgEEAwMFAwIDAQIEBAQEBAQEBAQEAQMDAwQEBAQDBAAAAAACAgMDAgICAgMDAAAAAAAAAAAAAAAAAAAAAAAAAAAA'
    'AAAAAAAAAAMCAwICAwMCAgYGAQMBAgYHAQMGBwIDBgcCAwYHAgMFBQUFBQUFBQUFBQUFBQUFAwQDBAMDAwMDAwgIBQUICAUFCAkF'
    'BQgJBQUGBgQFBgYEBQQFAwMEBQMDBgcCAwUGAgMHCAMEBwgDBAUFAQIEBQMEBQYBAwMDAwQEAwMEBgYCAwQFBAUEBQQFBgYGAgAK'
    'CgUAAgICBAMDBQQCBAICBQUFBQUFBAUFBQIEAwMFBQQEBAUAAAAAAgIDAwMDAgIDAwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA'
    'AAADAgMCAwMEAgMGBgIDAQIHBwIDBwcCAwcHAgMHBwIDBQYFBQUGBQUFBgUFBQYFBQQFBAUDBAMEAwQJCQUGCQkFBgkKBgYJCgYG'
    'BgcFBQYHBQUEBQMEBAUDBAcHAwMGBgMDCAgEBAgIBAQFBgICBAUDBAUGAgMDBAMEBAQDBAYGAgMFBgUGBQYFBgYGBwIACwsGAAMD'
    'AgUDAwYEAgQCAgUFBQUFBQUFBQUCBAMDBQUFBQQFAAAAAAMDAwMDAwICBAMAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAwMD'
    'AgMEBAIDBwcCAwIDBwgCAwcIAgMHCAIDBwgCAwYGBgYGBgYGBgYGBgYGBgYEBQQFAwQDBAMECgoGBgoKBgYKCwYHCgsGBwcIBQYH'
    'CAUGBQYEBAUGBAQHCAMEBgcDBAkJBAUJCQQFBgcCAwQGBAUGBwIDBAQDBQQEAwUHBwIDBQYFBgUGBQYHBwgCAAwMBgADAwIFAwMG'
    'BAIEAgIGBgYGBgYFBgYGAgQEBAYGBQUEBgAAAAADAwQEAwMCAgQEAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAQDBAMDBAUD'
    'AwgHAgQCAwgJAgQICQMECAkDBAgJAwQGBwYHBgcGBwYHBgcGBwYHBQYFBgQFBAUEBQoLBgcKCwYHCwwHBwsMBwcICAYGCAgGBgUG'
    'BAQFBgQECAkDBAcHAwQKCgQFCgoEBQYHAgMFBgQFBwcCBAQFBAUFBQQFCAcDBAYHBgcGBwYHCAcIAgANDAcAAwMCBgQEBwUDBQID'
    'BgYGBgYGBgYGBgIFBAQGBgUFBQYAAAAAAwMEBAMDAwMEBAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAEAwQDAwQFAwMICAIE'
    'AgMJCgIECQoDBAkKAwQJCgMEBwcHBwcHBwcHBwcHBwcHBwUGBQYEBQQFBAULDAcHCwwHBwwMBwgMDAcICAkGBwgJBgcGBwQFBgcE'
    'BQkJAwQHCAMECgsFBgoLBQUHCAIDBQcEBgcIAgQEBQQFBQUEBQgIAwQGBwYHBgcGBwgICQMADg0HAAMDAgYEBAcFAwUCAwcHBwcH'
    'BwYHBwcCBQQEBwcGBgUGAAAAAAMDBAQEBAMDBQQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABAMEAwQFBQMECQkCBAIDCQoC'
    'BAkKAwQJCgMECQoDBAcIBwgHCAcIBwgHCAcIBwgFBwUHBAUEBQQFDA0HCAwNBwgNDQgIDQ0ICAkKBwcJCgcHBgcFBQYHBQUJCgQF'
    'CAkEBQsMBQYLDAUGCAgCAwYHBQYICQIEBQUEBgYFBAYJCQMEBwgHCAcIBwgJCQoDAA8OCAAEBAIHBAQIBgMGAgMHBwcHBwcHBwcH'
    'AgYFBQcHBgYGBwAAAAAEBAUFBAQDAwUFAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAUEBAMEBQYDBAoJAgUCBAoLAgUKCwMF'
    'CgsDBQoLAwUICAgICAgICAgICAgICAgIBgcGBwUGBQYFBg0OCAkNDggJDg4ICQ4OCAkKCgcICgoHCAcIBQUHCAUFCgsEBQgJBAUM'
    'DQYGDA0GBggJAgMGCAUHCAkCBQUGBQYGBgUGCgkDBQcJBwkHCQcJCgkKAwAQDwgABAQDBwUFCAYDBgMDCAgICAgIBwgICAMGBQUI'
    'CAcHBgcAAAAABAQFBQQEAwMFBQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAFBAUDBAUGAwQKCgMFAgQLDAMFCwwEBQsMAwUL'
    'DAMFCAkICQgJCAkICQgJCAkICQYIBggFBgUGBQYODwgJDg8ICQ8PCQoPDwkKCgsICAoLCAgHCAYGBwgGBgsMBAUJCgQFDQ4GBw0O'
    'BgcJCgMEBggFBwkKAwUFBgUHBgYFBwoKBAUICQgJCAkICQoKCwMAERAJAAQEAwcFBQkGAwYDBAgICAgICAcICAgDBgUFCAgHBwYI'
    'AAAAAAQEBQUEBAMDBgUAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABQQFBAQFBgQECwsDBQIEDA0DBQwNBAUMDQQFDA0EBQkK'
    'CQkJCgkJCQoJCQkKCQkGCAYIBQYFBgUGDxAJCg8QCQoQEAkKEBAJCgsMCAkLDAgJCAkGBggJBgYLDAQGCgoEBg0OBgcNDgYHCQoD'
    'BAcJBQgJCwMFBQYFBwcHBQcLCwQFCAoICggKCAoLCwwDABIRCQAEBAMIBQUJBwQHAwQICAgICAgICAgIAwcGBgkJCAgHCAAAAAAE'
    'BAYGBQUEBAYFAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAYEBQQFBgcEBQwLAwYDBAwNAwYMDQQGDA0EBgwNBAYJCgkKCQoJ'
    'CgkKCQoJCgkKBwkHCQYHBgcGBxARCQoQEQkKEREKCxERCgsMDAkKDAwJCggJBgYICQYGDA0FBgoLBQYODwcIDg8HCAoLAwQHCQYI'
    'CgsDBgYHBgcHBwYHDAsEBgkKCQoJCgkKDAsMBAATEgoABQUDCAUFCgcEBwMECQkJCQkJCAkJCQMHBgYJCQgIBwkAAAAABQUGBgUF'
    'BAQGBgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAGBQYEBQYHBAUMDAMGAwQNDgMGDQ4EBg0OBAYNDgQGCgsKCgoLCgoKCwoK'
    'CgsKCgcJBwkGBwYHBgcQEgoLEBIKCxESCwsREgsLDA0JCgwNCQoJCgcHCQoHBw0OBQYLDAUGDxAHCA8QBwgKCwMECAoGCAoMAwYG'
    'BwYIBwcGCAwMBAYJCwkLCQsJCwwMDQQAFBMKAAUFAwkGBgoHBAcDBAkJCQkJCQkJCQkDBwYGCgoICAcJAAAAAAUFBgYFBQQEBwYA'
    'AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABgUGBAUGCAQFDQwDBgMFDg8DBg4PBAYODwQGDg8EBgoLCgsKCwoLCgsKCwoLCgsI'
    'CQgJBggGCAYIERMKDBETCgwSEwsMEhMLDA0OCgsNDgoLCQoHBwkKBwcNDwUHCwwFBxARBwkQEQcICwwDBQgKBgkLDAMGBggGCAgI'
    'BggNDAQGCgsKCwoLCgsNDA4EABUUCwAFBQMJBgYLCAQIAwQKCgoKCgoJCgoKAwgHBwoKCQkICgAAAAAFBQcHBQUEBAcGAAAAAAAA'
    'AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAcFBgQGBwgEBQ4NAwYDBQ4PAwYODwUGDg8EBg4PBAYLDAsLCwwLCwsMCwsLDAsLCAoICgcI'
    'BwgHCBIUCwwSFAsMExQMDRMUDA0ODgoLDg4KCwkLBwcJCwcHDg8FBwwNBQcREggJERIICQsNAwUICwcJCw0DBgcIBwkICAcJDg0F'
    'BgoMCgwKDAoMDg0PBAAWFQsABQUECgYGCwgECAQFCgoKCgoKCgoKCgQIBwcLCwkJCAoAAAAABQUHBwYGBAQIBwAAAAAAAAAAAAAA'
    'AAAAAAAAAAAAAAAAAAAAAAAHBQYFBgcIBQYODgQHAwUPEAQHDxAFBw8QBQcPEAUHCwwLDAsMCwwLDAsMCwwLDAgKCAoHCAcIBwgT'
    'FQsNExULDRQVDA0UFQwNDg8LDA4PCwwKCwgICgsICA8QBgcMDgYHERMICRETCAkMDQQFCQsHCgwOBAcHCAcJCQkHCQ4OBQcLDQsN'
    'Cw0LDQ4ODwQAFxYMAAYGBAoGBgwJBQkEBQsLCwsLCwoLCwsECQcHCwsKCgkLAAAAAAYGBwcGBgUFCAcAAAAAAAAAAAAAAAAAAAAA'
    'AAAAAAAAAAAAAAAABwYHBQYHCQUGDw4EBwMFEBEEBxARBQcQEQUHEBEFBwwNDA0MDQwNDA0MDQwNDA0JCwkLBwkHCQcJFBYMDRQW'
    'DA0VFg0OFRYNDg8QCwwPEAsMCgwICAoMCAgPEQYIDQ4GCBITCQoSEwkKDA4EBQkMBwoNDgQHBwkHCgkJBwoPDgUHCw0LDQsNCw0P'
    'DhAFABgXDAAGBgQKBwcMCQUJBAULCwsLCwsKCwsLBAkICAsLCgoJCwAAAAAGBggIBgYFBQgHAAAAAAAAAAAAAAAAAAAAAAAAAAAA'
    'AAAAAAAAAAcGBwUGCAkFBhAPBAcEBhASBAcQEgUHEBIFBxASBQcMDQwNDA0MDQwNDA0MDQwNCQsJCwgJCAkICRUXDA4VFwwOFhcN'
    'DhYXDQ4QEQwNEBEMDQsMCAgLDAgIEBEGCA4PBggTFAkKExQJCg0OBAYKDAgLDQ8EBwgJCAoJCQgKEA8FBwwODA4MDgwOEA8RBQAA'
    'AQAAAAEAABs/hYZfDzz1AAAIAAAAAAC2W95ZAAAAALZb39//LfvrB8YIqwAAAAMAAgABAAAAAAABAAAFP/0/AAAHq/8t/d8HxgAB'
    'AAAAAAAAAAAAAAAAAAAA2QQAAIAAAAAAAfQAAAH0AAABTQCaA34ARQJAAJwCQACbBBUAmwL/ADcBogCfAv8ANwFOAJsBpwArA8AB'
    'HAPAARMDwAB2A8AANQPAAGoDwABkA3gAUAPAAEwDwABLA8AApAFOAJsC/wA3AoAA1gKAAIcD1QEfA9UAlgNWAEYDVgAyAv8ASQOy'
    'AGkAJQAAACUAAAAlAAAAJf8tAesAfQHrAGkCigBGAooATgIAAEQCAABAAaIAnAGiAJwCvACbAmr/6QAAAGUAAAArAAAAmQAAAGUA'
    'AAAsAAAAmAAAACoAAACDAAAAewAAAEQAAAB4AAAAaQAAAGgAAAAvAAAASAAAADsAAAA9AAAAYgAAABwAAABHAAAAYgAA//4AAABH'
    'AAAAMgAAAE8AAAA0AAAARgAAAEYCegCeAfoAdgJdAHYBqwCCAh4AiwKTAFQDAABWAasApgIVALcFLgCbBP0AmwFN/+YCdv/mAS0A'
    'mwHgAJsFawCbBecAmwFN/+YCdv/mBWsAmwXnAJsBzP/mAnb/5gVrAJsF5wCbAbH/5gJ2/+YFawCbBecAmwGx/+YCdv/mBCUAmAR5'
    'AJgEDv/mBF//5gQlAJgEeQCYBA7/5gRf/+YEJQCYBHkAmAQO/+YEX//mBCUAmAR5AJgEDv/mBF//5gMOAJoDywCbAw4AmgPLAJsC'
    'hwBFAwIARQKHAEUDAgBFAocARQMCAEUG8ACbB4sAmwQR/+YEnP/mBvAAmweLAJsEEf/mBJz/5gdcAJsHqwCbBHf/5gTI/+YHXACb'
    'B6sAmwR3/+YEyP/mBTEAngWBAJ4D6v/mBDv/5gUxAJ4FgQCeA+r/5gQ7/+YDmACbBCUAmwLC/+YCz//mA5gAmwQlAJsCwv/mAs//'
    '5gVjAJoF0QCbAgn/5gK0/+YEhQCbBO8AmwIJ/+YCtP/mBlYAmwbHAJsC/f/mA3D/5gZZAJsGxwCbAv3/5gNi/+YETgCbBMwAmwFI'
    '/+YB2f/mAzMAlwQHAJcClv/mA5T/5gReAJsE8gCbAU3/5gJ2/+YCkwBUAwAAVgKBAJsDUgCbAyj/5gMb/+YCgQCbA1IAmwUuAJsE'
    '/QCbAcf/5gJ2/+YD3P/gBJL/4APcABkEkgAZA9wAUwSSAEwD3ABTBJIATAUuAJsE/QCbBYsAmwGVAFUAAAAAAAAAfgAAAH4AAAB+'
    'AAAAfgAAASwAAAHiAAACdAAAAwQAAARyAAAFCAAABY4AAAXoAAAGTAAABrQAAAcwAAAHpAAACEYAAAkuAAAKHgAACvYAAAu+AAAM'
    'aAAADRAAAA3KAAAOXAAADtgAAA9UAAAPzgAAEKYAABF+AAASCAAAEpAAABNgAAAUFgAAFBYAABQWAAAUvAAAFV4AABXwAAAWfAAA'
    'FzgAABf2AAAYWgAAGLwAABlGAAAZ+gAAGvQAABtOAAAb7gAAHNQAAB1yAAAd2AAAHpIAAB74AAAfuAAAIEQAACBWAAAgaAAAIHoA'
    'ACCMAAAgngAAILAAACDCAAAg1AAAIOwAACI4AAAjZAAAI3wAACRmAAAligAAJZwAACWuAAAlwAAAJdIAACXkAAAl9gAAJq4AACdy'
    'AAAoTAAAKT4AACpOAAArmAAALPoAAC3oAAAu9AAAMGoAADHQAAAyzgAAM9QAADRIAAA01AAANb4AADa4AAA3eAAAOEYAADmyAAA7'
    'LgAAPHAAAD2+AAA+6gAAQCYAAEEmAABCMAAAQ54AAEUcAABGXgAAR6oAAEjOAABKJgAASzIAAExwAABOFAAAT+4AAFF6AABTOAAA'
    'VBgAAFUuAABV9gAAVvAAAFgSAABZaAAAWnIAAFuuAABcVAAAXRoAAF4CAABfCgAAX5gAAGBMAABhHAAAYhIAAGNiAABk2gAAZhgA'
    'AGdYAABoLgAAaQAAAGsCAABtBAAAbp4AAHAyAABxVAAAcqoAAHN6AAB0cAAAddIAAHdoAAB4eAAAea4AAHrMAAB8DgAAfRoAAH5K'
    'AAB/qgAAgS4AAIJ8AACD7gAAhOAAAIW4AACGhAAAhzYAAIhqAACJqgAAis4AAIvMAACM+gAAjkQAAI9MAACQggAAkgwAAJO6AACU'
    '9gAAllQAAJdCAACYVAAAmRYAAJnuAACbHgAAnF4AAJ1eAACebAAAnzgAAKAaAACgwAAAoWAAAKI2AACjTAAApAgAAKTiAACl8AAA'
    'px4AAKfeAACoqAAAqXAAAKpSAACq8gAAq84AAKzMAACtxAAAruIAALA6AACxLgAAsgwAALMMAAC0GAAAtV4AALasAAC4HgAAuZgA'
    'ALsOAAC8hgAAvXwAAL56AAC/bgAAwEwAAMIWAADCegABAAAA2QB+AAkAVQADAAIACABAAAoAAACMAQYAAgACAAAAGAEmAAAAAAAA'
    'AAAALAAWAAAAAAAAAAEAEgBLAAAAAAAAAAIADgBkAAAAAAAAAAMAGgCaAAAAAAAAAAQAEgB7AAAAAAAAAAUAMgDNAAAAAAAAAAYA'
    'EAEHAAAAAAAAAAcAOAEzAAEAAAAAAAAAFgAAAAEAAAAAAAEACQBCAAEAAAAAAAIABwBdAAEAAAAAAAMADQCNAAEAAAAAAAQACQBy'
    'AAEAAAAAAAUAGQC0AAEAAAAAAAYACAD/AAEAAAAAAAcAHAEXAAMAAQQJAAAALAAWAAMAAQQJAAEAEgBLAAMAAQQJAAIADgBkAAMA'
    'AQQJAAMAGgCaAAMAAQQJAAQAEgB7AAMAAQQJAAUAMgDNAAMAAQQJAAYAEAEHAAMAAQQJAAcAOAEzKGMpIDIwMDAgQm9ybmEgUmF5'
    'YW5laAAoAGMAKQAgADIAMAAwADAAIABCAG8AcgBuAGEAIABSAGEAeQBhAG4AZQBoQiBOYXphbmluAEIAIABOAGEAegBhAG4AaQBu'
    'UmVndWxhcgBSAGUAZwB1AGwAYQByQiBOYXphbmluAEIAIABOAGEAegBhAG4AaQBuQm9ybmEgTmF6YW5pbgBCAG8AcgBuAGEAIABO'
    'AGEAegBhAG4AaQBuVmVyc2lvbiAyLjAxIC0gQnVpbGQgMTM3OQBWAGUAcgBzAGkAbwBuACAAMgAuADAAMQAgAC0AIABCAHUAaQBs'
    'AGQAIAAxADMANwA5Qk5hemFuaW4AQgBOAGEAegBhAG4AaQBuUGFyc2EgMjAwMaggLSBCb3JuYSBSYXlhbmVoqABQAGEAcgBzAGEA'
    'IAAyADAAMAAxAK4AIAAtACAAQgBvAHIAbgBhACAAUgBhAHkAYQBuAGUAaACuAAAAAAIAAAAAAAD9WAA8AAAAAAAAAAAAAAAAAAAA'
    'AAAAAAAA2QAAAAEAAgADAAQACAALAAwADQAOAA8AEAARABIAEwAUABUAFgAXABgAGQAaABsAHAAdACAAPgBAAF4AYACpAKoA8AC4'
    'AQIBAwEEAQUAtgC3ALQAtQC+AL8BBgEHAQgBCQEKAQsBDAENAQ4BDwEQAREBEgETARQBFQEWARcBGAEZARoBGwEcAR0BHgEfASAB'
    'IQEiASMBJAElASYBJwEoASkBKgErASwBLQEuAS8BMAExATIBMwE0ATUBNgE3ATgBOQE6ATsBPAE9AT4BPwFAAUEBQgFDAUQBRQFG'
    'AUcBSAFJAUoBSwFMAU0BTgFPAVABUQFSAVMBVAFVAVYBVwFYAVkBWgFbAVwBXQFeAV8BYAFhAWIBYwFkAWUBZgFnAWgBaQFqAWsB'
    'bAFtAW4BbwFwAXEBcgFzAXQBdQF2AXcBeAF5AXoBewF8AX0BfgF/AYABgQGCAYMBhAGFAYYBhwGIAYkBigGLAYwBjQGOAY8BkAGR'
    'AZIBkwGUAZUBlgGXAZgBmQGaAZsBnAGdAZ4BnwGgAaEBogGjAaQBpQGmAacBqAGpAaoBqwGsAa0BrgGvAbABsQDDCnplcm9ub2pv'
    'aW4IemVyb2pvaW4LbGVmdHRvcmlnaHQLcmlnaHR0b2xlZnQFdTA2MEMFdTA2MUIFdTA2MUYFdTA2NDAFdTA2NEIFdTA2NEMFdTA2'
    'NEQFdTA2NEUFdTA2NEYFdTA2NTAFdTA2NTEFdTA2NTIFdUU4MjAFdUU4MjEFdUU4MjIFdUU4MjMFdUU4MjQFdUU4MjUFdUU4MjYF'
    'dUU4MjcFdUU4MjgFdUU4MjkFdUU4MkEFdUU4MkIFdUU4MkMFdUU4MkQFdUZDNjAFdUZDNjEFdUU4MTgFdUZDNUUFdUZDNjIFdUZD'
    'NUYFdTA2MjEFdTA2MjILSGNpcmN1bWZsZXgFdTA2MjMFdUZFODQFdTA2MjQFdUZFODYFdTA2MjUFdUZFODgFdTA2MjYFdUZFOEEF'
    'dUZFOEIFdUZFOEMFdTA2MjcFdUZFOEUFdTA2MjgFdUZFOTAFdUZFOTEFdUZFOTIFdTA2N0UFdUZCNTcFdUZCNTgFdUZCNTkFdTA2'
    'MkEFdUZFOTYFdUZFOTcFdUZFOTgFdTA2MkIFdUZFOUEFdUZFOUIFdUZFOUMFdTA2MkMFdUZFOUUFdUZFOUYFdUZFQTAFdTA2ODYF'
    'dUZCN0IFdUZCN0MFdUZCN0QFdTA2MkQFdUZFQTIFdUZFQTMFdUZFQTQFdTA2MkUFdUZFQTYFdUZFQTcFdUZFQTgFdTA2MkYFdUZF'
    'QUEFdTA2MzAFdUZFQUMFdTA2MzEFdUZFQUUFdTA2MzIFdUZFQjAFdTA2OTgFdUZCOEIFdTA2MzMFdUZFQjIFdUZFQjMFdUZFQjQF'
    'dTA2MzQFdUZFQjYFdUZFQjcFdUZFQjgFdTA2MzUFdUZFQkEFdUZFQkIFdUZFQkMFdTA2MzYFdUZFQkUFdUZFQkYFdUZFQzAFdTA2'
    'MzcFdUZFQzIFdUZFQzMFdUZFQzQFdTA2MzgFdUZFQzYFdUZFQzcFdUZFQzgFdTA2MzkFdUZFQ0EFdUZFQ0IFdUZFQ0MFdTA2M0EF'
    'dUZFQ0UFdUZFQ0YFdUZFRDAFdTA2NDEFdUZFRDIFdUZFRDMFdUZFRDQFdTA2NDIFdUZFRDYFdUZFRDcFdUZFRDgFdTA2NDMFdUZF'
    'REEFdUZFREIFdUZFREMFdTA2QUYFdUZCOTMFdUZCOTQFdUZCOTUFdTA2NDQFdUZFREUFdUZFREYFdUZFRTAFdTA2NDUFdUZFRTIF'
    'dUZFRTMFdUZFRTQFdTA2NDYFdUZFRTYFdUZFRTcFdUZFRTgFdTA2NDgFdUZFRUUFdTA2NDcFdUZFRUEFdUZFRUIFdUZFRUMFdTA2'
    'MjkFdUZFOTQFdTA2NEEFdUZFRjIFdUZFRjMFdUZFRjQFdUZFRjUFdUZFRjYFdUZFRjcFdUZFRjgFdUZFRjkFdUZFRkEFdUZFRkIF'
    'dUZFRkMFdTA2NDkFdUZFRjAFdUZERjIAAEAjFBQTExISEREQEA8PDg4NDQwMCwsKCgkJCAgHBwYGBQUAAAGNuAH/hUVoREVoREVo'
    'REVoREVoREVoREVoREVoREVoREVoREVoREVoREVoREVoREVoREVoREVoRLMCAUYAK7MEA0YAK7EBAUVoRLEDA0VoRAAAAA=='
)

LOGO_B64 = (
    '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'
    'Aw2J4jYNESd76IT7S52+h/ywZZiy+pLU2O9prraRX6B59vEv/3nu97U+FiDvC91wjN8Xs9w5g5sg3321xq4SXBCuoo6iN7ku3Ze9'
    'o6x0llQCCoeRRbRqaipWt8PSNLPSUqpTCFc/YcilHVO+Dy6qn5QUF8JSKHNFh5NP+ip6I17MlENRsSNnKGCFoUQyvwgMvlb1usb0'
    'N7kl18VKK+Hvl5sA9g6ctm1x1INThlv2cdlEg0vuhaH8HKVQij1PJCNtSXfGocWEelgQN0DNg7UuhcW4Z87EQsPmRlGHsuK5LSbT'
    'ldDJo3U85/go+i/dwMHdweJH4kNSYrvST6WV09E2iKqxfZGrz1/94WmVnvG/3TfP7IJMTEZR+uLcXpL5XlxClDVMvfgqO5epM9L+'
    '0j6XE+W9o2y0Y522Pvel6O0e1YNXHgrgYNqlK/ZWbbLGVjaFz5mhVpltnn5MgytQ7O1OTjcec4aHjRbG6PacOwhrn0Xdfx0DIAzc'
    'Ayi/pDjmEQ8P/Fl1sZEyrTs+w7n9DwdwB91/cPtUagJyZErXH8Dj6YDwcVqEbch4f/As8l2zOMOZX52dYif7ZEBppmdNAGGlCtc5'
    'ML5i/KkUHNsgpreZcUEOTyyenQcNqonXM5KK1aQ5OsFWJmARURPkOz3huZ/cYZ2LShuyDT1GNrw5f1hQcfrAjZLsmI4r/LMRZgr6'
    'Tficy/aukLaGBqjThfvLTdxAO/LyxDxIxKqia3ZdCBatg3+FmR2IRlJnpiKnfB+oF7avM24vDGXI6obAgTescF16dYK54PpUtXMn'
    '0yR3lzwQ6hxn+y8CZmRLtr+1XV5cpzgffw10GfSPfsNSTtoukOrwKdILpYxVGtkXObcXr0hPY+5pfwvO3CdLBJ7X3/gOw3sut3/V'
    '/gDba23ZOmPscE71f6qsMkQDqEWR0wf3tVSSjuC4DQYifJJiNDGH/b+GzX70W6Ml7EEKdcOH7eUdu/OOg4XJGdjiEKdb0AIaBie+'
    'e4U4MRjpcOiyQW4eJhHTfn+ZTHRPwtvWqVIZGRyr/rgFpJ4VGPwXXcSwbiv+TdAycTcuoym4Dba6HGH83OtWTIsz0mSaJ/q0+X+j'
    'hcPAO8MBuu0UCEAdQEA+LWOr9SoepzKw4ushXZn+orY1Xta+cpdCiGQfXyc7baQ243a0sEbmh8c3Cgp2ZHNgxeA0dsPUMTlWW68m'
    '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'
)


if __name__ == '__main__':
    main()
