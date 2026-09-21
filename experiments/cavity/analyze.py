"""Plot stored cavity runs; never starts a computation."""
import sys
from pathlib import Path
if __package__ in (None,''): sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import argparse
import csv
import json
from datetime import datetime,timezone
from uuid import uuid4
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.workflow import PROJECT,load_record,write_json,provenance


def analyze(inputs: list[Path]) -> Path:
    if not inputs: raise ValueError('Explicit runs required')
    records=[]
    for path in inputs:
        cfg,manifest=load_record(path,require_complete=True)
        if cfg['experiment']!='cavity': raise ValueError('Cavity runs required')
        with h5py.File(path/'results.h5') as data:
            records.append((cfg,manifest,data['final/u'][:],data['final/v'][:],float(data['final/t'][()])))
    for cfg,_,_,_,_ in records:
        if any(cfg[k]!=records[0][0][k] for k in ('lx','ly','nu','lid_speed','T','scheme')):
            raise ValueError('Different physical parameters in grid comparison')
    report=PROJECT/'reports/cavity'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'-'+uuid4().hex[:8])
    (report/'figures').mkdir(parents=True); (report/'tables').mkdir()
    metadata={'status':'running','source':provenance(),
              'inputs':[{'run_id':r[1]['run_id'],'results_sha256':r[1]['results_sha256'],'config_sha256':r[1]['config_sha256']} for r in records],
              'parameters':{'vorticity':'interior MAC vertex curl','streamlines':'discrete streamfunction contours','color_clip_percentile':98}}
    write_json(report/'analysis.json',metadata)
    figures=[]
    try:
        plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
        cfg,manifest,u,v,t=max(records,key=lambda r:r[0]['nx']*r[0]['ny'])
        nx,ny=cfg['nx'],cfg['ny']; hx=cfg['lx']/nx; hy=cfg['ly']/ny
        xc=(np.arange(nx)+.5)*hx; yc=(np.arange(ny)+.5)*hy
        uc=.5*(u[:,:-1]+u[:,1:]); vc=.5*(v[:-1,:]+v[1:,:])
        speed=np.hypot(uc,vc)
        psi=np.vstack([np.zeros(nx+1),np.cumsum(u,axis=0)*hy])
        omega=(v[1:-1,1:]-v[1:-1,:-1])/hx-(u[1:,1:-1]-u[:-1,1:-1])/hy
        fig,axes=plt.subplots(1,2,figsize=(12,5.3),layout='constrained'); figures.append(fig)
        im=axes[0].pcolormesh(np.arange(nx+1)*hx,np.arange(ny+1)*hy,speed/cfg['lid_speed'],cmap='viridis',vmin=0,vmax=1,shading='flat')
        levels=np.sort(-np.geomspace(abs(psi.min())*.98,1e-5,14))
        if psi.max()>1e-6: levels=np.r_[levels,np.geomspace(1e-6,psi.max()*.98,4)]
        axes[0].contour(np.arange(nx+1)*hx,np.arange(ny+1)*hy,psi,levels=levels,colors='white',linewidths=.65,linestyles='solid')
        stride=max(1,nx//8)
        axes[0].quiver(xc[::stride],yc[::stride],uc[::stride,::stride],vc[::stride,::stride],color='white',scale=8,width=.003)
        j,i=np.unravel_index(np.argmin(psi),psi.shape)
        axes[0].plot(i*hx,j*hy,'o',color='#ffb347',ms=5)
        axes[0].set_title('Speed and streamlines',pad=28)
        fig.colorbar(im,ax=axes[0],label='Speed / lid speed',shrink=.83)
        limit=float(np.percentile(np.abs(omega),98))
        im2=axes[1].pcolormesh(np.arange(1,nx)*hx,np.arange(1,ny)*hy,omega,cmap='RdBu_r',vmin=-limit,vmax=limit,shading='auto')
        axes[1].set_title('Interior vorticity',pad=28)
        fig.colorbar(im2,ax=axes[1],label=r'$\omega=\partial_x v-\partial_y u$',extend='both',shrink=.83)
        for ax in axes:
            ax.set(xlim=(0,cfg['lx']),ylim=(0,cfg['ly']),xlabel='x',ylabel='y',aspect='equal')
            ax.annotate('',xy=(.85*cfg['lx'],1.025*cfg['ly']),xytext=(.15*cfg['lx'],1.025*cfg['ly']),arrowprops={'arrowstyle':'->','color':'#c24b2d','lw':2},annotation_clip=False)
        fig.suptitle(f"Lid-driven cavity | Re={manifest['metrics']['reynolds']:g} | {nx} x {ny} MAC | t={t:.0f}\nSDIRK2-mr-ccSAV",fontsize=14)
        for ext in ('png','pdf'): fig.savefig(report/f'figures/cavity_fields.{ext}',dpi=200)
        summary={'main_vortex_vertex':[float(i*hx),float(j*hy)],'streamfunction_min':float(psi.min()),
                 'vorticity_color_limit':limit,'steady_rhs_l2':manifest['metrics']['steady_rhs_l2'],
                 'maximum_divergence_final':float(np.max(np.abs(np.diff(u,axis=1)/hx+np.diff(v,axis=0)/hy)))}
        fig2,axs=plt.subplots(1,2,figsize=(10,4.4),layout='constrained'); figures.append(fig2)
        profiles=[]
        for c,m,uu,vv,tt in sorted(records,key=lambda r:r[0]['nx']):
            dx=c['lx']/c['nx']; dy=c['ly']/c['ny']
            y=np.r_[0,(np.arange(c['ny'])+.5)*dy,c['ly']]
            x=np.r_[0,(np.arange(c['nx'])+.5)*dx,c['lx']]
            midu=np.r_[0,[np.interp(c['lx']/2,np.arange(c['nx']+1)*dx,row) for row in uu],c['lid_speed']]
            midv=np.r_[0,[np.interp(c['ly']/2,np.arange(c['ny']+1)*dy,col) for col in vv.T],0]
            label=f"{c['nx']} x {c['ny']}"
            axs[0].plot(midu,y,label=label); axs[1].plot(x,midv,label=label)
            profiles.append((x,y,midu,midv))
            with (report/f"tables/centerlines_{c['nx']}x{c['ny']}.csv").open('w',newline='') as stream:
                writer=csv.writer(stream);writer.writerow(['component','coordinate','velocity'])
                writer.writerows(('u',yy,a) for yy,a in zip(y,midu));writer.writerows(('v',xx,b) for xx,b in zip(x,midv))
        if len(profiles)==2:
            coarse,fine=profiles
            summary['centerline_grid_difference_u_max']=float(np.max(np.abs(coarse[2]-np.interp(coarse[1],fine[1],fine[2]))))
            summary['centerline_grid_difference_v_max']=float(np.max(np.abs(coarse[3]-np.interp(coarse[0],fine[0],fine[3]))))
        axs[0].set(xlabel='u(x=L/2, y)',ylabel='y');axs[1].set(xlabel='x',ylabel='v(x, y=H/2)')
        for ax in axs: ax.grid(alpha=.25);ax.legend()
        fig2.suptitle('Centerline profiles: grid sensitivity (not a reference-data validation)')
        fig2.savefig(report/'figures/centerlines.png',dpi=180)
        write_json(report/'tables/summary.json',summary)
        metadata.update(status='complete',summary=summary)
    except Exception as error:
        metadata.update(status='failed',error=f'{type(error).__name__}: {error}')
        raise
    finally:
        for fig in figures: plt.close(fig)
        write_json(report/'analysis.json',metadata)
        (report/'analysis.log').write_text(json.dumps(metadata,indent=2)+'\n')
    return report

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--runs',nargs='+',type=Path,required=True)
    print(analyze(parser.parse_args().runs))
