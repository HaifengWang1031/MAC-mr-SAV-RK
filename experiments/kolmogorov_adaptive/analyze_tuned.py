"""Six-panel figures for selected tuned runs and four fixed-step controls."""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.workflow import PROJECT, write_json
from experiments.kolmogorov_adaptive.analyze import aligned_output_times, read_member
from solver.mac.grid import MACGrid

def physical_config(config):
    return {key:config[key] for key in ('nx','ny','nu','m','epsilon','initial_modes',
                                         'gamma','T','output_every')}

def run(screen_path):
    screen=json.loads(screen_path.read_text())
    base=json.loads(Path(screen['base_batch']).read_text())
    if screen['status']!='complete' or len(screen['selected'])!=8 or len(screen['fixed_005'])!=4:
        raise ValueError('All eight tuned runs and four 0.005 controls must be complete')
    config=base['config']
    grid=MACGrid(config['nx'],config['ny'],2*np.pi,2*np.pi)
    ref_config,ref_manifest,reference=read_member(Path(base['reference']))
    if ref_manifest['status']!='complete' or physical_config(ref_config)!=physical_config(config):
        raise ValueError('Incompatible reference')
    reference_times=aligned_output_times(config,reference,'reference')
    reference_velocity=[grid.pack(u,v) for u,v in zip(reference['output_u'],reference['output_v'])]
    identity=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')
    folder=PROJECT/'reports/kolmogorov_adaptive'/f'tuned-{identity}'
    figures=folder/'figures'
    figures.mkdir(parents=True)
    report={'status':'running','screen':str(screen_path.resolve()),'reference':base['reference'],
            'mixed_source_note':'Reference and older fixed controls were produced before the output-alignment controller fix; fixed schemes do not use that controller.',
            'figures':[]}
    write_json(folder/'analysis.json',report)
    plt.rcParams.update({'font.family':'serif','mathtext.fontset':'cm','font.size':12})
    try:
        for scheme in ('sdirk2','sdirk2_mrsav','sdirk3','sdirk3_mrsav'):
            controls=[item for item in base['fixed_members'] if item['scheme']==scheme]
            controls += [item for item in screen['fixed_005'] if item['scheme']==scheme]
            if sorted(item['step'] if 'step' in item else .005 for item in controls) != [.0005,.001,.0025,.005]:
                raise ValueError(f'Incomplete fixed controls for {scheme}')
            controls.sort(key=lambda item:-(item['step'] if 'step' in item else .005))
            for name in ('I','PI'):
                member=screen['selected'][f'{scheme}-{name}']
                fig,panels=plt.subplots(2,3,figsize=(18,10))
                axes=panels.ravel()
                entries=[(f'{scheme} {name}',member['path'],'C0','-','o',True)]
                entries += [(rf'Fixed $\tau={item["step"] if "step" in item else .005:g}$',
                             item['path'],f'C{j}','--',marker,False)
                            for j,(item,marker) in enumerate(zip(controls,('*','d','s','^')),1)]
                for label,path,color,linestyle,marker,adaptive in entries:
                    actual,manifest,data=read_member(Path(path))
                    if manifest['status']!='complete' or physical_config(actual)!=physical_config(config):
                        raise ValueError(f'Incompatible or failed member: {path}')
                    t,h=data['attempt_t'],data['attempt_h']
                    accepted=data['attempt_accepted']
                    end_times=(t+h)[accepted]
                    style={'label':label,'color':color,'linestyle':linestyle,
                           'marker':marker,'markevery':max(1,len(t)//18),
                           'markersize':3.5,'linewidth':1.1}
                    axes[0].step(np.r_[0.,end_times],np.r_[h[accepted][0],h[accepted]],where='post',**style)
                    axes[1].plot(np.r_[0.,end_times],np.arange(len(end_times)+1),**style)
                    axes[2].plot(np.r_[0.,end_times],np.r_[0.,data['attempt_cpu'][accepted]],**style)
                    if 'mrsav' in scheme:
                        axes[3].plot(data['times'],np.abs(data['diagnostic_r']),**style)
                    if adaptive:
                        valid=np.isfinite(data['attempt_error']) & (data['attempt_error']>0)
                        axes[4].semilogy((t+h)[valid & accepted],data['attempt_error'][valid & accepted],**style)
                        axes[4].scatter((t+h)[valid & ~accepted],data['attempt_error'][valid & ~accepted],
                                        color=color,marker='x',s=20)
                    output_nodes=aligned_output_times(config,data,path)
                    if not np.allclose(output_nodes,reference_times,atol=1e-8,rtol=0):
                        raise ValueError(f'Output times differ from reference: {path}')
                    errors=[grid.norm(grid.pack(u,v)-target)/max(grid.norm(target),1e-14)
                            for u,v,target in zip(data['output_u'],data['output_v'],reference_velocity)]
                    axes[5].semilogy(output_nodes,errors,**style)
                for j,ylabel in enumerate(('Step sizes','Step count','CPU time (s)',r'$|r|$',
                                            r'$L^2$ embedded error / tolerance',
                                            r'Relative $L^2$ error of velocity')):
                    axes[j].set(xlabel=r'$t$',ylabel=ylabel)
                    axes[j].grid(True,color='0.8',linestyle='--',linewidth=.6)
                    axes[j].text(.5,-.30,f'({chr(ord("a")+j)})',transform=axes[j].transAxes,
                                 ha='center',va='top',fontsize=15)
                axes[0].set_yscale('log')
                axes[4].axhline(1.,color='0.35',linestyle='--',linewidth=.8)
                if 'mrsav' not in scheme:
                    axes[3].set_axis_off()
                    axes[3].text(.5,.5,'No auxiliary variable',
                                 transform=axes[3].transAxes,ha='center',va='center',
                                 color='0.4',fontsize=14)
                handles,labels=axes[0].get_legend_handles_labels()
                fig.legend(handles,labels,loc='lower center',ncol=5,fontsize=10,
                           bbox_to_anchor=(.5,.01))
                fig.subplots_adjust(left=.08,right=.97,bottom=.16,top=.98,wspace=.34,hspace=.52)
                stem=f'{scheme}-{name}'
                fig.savefig(figures/f'{stem}.pdf',bbox_inches='tight')
                fig.savefig(figures/f'{stem}.png',dpi=180,bbox_inches='tight')
                plt.close(fig)
                report['figures'].append(stem)
                write_json(folder/'analysis.json',report)
        report['status']='complete'
    except Exception as exc:
        report.update(status='failed',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        write_json(folder/'analysis.json',report)
    print(folder)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--screen',type=Path,required=True)
    run(parser.parse_args().screen)
