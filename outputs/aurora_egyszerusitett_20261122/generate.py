from pathlib import Path
import os, sys, ast
sys.path.append(str(Path(os.environ['TEMP'])/'codex_aurora_plot_deps'))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parents[2]
base=root.parents[1]
source=base/'Kódok_Excel'/'calculate_azimuth_tracking.py'
tree=ast.parse(source.read_text(encoding='utf-8-sig'))
g={'np':np,'D_OFFSETS':[500,750,1000,1250,1500]}
names={'calculate_3d_tracking','gaussian_smooth'}
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names],type_ignores=[]),str(source),'exec'),g)
f=pd.read_csv(base/'Források'/'Repülési adatok'/'Aurora_EuroC_CATS.csv')
f=f[f.link==1].drop_duplicates('ts[deciseconds]').sort_values('ts[deciseconds]')
tr=(f['ts[deciseconds]'].to_numpy()-f['ts[deciseconds]'].iloc[0])/10
lat=np.radians(f['lat[deg/10000]'].to_numpy()/10000)
lon=np.radians(f['lon[deg/10000]'].to_numpy()/10000)
en=np.array([6378137*(lon-lon[0])*np.cos(lat[0]),6378137*(lat-lat[0])])
ids=np.where((tr>=61)&(tr<=72.2))[0]
i,j=ids[0],ids[-1]
u=en[:,i]/np.linalg.norm(en[:,i])
q=np.array([u,[-u[1],u[0]]])@en
v=np.polyfit(tr[ids],q[:,ids].T,1)[0]
rec=q.copy()
before=tr<tr[i]; after=tr>tr[j]
tau=tr[before]/tr[i]
rec[:,before]=q[:,i,None]*(3*tau**2-2*tau**3)
rec[:,after]=q[:,j,None]+v[:,None]*(tr[None,after]-tr[j])
t=np.arange(0,tr[-1]+.05,.1)
z0=f['altitude[m]'].to_numpy()
z=g['gaussian_smooth'](np.interp(t,tr,z0-z0[0]))
vz=np.gradient(z,.1)
xy=np.array([g['gaussian_smooth'](np.interp(t,tr,r),window_len=25,sigma=6.) for r in rec])
vel=np.gradient(xy,.1,axis=1)
res=g['calculate_3d_tracking'](t,xy[0],xy[1],z,vel[0],vel[1],vz)
out=Path(__file__).parent.parent/'aurora_harmadfoku'
out.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.titlesize':13})
def save(fig,name):
    if '--horizontal-only' in sys.argv:
        if 'horizontal_drift' not in name:
            plt.close(fig)
            return
        name=name.replace('horizontal_drift','horizontal_drift_EN')
    fig.savefig(out/name,dpi=180,bbox_inches='tight')
    plt.close(fig)
    print(name)
time_label='Az első felhasznált mintától eltelt idő (s)'
fig,ax=plt.subplots(figsize=(10,5),layout='constrained')
ax.plot(t,z,color='#176c9c',label=f'Magasság, maximum: {max(z):.0f} m')
ax.set(xlabel=time_label,ylabel='Relatív magasság (m)',title='Aurora – EuRoC 2025: feldolgozott függőleges mozgás')
ax.grid(alpha=.25)
a2=ax.twinx()
a2.plot(t,vz,color='#c05032',ls='--',label=f'Függőleges sebesség, maximum: {max(vz):.1f} m/s')
a2.set_ylabel('Függőleges sebesség (m/s)')
lines=list(ax.lines)+list(a2.lines)
ax.legend(lines,[l.get_label() for l in lines],loc='upper right')
save(fig,'aurora_euroc_cats_flight_profile.png')
en_smooth=np.array([u,[-u[1],u[0]]]).T @ xy
fig,ax=plt.subplots(figsize=(10,5.2),layout='constrained')
for n,label in enumerate(['Keleti elmozdulás, E','Északi elmozdulás, N']):
    extreme=en_smooth[n,np.argmax(np.abs(en_smooth[n]))]
    ax.plot(t,en_smooth[n],color=['#ff7f0e','#2ca02c'][n],lw=2,
            label=f'{label} (szélsőérték: {extreme:+.1f} m)')
ax.set(title='Aurora – EuRoC 2025: becsült vízszintes mozgás',
       xlabel=time_label,ylabel='Elmozdulás a referenciahelyhez képest (m)')
ax.grid(alpha=.3,linestyle='--')
ax.legend(fontsize=10,loc='best')
save(fig,'aurora_euroc_cats_horizontal_drift.png')
for axis,name in [('elevation','elev'),('azimuth','azimuth')]:
    fig,axs=plt.subplots(3,1,figsize=(10,10),sharex=True,layout='constrained')
    fig.suptitle('Aurora – EuRoC 2025: '+('elevációkövetés' if axis=='elevation' else 'azimutkövetés')+' a becsült pályán')
    for D,r in res.items():
        for ax,key,unit in zip(axs,[axis+'_deg',axis+'_rate',axis+'_accel'],['°','°/s','°/s²']):
            ar=r[key]-(r[key][0] if key=='azimuth_deg' else 0)
            ax.plot(t,ar,label=f'D = {D} m; max. |érték| = {max(abs(ar)):.3f} {unit}',lw=1.5)
    for ax,label in zip(axs,['Szög (°)','Szögsebesség (°/s)','Szöggyorsulás (°/s²)']):
        ax.set_ylabel(label)
        ax.grid(alpha=.25)
        ax.legend(fontsize=8,loc='best')
    axs[-1].set_xlabel(time_label)
    save(fig,f'aurora_euroc_cats_{name}_tracking.png')
